use std::{io::Write, path::Path, sync::Arc};

use reqwest::Client;
use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::Semaphore,
    task::JoinSet,
};

use crate::util::{DynError, MAX_CONCURRENT_DOWNLOADS, log_progress, log_step};

pub async fn download_fits_files(
    base_url: &str,
    filenames: &[String],
    data_dir: &Path,
) -> Result<(), DynError> {
    let (mut downloaded, mut skipped, mut failed) =
        download_fits_files_pass(base_url, filenames, "Starting downloads", data_dir).await?;

    while !failed.is_empty() {
        log_step(&format!("{} files failed to download.", failed.len()));

        while !prompt_retry_failed(failed.len()).await? {
            log_step("Retry required to complete all downloads.");
        }

        let (retry_downloaded, retry_skipped, retry_failed) =
            download_fits_files_pass(base_url, &failed, "Retrying failed downloads", data_dir)
                .await?;
        downloaded += retry_downloaded;
        skipped += retry_skipped;
        failed = retry_failed;
    }

    log_step(&format!(
        "Downloaded {} files, skipped {} existing files",
        downloaded, skipped
    ));

    Ok(())
}

pub async fn ensure_all_files_are_present(
    filenames: &[String],
    data_dir: &Path,
) -> Result<(), DynError> {
    let mut missing_files = Vec::new();

    for filename in filenames {
        let output_path = data_dir.join(filename);
        if !fs::try_exists(&output_path).await? {
            missing_files.push(filename.clone())
        }
    }

    if missing_files.is_empty() {
        log_step(&format!(
            "Verified {} FITS files in {}",
            filenames.len(),
            data_dir.display()
        ));
        Ok(())
    } else {
        Err(format!(
            "Missing {} FITS files in {}: {:?}",
            missing_files.len(),
            data_dir.display(),
            missing_files
        )
        .into())
    }
}

pub async fn download_fits_file(
    client: &Client,
    base_url: &str,
    filename: &str,
    data_dir: &Path,
) -> Result<bool, DynError> {
    let output_path = data_dir.join(filename);

    if fs::try_exists(&output_path).await? {
        println!("\nSkipping existing {}", filename);
        return Ok(false);
    }

    let download_url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
    let temp_path = data_dir.join(format!("{}.part", filename));

    if let Some(parent) = temp_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let response = client.get(&download_url).send().await?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to download {}: HTTP {}",
            filename,
            response.status()
        )
        .into());
    }

    let bytes = response.bytes().await?;
    let mut file = fs::File::create(&temp_path).await?;
    file.write_all(&bytes).await?;
    file.flush().await?;
    fs::rename(&temp_path, &output_path).await?;

    Ok(true)
}

pub async fn prompt_retry_failed(failed_count: usize) -> Result<bool, DynError> {
    print!(
        "{} downloads failed. Retry failed files? [y/N]: ",
        failed_count
    );
    std::io::stdout().flush()?;

    let mut line = String::new();
    let mut reader = BufReader::new(tokio::io::stdin());
    reader.read_line(&mut line).await?;
    let answer = line.trim().to_ascii_lowercase();
    Ok(matches!(answer.as_str(), "y" | "yes"))
}

pub async fn download_fits_files_pass(
    base_url: &str,
    filenames: &[String],
    label: &str,
    data_dir: &Path,
) -> Result<(usize, usize, Vec<String>), DynError> {
    if filenames.is_empty() {
        return Ok((0, 0, Vec::new()));
    }

    let data_dir = data_dir.to_path_buf();
    let client = Client::new();
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_DOWNLOADS));
    let mut join_set = JoinSet::new();

    log_step(&format!(
        "{} (max {} concurrent connections)...",
        label, MAX_CONCURRENT_DOWNLOADS
    ));
    log_step(&format!("Total files to download: {}", filenames.len()));

    for filename in filenames.iter() {
        let filename = filename.clone();
        let client = client.clone();
        let semaphore = Arc::clone(&semaphore);
        let base_url = base_url.to_string();
        let data_dir = data_dir.clone();

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|error| -> DynError { Box::new(error) })?;

            let result = download_fits_file(&client, &base_url, &filename, &data_dir).await;
            Ok::<_, DynError>((filename, result))
        });
    }

    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut completed = 0usize;
    let mut failed = Vec::new();

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok((_, Ok(true)))) => downloaded += 1,
            Ok(Ok((_, Ok(false)))) => skipped += 1,
            Ok(Ok((filename, Err(error)))) => {
                failed.push(filename);
                eprintln!("Download failed: {}", error);
            }
            Ok(Err(error)) => eprintln!("Task error: {}", error),
            Err(error) => eprintln!("Task join error: {}", error),
        }

        completed += 1;
        log_progress("Downloading", completed, filenames.len())?;
    }

    println!();
    Ok((downloaded, skipped, failed))
}

pub async fn prompt_group_selection(
    groups: &[(String, Vec<String>)],
) -> Result<Vec<usize>, DynError> {
    println!("\nAvailable datasets:");
    for (i, (group_name, files)) in groups.iter().enumerate() {
        println!("  {:3}: {} ({} files)", i + 1, group_name, files.len());
    }
    print!("\nSelect dataset(s) to download (e.g. \"1,3\", \"1-3\", or \"all\"): ");
    std::io::stdout().flush()?;

    let mut line = String::new();
    let mut reader = BufReader::new(tokio::io::stdin());
    reader.read_line(&mut line).await?;
    let answer = line.trim();

    if answer.eq_ignore_ascii_case("all") {
        return Ok((0..groups.len()).collect());
    }

    let mut indices = Vec::new();
    for part in answer.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let start: usize = start
                .trim()
                .parse()
                .map_err(|e| format!("Invalid range: {}", e))?;
            let end: usize = end
                .trim()
                .parse()
                .map_err(|e| format!("Invalid range: {}", e))?;
            for i in start..=end {
                if i > 0 && i <= groups.len() {
                    indices.push(i - 1);
                }
            }
        } else if let Ok(num) = part.parse::<usize>() {
            if num > 0 && num <= groups.len() {
                indices.push(num - 1);
            }
        }
    }
    indices.sort();
    indices.dedup();
    Ok(indices)
}

pub fn extract_fits_files(html: &str) -> Vec<String> {
    html.lines()
        .filter_map(|line| {
            let href_start = line.find("href=\"")? + "href=\"".len();
            let href_end = line[href_start..].find('"')? + href_start;
            let filename = &line[href_start..href_end];
            if filename.ends_with(".fits") {
                Some(filename.to_string())
            } else {
                None
            }
        })
        .collect()
}

pub async fn fetch_directory_listing(
    client: &reqwest::Client,
    url: &str,
) -> Result<String, DynError> {
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("Failed to fetch data: HTTP {}", response.status()).into());
    }

    let bytes = response.bytes().await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
