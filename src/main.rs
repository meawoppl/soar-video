use clap::Parser;
use fitsio::{FitsFile, hdu::HduInfo, sys::stdout};
use image::{GrayImage, io};
use reqwest::Client;
use std::{
    error::Error,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs,
    io::AsyncWriteExt,
    process::Command,
    sync::Semaphore,
    task::{JoinSet, spawn_blocking},
};

const BASE_URL: &str = "https://www.sidc.be/EUI/data/L2";
const DATA_DIR: &str = "data";
const MAX_CONCURRENT_DOWNLOADS: usize = 4;

type DynError = Box<dyn Error + Send + Sync>;

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Day
    #[arg(short, long)]
    day: u8,
    /// Month
    #[arg(short, long)]
    month: u8,
    /// Year
    #[arg(short, long)]
    year: u16,
}

fn find_image_hdu(file: &mut FitsFile) -> Result<(usize, Vec<usize>), DynError> {
    for hdu_index in 0..file.num_hdus()? {
        let hdu = file.hdu(hdu_index)?;
        if let HduInfo::ImageInfo { shape, .. } = &hdu.info {
            if !shape.is_empty() {
                return Ok((hdu_index, shape.clone()));
            }
        }
    }
    Err("No image HDU found".into())
}

fn extract_hrieuv174_fits_files(html: &str) -> Vec<String> {
    html.lines()
        .filter_map(|line| {
            let href_start = line.find("href=\"")? + "href=\"".len();
            let href_end = line[href_start..].find('"')? + href_start;
            let filename = &line[href_start..href_end];

            if filename.contains("hrieuv174") && filename.ends_with(".fits") {
                Some(filename.to_string())
            } else {
                None
            }
        })
        .collect()
}

async fn download_hrieuv174_fits_file(
    client: &Client,
    base_url: &str,
    filename: &str,
) -> Result<bool, DynError> {
    let output_path = Path::new(DATA_DIR).join(filename);

    if fs::try_exists(&output_path).await? {
        println!("Skipping existing {}", filename);
        return Ok(false);
    }

    let download_url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
    let temp_path = Path::new(DATA_DIR).join(format!("{}.part", filename));

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

    println!("Downloaded {}", filename);
    Ok(true)
}

async fn download_hrieuv174_fits_files(
    base_url: &str,
    filenames: &[String],
) -> Result<(), DynError> {
    fs::create_dir_all(DATA_DIR).await?;

    let client = Client::new();
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_DOWNLOADS));
    let mut join_set = JoinSet::new();

    for filename in filenames {
        let filename = filename.clone();
        let client = client.clone();
        let semaphore = Arc::clone(&semaphore);
        let base_url = base_url.to_string();

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|error| -> DynError { Box::new(error) })?;

            download_hrieuv174_fits_file(&client, &base_url, &filename).await
        });
    }

    let mut downloaded = 0usize;
    let mut skipped = 0usize;

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(true)) => downloaded += 1,
            Ok(Ok(false)) => skipped += 1,
            Ok(Err(error)) => eprintln!("Download failed: {}", error),
            Err(error) => eprintln!("Task join error: {}", error),
        }
    }

    println!(
        "Downloaded {} files, skipped {} existing files",
        downloaded, skipped
    );

    Ok(())
}

async fn ensure_all_files_are_present(filenames: &[String]) -> Result<(), DynError> {
    let mut missing_files = Vec::new();

    for filename in filenames {
        let output_path = Path::new(DATA_DIR).join(filename);
        if !fs::try_exists(&output_path).await? {
            missing_files.push(filename.clone());
        }
    }

    if missing_files.is_empty() {
        println!("Verified {} FITS files in {}", filenames.len(), DATA_DIR);
        Ok(())
    } else {
        Err(format!(
            "Missing {} FITS files in {}: {:?}",
            missing_files.len(),
            DATA_DIR,
            missing_files
        )
        .into())
    }
}
async fn generate_video_from_fits_files(
    filenames: &[String],
    output_path: &Path,
) -> Result<(), DynError> {
    let frame_dir = output_path.with_extension("");
    if fs::try_exists(&frame_dir).await? {
        fs::remove_dir_all(&frame_dir).await?;
    }
    fs::create_dir_all(&frame_dir).await?;

    println!("Computing global intensity range...");
    let filenames_for_range = filenames.to_vec();
    let (global_min, global_max) =
        spawn_blocking(move || compute_global_range(&filenames_for_range, 1.0, 99.0)).await??;
    println!("Global range: {} to {}", global_min, global_max);

    println!("Converting FITS files to PNG frames...");
    for (index, filename) in filenames.iter().enumerate() {
        print!(
            "\rProcessing frame {}/{}: {}",
            index + 1,
            filenames.len(),
            filename
        );
        std::io::stdout().flush()?;

        let fits_path = Path::new(DATA_DIR).join(filename);
        let frame_path = frame_dir.join(format!("frame_{:05}.png", index));
        let (min, max) = (global_min, global_max);

        spawn_blocking(move || fits_to_png_frame(&fits_path, &frame_path, min, max)).await??;
    }

    println!("\nAll frames processed, generating video...");

    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg("10")
        .arg("-i")
        .arg(frame_dir.join("frame_%05d.png"))
        .arg("-c:v")
        .arg("libx264")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(output_path)
        .status()
        .await?;

    if !status.success() {
        return Err(format!("ffmpeg failed for {}", output_path.display()).into());
    }

    println!("Generated video at {}", output_path.display());
    Ok(())
}

fn compute_global_range(
    filenames: &[String],
    p_low: f64,
    p_high: f64,
) -> Result<(f64, f64), DynError> {
    let mut lows = Vec::with_capacity(filenames.len());
    let mut highs = Vec::with_capacity(filenames.len());

    println!("Computing per-frame intensity percentiles...");
    for (index, filename) in filenames.iter().enumerate() {
        print!(
            "\rProcessing {}/{}: {}",
            index + 1,
            filenames.len(),
            filename
        );
        let path = Path::new(DATA_DIR).join(filename);
        let mut file = FitsFile::open(&path)?;
        let (hdu_index, _) = find_image_hdu(&mut file)?;
        let hdu = file.hdu(hdu_index)?;
        let pixels: Vec<f32> = hdu.read_image(&mut file)?;

        let mut finite: Vec<f32> = pixels
            .into_iter()
            .filter(|value| value.is_finite())
            .collect();
        if finite.is_empty() {
            continue;
        }
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = finite.len();
        let idx_low = ((n - 1) as f64 * p_low / 100.0).round() as usize;
        let idx_high = ((n - 1) as f64 * p_high / 100.0).round() as usize;

        lows.push(finite[idx_low.clamp(0, n - 1)] as f64);
        highs.push(finite[idx_high.clamp(0, n - 1)] as f64);
    }

    if lows.is_empty() {
        return Ok((0.0, 1.0));
    }

    lows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    highs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_min = lows[lows.len() / 2];
    let median_max = highs[highs.len() / 2];
    println!(
        "\nMedian 1st percentile: {}, Median 99th percentile: {}",
        median_min, median_max
    );
    Ok((median_min, median_max))
}

fn fits_to_png_frame(
    fits_path: &Path,
    png_path: &Path,
    global_min: f64,
    global_max: f64,
) -> Result<(), DynError> {
    let mut fits_file = FitsFile::open(fits_path)?;
    let (hdu_index, shape) = find_image_hdu(&mut fits_file)?;

    if shape.len() != 2 {
        return Err(format!("Expected 2D image, got shape {:?}", shape).into());
    }

    // Verify axis order: FITS NAXIS1 is usually width. If fitsio returns
    // shape = [NAXIS1, NAXIS2], then width = shape[0], height = shape[1].
    // Your original code swapped these; if your images are non-square,
    // they were likely sheared, not just rotated.
    let width = shape[0] as u32;
    let height = shape[1] as u32;

    let hdu = fits_file.hdu(hdu_index)?;
    let pixels: Vec<f32> = hdu.read_image(&mut fits_file)?;

    let offset = if global_min > 0.0 {
        0.0
    } else {
        global_min.abs() + 1.0
    };

    let log_min = (global_min + offset).ln();
    let log_max = (global_max + offset).ln();
    let log_range = (log_max - log_min).max(f64::EPSILON);

    let bytes: Vec<u8> = pixels
        .into_iter()
        .map(|v| {
            if !v.is_finite() {
                return 0;
            }
            let v = (v as f64).clamp(global_min, global_max);
            let shifted = v + offset;
            let log_val = shifted.ln();
            let normalized = (log_val - log_min) / log_range;
            (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();

    let image = GrayImage::from_raw(width, height, bytes).ok_or_else(|| {
        format!(
            "Buffer size mismatch for {} ({}x{})",
            fits_path.display(),
            width,
            height
        )
    })?;

    image.save(png_path)?;
    Ok(())
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let url = format!(
        "{}/{}/{:02}/{:02}/",
        BASE_URL, args.year, args.month, args.day
    );
    println!("Fetching data from: {}", url);

    let client = reqwest::Client::builder()
        .user_agent("sun/0.1.0")
        .build()
        .expect("failed to build HTTP client");

    match client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                match response.bytes().await {
                    Ok(bytes) => {
                        let html = String::from_utf8_lossy(&bytes);
                        let fits_files = extract_hrieuv174_fits_files(&html);
                        println!("Found {} FITS files", fits_files.len());

                        if let Err(error) = download_hrieuv174_fits_files(&url, &fits_files).await {
                            eprintln!("Error downloading files: {}", error);
                            return;
                        }

                        if let Err(error) = ensure_all_files_are_present(&fits_files).await {
                            eprintln!("Data verification failed: {}", error);
                            return;
                        }

                        let video_name =
                            format!("{}-{:02}-{:02}.mp4", args.year, args.month, args.day);
                        let video_path = PathBuf::from(DATA_DIR).join(video_name);

                        if let Err(error) =
                            generate_video_from_fits_files(&fits_files, &video_path).await
                        {
                            eprintln!("Error generating video: {}", error);
                        }
                    }
                    Err(e) => eprintln!("Error reading response bytes: {}", e),
                }
            } else {
                eprintln!("Failed to fetch data: HTTP {}", response.status());
            }
        }
        Err(e) => eprintln!("Error making request: {}", e),
    }
}
