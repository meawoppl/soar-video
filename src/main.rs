use clap::Parser;
use fitsio::{FitsFile, hdu::HduInfo};
use image::GrayImage;
use reqwest::Client;
use std::{
    collections::BTreeMap,
    error::Error,
    io::Write,
    path::Path,
    sync::Arc,
};
use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::Semaphore,
    task::{JoinSet, spawn_blocking},
};

const BASE_URL: &str = "https://www.sidc.be/EUI/data/L1";
const DATA_DIR: &str = "data";
const MAX_CONCURRENT_DOWNLOADS: usize = 4;

const CLAHE_TILES_X: usize = 8;
const CLAHE_TILES_Y: usize = 8;
const CLAHE_CLIP_LIMIT: f64 = 3.0;

type DynError = Box<dyn Error + Send + Sync>;

fn log_step(message: &str) {
    println!("{message}");
}

fn log_progress(prefix: &str, current: usize, total: usize) -> Result<(), DynError> {
    print!("\r{} {}/{}", prefix, current, total);
    std::io::stdout().flush()?;
    Ok(())
}

#[derive(Clone, Copy)]
struct AsinhMapping {
    min: f64,
    max: f64,
    beta: f64,
    inv_asinh_range: f64,
}

impl AsinhMapping {
    fn from_range(global_min: f64, global_max: f64) -> Self {
        let range = (global_max - global_min).max(f64::EPSILON);
        let beta = 0.02 * range;
        let inv_asinh_range = ((range / beta).asinh()).max(f64::EPSILON).recip();
        Self {
            min: global_min,
            max: global_max,
            beta,
            inv_asinh_range,
        }
    }

    #[inline]
    fn apply(&self, value: f64) -> f64 {
        let clamped = value.clamp(self.min, self.max);
        let shifted = clamped - self.min;
        ((shifted / self.beta).asinh() * self.inv_asinh_range).clamp(0.0, 1.0)
    }
}

struct ClaheLut {
    luts: Vec<[u8; 256]>,
    tiles_x: usize,
    tiles_y: usize,
    width: u32,
    height: u32,
}

impl ClaheLut {
    fn build(
        stacked: &[Vec<u64>],
        tiles_x: usize,
        tiles_y: usize,
        width: u32,
        height: u32,
        clip_limit: f64,
    ) -> Self {
        let num_tiles = tiles_x * tiles_y;
        let mut luts: Vec<[u8; 256]> = vec![[0u8; 256]; num_tiles];

        for (hist, lut) in stacked.iter().zip(luts.iter_mut()) {
            let total: u64 = hist.iter().sum();
            if total == 0 {
                for (i, v) in lut.iter_mut().enumerate() {
                    *v = i as u8;
                }
                continue;
            }

            let clip_abs = ((clip_limit * total as f64) / 256.0).ceil() as u64;
            let mut clipped = hist.to_vec();
            let mut excess = 0u64;
            for bin in clipped.iter_mut() {
                if *bin > clip_abs {
                    excess += *bin - clip_abs;
                    *bin = clip_abs;
                }
            }
            let add_per_bin = excess / 256;
            let remainder = (excess % 256) as usize;
            for bin in clipped.iter_mut() {
                *bin += add_per_bin;
            }
            for bin in clipped.iter_mut().take(remainder) {
                *bin += 1;
            }

            let cdf_total: u64 = clipped.iter().sum();
            let mut cdf = 0u64;
            for (&count, out) in clipped.iter().zip(lut.iter_mut()) {
                cdf += count;
                *out = ((cdf as f64 / cdf_total as f64) * 255.0).round() as u8;
            }
        }

        ClaheLut {
            luts,
            tiles_x,
            tiles_y,
            width,
            height,
        }
    }

    #[inline]
    fn apply(&self, x: u32, y: u32, bin: usize) -> u8 {
        let tile_w = self.width as f64 / self.tiles_x as f64;
        let tile_h = self.height as f64 / self.tiles_y as f64;

        let fx = (x as f64 - tile_w * 0.5) / tile_w;
        let fy = (y as f64 - tile_h * 0.5) / tile_h;

        let tx0 = (fx.floor() as isize).clamp(0, self.tiles_x as isize - 1) as usize;
        let ty0 = (fy.floor() as isize).clamp(0, self.tiles_y as isize - 1) as usize;
        let tx1 = (tx0 + 1).min(self.tiles_x - 1);
        let ty1 = (ty0 + 1).min(self.tiles_y - 1);
        let wx = (fx - tx0 as f64).clamp(0.0, 1.0);
        let wy = (fy - ty0 as f64).clamp(0.0, 1.0);

        let v00 = self.luts[ty0 * self.tiles_x + tx0][bin] as f64;
        let v10 = self.luts[ty0 * self.tiles_x + tx1][bin] as f64;
        let v01 = self.luts[ty1 * self.tiles_x + tx0][bin] as f64;
        let v11 = self.luts[ty1 * self.tiles_x + tx1][bin] as f64;

        let v = v00 * (1.0 - wx) * (1.0 - wy)
            + v10 * wx * (1.0 - wy)
            + v01 * (1.0 - wx) * wy
            + v11 * wx * wy;
        v.round() as u8
    }
}

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    day: u8,
    #[arg(short, long)]
    month: u8,
    #[arg(short, long)]
    year: u16,
    #[arg(short = 'b', long, default_value = BASE_URL)]
    base_url: String,
}

fn strip_timestamp_version(filename: &str) -> &str {
    let s = filename.strip_suffix(".fits").unwrap_or(filename);
    let s = if let Some(pos) = s.rfind("_V") {
        if s[pos + 2..].chars().all(|c| c.is_ascii_digit()) {
            &s[..pos]
        } else {
            s
        }
    } else {
        s
    };
    if let Some(pos) = s.rfind('_') {
        let candidate = &s[pos + 1..];
        if candidate.len() >= 9
            && candidate[..8].chars().all(|c| c.is_ascii_digit())
            && candidate.as_bytes()[8] == b'T'
        {
            &s[..pos]
        } else {
            s
        }
    } else {
        s
    }
}

fn extract_fits_files(html: &str) -> Vec<String> {
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

fn group_fits_files(files: &[String]) -> Vec<(String, Vec<String>)> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in files {
        let group = strip_timestamp_version(file).to_string();
        groups.entry(group).or_default().push(file.clone());
    }
    groups.into_iter().collect()
}

fn find_image_hdu(file: &mut FitsFile) -> Result<(usize, Vec<usize>), DynError> {
    for hdu_index in 0..file.num_hdus()? {
        let hdu = file.hdu(hdu_index)?;
        if let HduInfo::ImageInfo { shape, .. } = &hdu.info
            && !shape.is_empty()
        {
            return Ok((hdu_index, shape.clone()));
        }
    }
    Err("No image HDU found".into())
}

fn processing_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4)
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    (((len - 1) as f64 * percentile / 100.0).round() as usize).clamp(0, len - 1)
}

fn compute_frame_percentiles(
    fits_path: &Path,
    p_low: f64,
    p_high: f64,
) -> Result<Option<(f64, f64)>, DynError> {
    let mut fits_file = FitsFile::open(fits_path)?;
    let (hdu_index, _) = find_image_hdu(&mut fits_file)?;
    let hdu = fits_file.hdu(hdu_index)?;
    let mut pixels: Vec<f32> = hdu.read_image(&mut fits_file)?;

    pixels.retain(|value| value.is_finite());
    if pixels.is_empty() {
        return Ok(None);
    }

    let low_index = percentile_index(pixels.len(), p_low);
    let high_index = percentile_index(pixels.len(), p_high);

    let low = {
        let (_, value, _) =
            pixels.select_nth_unstable_by(low_index, |a, b| a.partial_cmp(b).unwrap());
        *value as f64
    };

    let high = if high_index == low_index {
        low
    } else {
        let (_, value, _) =
            pixels.select_nth_unstable_by(high_index, |a, b| a.partial_cmp(b).unwrap());
        *value as f64
    };

    Ok(Some((low, high)))
}

async fn download_fits_file(
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

async fn prompt_retry_failed(failed_count: usize) -> Result<bool, DynError> {
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

async fn download_fits_files_pass(
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

async fn download_fits_files(
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

async fn prompt_group_selection(groups: &[(String, Vec<String>)]) -> Result<Vec<usize>, DynError> {
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

async fn ensure_all_files_are_present(filenames: &[String], data_dir: &Path) -> Result<(), DynError> {
    let mut missing_files = Vec::new();

    for filename in filenames {
        let output_path = data_dir.join(filename);
        if !fs::try_exists(&output_path).await? {
            missing_files.push(filename.clone());
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

async fn generate_video_from_fits_files(
    filenames: &[String],
    output_path: &Path,
    data_dir: &Path,
) -> Result<(), DynError> {
    let frame_dir = output_path.with_extension("");
    if fs::try_exists(&frame_dir).await? {
        fs::remove_dir_all(&frame_dir).await?;
    }
    fs::create_dir_all(&frame_dir).await?;

    log_step("Computing global intensity range...");
    let (global_min, global_max) = compute_global_range(filenames, 1.0, 99.0, data_dir).await?;
    log_step(&format!("Global range: {} to {}", global_min, global_max));

    let mapping = AsinhMapping::from_range(global_min, global_max);
    log_step(&format!(
        "Asinh beta: {:.4} (softening = 2 % of range)",
        mapping.beta
    ));

    log_step("Building fixed CLAHE LUT from stacked histograms...");
    let clahe = Arc::new(build_clahe_lut(filenames, mapping, data_dir).await?);
    log_step(&format!(
        "CLAHE LUT ready ({} tiles, clip limit {}x)",
        CLAHE_TILES_X * CLAHE_TILES_Y,
        CLAHE_CLIP_LIMIT
    ));

    log_step("Converting FITS files to PNG frames...");
    convert_fits_files_to_png_frames(filenames, &frame_dir, mapping, clahe, data_dir).await?;

    log_step("All frames processed, generating video...");

    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg("24")
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

    log_step(&format!("Generated video at {}", output_path.display()));
    Ok(())
}

async fn compute_global_range(
    filenames: &[String],
    p_low: f64,
    p_high: f64,
    data_dir: &Path,
) -> Result<(f64, f64), DynError> {
    let concurrency = processing_concurrency();
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();
    let mut lows = Vec::with_capacity(filenames.len());
    let mut highs = Vec::with_capacity(filenames.len());
    let data_dir = data_dir.to_path_buf();

    log_step("Computing per-frame intensity percentiles...");
    for filename in filenames {
        let fits_path = data_dir.join(filename);
        let semaphore = Arc::clone(&semaphore);

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|error| -> DynError { Box::new(error) })?;

            spawn_blocking(move || compute_frame_percentiles(&fits_path, p_low, p_high))
                .await
                .map_err(|error| -> DynError { Box::new(error) })?
        });
    }

    let mut completed = 0usize;
    let mut first_error: Option<DynError> = None;
    while let Some(result) = join_set.join_next().await {
        completed += 1;
        log_progress("Processing", completed, filenames.len())?;

        if first_error.is_some() {
            continue;
        }

        match result {
            Ok(Ok(Some((low, high)))) => {
                lows.push(low);
                highs.push(high);
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => {
                eprintln!("Percentile task failed: {}", error);
                first_error = Some(error);
                join_set.abort_all();
            }
            Err(error) => {
                eprintln!("Percentile task join error: {}", error);
                first_error = Some(Box::new(error));
                join_set.abort_all();
            }
        }
    }

    if !filenames.is_empty() {
        println!();
    }

    if lows.is_empty() {
        return Ok((0.0, 1.0));
    }

    if let Some(error) = first_error {
        return Err(error);
    }

    let median_min = {
        let mid = lows.len() / 2;
        let (_, value, _) = lows.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        *value
    };
    let median_max = {
        let mid = highs.len() / 2;
        let (_, value, _) = highs.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        *value
    };
    log_step(&format!(
        "Median 1st percentile: {}, Median 99th percentile: {}",
        median_min, median_max
    ));
    Ok((median_min, median_max))
}

fn compute_frame_tile_histograms(
    fits_path: &Path,
    mapping: AsinhMapping,
    tiles_x: usize,
    tiles_y: usize,
) -> Result<(u32, u32, Vec<Vec<u64>>), DynError> {
    let mut fits_file = FitsFile::open(fits_path)?;
    let (hdu_index, shape) = find_image_hdu(&mut fits_file)?;
    if shape.len() != 2 {
        return Err(format!("Expected 2D image, got {:?}", shape).into());
    }
    let width = shape[0] as u32;
    let height = shape[1] as u32;
    let hdu = fits_file.hdu(hdu_index)?;
    let pixels: Vec<f32> = hdu.read_image(&mut fits_file)?;

    let num_tiles = tiles_x * tiles_y;
    let mut histograms: Vec<Vec<u64>> = vec![vec![0u64; 256]; num_tiles];
    let tile_w = width as f64 / tiles_x as f64;
    let tile_h = height as f64 / tiles_y as f64;
    let w = width as usize;

    for (i, &v) in pixels.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let px = (i % w) as u32;
        let py = (i / w) as u32;
        let tx = ((px as f64 / tile_w) as usize).min(tiles_x - 1);
        let ty = ((py as f64 / tile_h) as usize).min(tiles_y - 1);
        let bin = (mapping.apply(v as f64) * 255.0).round() as usize;
        histograms[ty * tiles_x + tx][bin] += 1;
    }

    Ok((width, height, histograms))
}

async fn build_clahe_lut(
    filenames: &[String],
    mapping: AsinhMapping,
    data_dir: &Path,
) -> Result<ClaheLut, DynError> {
    let concurrency = processing_concurrency();
    let semaphore = Arc::new(Semaphore::new(concurrency));
    type TaskResult = Result<(u32, u32, Vec<Vec<u64>>), DynError>;
    let mut join_set: JoinSet<TaskResult> = JoinSet::new();
    let data_dir = data_dir.to_path_buf();

    log_step("Computing stacked tile histograms for CLAHE...");
    for filename in filenames {
        let fits_path = data_dir.join(filename);
        let semaphore = Arc::clone(&semaphore);

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|e| -> DynError { Box::new(e) })?;
            spawn_blocking(move || {
                compute_frame_tile_histograms(&fits_path, mapping, CLAHE_TILES_X, CLAHE_TILES_Y)
            })
            .await
            .map_err(|e| -> DynError { Box::new(e) })?
        });
    }

    let num_tiles = CLAHE_TILES_X * CLAHE_TILES_Y;
    let mut stacked: Vec<Vec<u64>> = vec![vec![0u64; 256]; num_tiles];
    let mut img_width = 0u32;
    let mut img_height = 0u32;
    let mut completed = 0usize;
    let mut first_error: Option<DynError> = None;

    while let Some(result) = join_set.join_next().await {
        completed += 1;
        log_progress("Histogram", completed, filenames.len())?;

        if first_error.is_some() {
            continue;
        }
        match result {
            Ok(Ok((w, h, hist))) => {
                img_width = w;
                img_height = h;
                for (tile_hist, tile_stacked) in hist.iter().zip(stacked.iter_mut()) {
                    for (src, dst) in tile_hist.iter().zip(tile_stacked.iter_mut()) {
                        *dst += src;
                    }
                }
            }
            Ok(Err(error)) => {
                eprintln!("\nHistogram task failed: {}", error);
                first_error = Some(error);
                join_set.abort_all();
            }
            Err(error) => {
                eprintln!("\nHistogram join error: {}", error);
                first_error = Some(Box::new(error));
                join_set.abort_all();
            }
        }
    }

    if !filenames.is_empty() {
        println!();
    }
    if let Some(error) = first_error {
        return Err(error);
    }

    Ok(ClaheLut::build(
        &stacked,
        CLAHE_TILES_X,
        CLAHE_TILES_Y,
        img_width,
        img_height,
        CLAHE_CLIP_LIMIT,
    ))
}

async fn convert_fits_files_to_png_frames(
    filenames: &[String],
    frame_dir: &Path,
    mapping: AsinhMapping,
    clahe: Arc<ClaheLut>,
    data_dir: &Path,
) -> Result<(), DynError> {
    let concurrency = processing_concurrency();
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();
    let data_dir = data_dir.to_path_buf();

    for (index, filename) in filenames.iter().enumerate() {
        let fits_path = data_dir.join(filename);
        let frame_path = frame_dir.join(format!("frame_{:05}.png", index));
        let semaphore = Arc::clone(&semaphore);
        let clahe = Arc::clone(&clahe);

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|error| -> DynError { Box::new(error) })?;

            spawn_blocking(move || fits_to_png_frame(&fits_path, &frame_path, mapping, &clahe))
                .await
                .map_err(|error| -> DynError { Box::new(error) })?
        });
    }

    let mut completed = 0usize;
    let mut first_error: Option<DynError> = None;
    while let Some(result) = join_set.join_next().await {
        completed += 1;
        log_progress("Processing frame", completed, filenames.len())?;

        if first_error.is_some() {
            continue;
        }

        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                eprintln!("Frame conversion failed: {}", error);
                first_error = Some(error);
                join_set.abort_all();
            }
            Err(error) => {
                eprintln!("Frame task join error: {}", error);
                first_error = Some(Box::new(error));
                join_set.abort_all();
            }
        }
    }

    if !filenames.is_empty() {
        println!();
    }

    if let Some(error) = first_error {
        return Err(error);
    }

    Ok(())
}

fn fits_to_png_frame(
    fits_path: &Path,
    png_path: &Path,
    mapping: AsinhMapping,
    clahe: &ClaheLut,
) -> Result<(), DynError> {
    let mut fits_file = FitsFile::open(fits_path)?;
    let (hdu_index, shape) = find_image_hdu(&mut fits_file)?;

    if shape.len() != 2 {
        return Err(format!("Expected 2D image, got shape {:?}", shape).into());
    }

    let width = shape[0] as u32;
    let height = shape[1] as u32;

    let hdu = fits_file.hdu(hdu_index)?;
    let pixels: Vec<f32> = hdu.read_image(&mut fits_file)?;
    let w = width as usize;

    let bytes: Vec<u8> = pixels
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            if !v.is_finite() {
                return 0;
            }
            let x = (i % w) as u32;
            let y = (i / w) as u32;
            let bin = (mapping.apply(v as f64) * 255.0).round() as usize;
            clahe.apply(x, y, bin)
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
        args.base_url, args.year, args.month, args.day
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
                        let all_fits_files = extract_fits_files(&html);

                        if all_fits_files.is_empty() {
                            eprintln!("No FITS files found for this date.");
                            return;
                        }

                        let groups = group_fits_files(&all_fits_files);

                        let selected_indices = match prompt_group_selection(&groups).await {
                            Ok(indices) => indices,
                            Err(e) => {
                                eprintln!("Error reading selection: {}", e);
                                return;
                            }
                        };

                        if selected_indices.is_empty() {
                            println!("No datasets selected. Exiting.");
                            return;
                        }

                        for &idx in &selected_indices {
                            let (group_name, files) = &groups[idx];
                            let data_dir = Path::new(DATA_DIR).join(group_name);

                            println!(
                                "\n--- Processing group: {} ({} files) ---",
                                group_name,
                                files.len()
                            );

                            if let Err(error) = download_fits_files(&url, files, &data_dir).await {
                                eprintln!("Error downloading files: {}", error);
                                continue;
                            }

                            if let Err(error) =
                                ensure_all_files_are_present(files, &data_dir).await
                            {
                                eprintln!("Data verification failed: {}", error);
                                continue;
                            }

                            let video_name = format!(
                                "{}-{:02}-{:02}.mp4",
                                args.year, args.month, args.day
                            );
                            let video_path = data_dir.join(video_name);

                            if let Err(error) =
                                generate_video_from_fits_files(files, &video_path, &data_dir).await
                            {
                                eprintln!("Error generating video: {}", error);
                            }
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
