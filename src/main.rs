use clap::Parser;
use fitsio::{FitsFile, hdu::HduInfo};
use image::GrayImage;
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

/// Asinh stretch mapping, derived once from the global percentile range and
/// applied identically to every frame so that the video has no flicker.
///
/// For a pixel value `v`:
///   normalized = asinh((v - min) / beta) / asinh((max - min) / beta)
///
/// `beta` (the "softening" parameter) sets the pivot between linear and
/// logarithmic behaviour.  Values well below `beta` are mapped linearly,
/// preserving faint coronal structure; values well above `beta` are mapped
/// logarithmically, compressing the extreme dynamic range while keeping
/// bright active-region features visually distinct.
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
        // 2 % of the full range: faint features stay linear while bright
        // structures are gently compressed rather than crushed.
        let beta = 0.02 * range;
        let inv_asinh_range = ((range / beta).asinh()).max(f64::EPSILON).recip();
        Self { min: global_min, max: global_max, beta, inv_asinh_range }
    }

    #[inline]
    fn apply(&self, value: f64) -> f64 {
        let clamped = value.clamp(self.min, self.max);
        let shifted = clamped - self.min;
        ((shifted / self.beta).asinh() * self.inv_asinh_range).clamp(0.0, 1.0)
    }
}

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

            if filename.contains("hrieuv174")
                && filename.ends_with(".fits")
                && !filename.contains("short")
            {
                Some(filename.to_string())
            } else {
                None
            }
        })
        .collect()
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
    let (global_min, global_max) = compute_global_range(filenames, 1.0, 99.0).await?;
    println!("Global range: {} to {}", global_min, global_max);

    let mapping = AsinhMapping::from_range(global_min, global_max);
    println!("Asinh beta: {:.4} (softening = 2 % of range)", mapping.beta);

    println!("Converting FITS files to PNG frames...");
    convert_fits_files_to_png_frames(filenames, &frame_dir, mapping).await?;

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

async fn compute_global_range(
    filenames: &[String],
    p_low: f64,
    p_high: f64,
) -> Result<(f64, f64), DynError> {
    let concurrency = processing_concurrency();
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();
    let mut lows = Vec::with_capacity(filenames.len());
    let mut highs = Vec::with_capacity(filenames.len());

    println!("Computing per-frame intensity percentiles...");
    for filename in filenames {
        let fits_path = Path::new(DATA_DIR).join(filename);
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
        print!("\rProcessing {}/{}", completed, filenames.len());
        std::io::stdout().flush()?;

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
    println!(
        "\nMedian 1st percentile: {}, Median 99th percentile: {}",
        median_min, median_max
    );
    Ok((median_min, median_max))
}

async fn convert_fits_files_to_png_frames(
    filenames: &[String],
    frame_dir: &Path,
    mapping: AsinhMapping,
) -> Result<(), DynError> {
    let concurrency = processing_concurrency();
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();

    for (index, filename) in filenames.iter().enumerate() {
        let fits_path = Path::new(DATA_DIR).join(filename);
        let frame_path = frame_dir.join(format!("frame_{:05}.png", index));
        let semaphore = Arc::clone(&semaphore);

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|error| -> DynError { Box::new(error) })?;

            spawn_blocking(move || fits_to_png_frame(&fits_path, &frame_path, mapping))
                .await
                .map_err(|error| -> DynError { Box::new(error) })?
        });
    }

    let mut completed = 0usize;
    let mut first_error: Option<DynError> = None;
    while let Some(result) = join_set.join_next().await {
        completed += 1;
        print!("\rProcessing frame {}/{}", completed, filenames.len());
        std::io::stdout().flush()?;

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

    let bytes: Vec<u8> = pixels
        .into_iter()
        .map(|v| {
            if !v.is_finite() {
                return 0;
            }
            (mapping.apply(v as f64) * 255.0).round() as u8
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
