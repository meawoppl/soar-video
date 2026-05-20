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
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::Semaphore,
    task::{JoinSet, spawn_blocking},
};

const BASE_URL: &str = "https://www.sidc.be/EUI/data/L2";
const DATA_DIR: &str = "data";
const MAX_CONCURRENT_DOWNLOADS: usize = 4;

/// Number of horizontal / vertical tiles for CLAHE.
const CLAHE_TILES_X: usize = 8;
const CLAHE_TILES_Y: usize = 8;
/// Clip limit relative to the average bin occupancy (typical: 2–4).
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

/// Per-tile LUTs derived from histograms **stacked across every frame**.
///
/// Because the same LUTs are applied to all frames the mapping is
/// frame-independent, so the video has no flicker.  Within each frame the
/// four surrounding tile LUTs are bilinearly interpolated per pixel, which
/// gives the local contrast benefit of adaptive histogram equalization.
struct ClaheLut {
    /// Flattened `[tiles_y * tiles_x]` array; each entry maps bin → output u8.
    luts: Vec<[u8; 256]>,
    tiles_x: usize,
    tiles_y: usize,
    width: u32,
    height: u32,
}

impl ClaheLut {
    /// Build LUTs from a stacked histogram (sum over all frames).
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

            // Clip and redistribute excess uniformly.
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

            // CDF → normalised LUT.
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

    /// Look up the output value for pixel `(x, y)` whose asinh-normalised
    /// value maps to `bin` (0–255), using bilinear tile interpolation.
    #[inline]
    fn apply(&self, x: u32, y: u32, bin: usize) -> u8 {
        let tile_w = self.width as f64 / self.tiles_x as f64;
        let tile_h = self.height as f64 / self.tiles_y as f64;

        // Fractional position in tile-centre space.
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
    /// Day
    #[arg(short, long)]
    day: u8,
    /// Month
    #[arg(short, long)]
    month: u8,
    /// Year
    #[arg(short, long)]
    year: u16,
    /// Apply unsharp masking after CLAHE.  Pass alone for the default strength
    /// of 1.0 (`--unsharp`), or with a value to tune it (`--unsharp 0.5`).
    #[arg(long, default_missing_value = "1.0", num_args = 0..=1, require_equals = false)]
    unsharp: Option<f32>,
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
        println!("\nSkipping existing {}", filename);
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

async fn download_hrieuv174_fits_files_pass(
    base_url: &str,
    filenames: &[String],
    label: &str,
) -> Result<(usize, usize, Vec<String>), DynError> {
    if filenames.is_empty() {
        return Ok((0, 0, Vec::new()));
    }

    fs::create_dir_all(DATA_DIR).await?;

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

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|error| -> DynError { Box::new(error) })?;

            let result = download_hrieuv174_fits_file(&client, &base_url, &filename).await;
            Ok::<_, DynError>((filename, result))
        });
    }

    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut completed = 0usize;
    let mut failed = Vec::new();

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok((filename, Ok(true)))) => downloaded += 1,
            Ok(Ok((filename, Ok(false)))) => skipped += 1,
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

async fn download_hrieuv174_fits_files(
    base_url: &str,
    filenames: &[String],
) -> Result<(), DynError> {
    let (mut downloaded, mut skipped, mut failed) =
        download_hrieuv174_fits_files_pass(base_url, filenames, "Starting downloads").await?;

    while !failed.is_empty() {
        log_step(&format!("{} files failed to download.", failed.len()));

        while !prompt_retry_failed(failed.len()).await? {
            log_step("Retry required to complete all downloads.");
        }

        let (retry_downloaded, retry_skipped, retry_failed) =
            download_hrieuv174_fits_files_pass(base_url, &failed, "Retrying failed downloads")
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

async fn ensure_all_files_are_present(filenames: &[String]) -> Result<(), DynError> {
    let mut missing_files = Vec::new();

    for filename in filenames {
        let output_path = Path::new(DATA_DIR).join(filename);
        if !fs::try_exists(&output_path).await? {
            missing_files.push(filename.clone()
        }
    }

    if missing_files.is_empty() {
        log_step(&format!(
            "Verified {} FITS files in {}",
            filenames.len(),
            DATA_DIR
        ));
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
/// Build a 1-D Gaussian kernel with the given sigma.
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let mut kernel: Vec<f32> = (0..=2 * radius)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= sum);
    kernel
}

/// Separable Gaussian blur (horizontal pass then vertical pass).
fn gaussian_blur(src: &[u8], width: u32, height: u32, sigma: f32) -> Vec<u8> {
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let w = width as usize;
    let h = height as usize;

    // Horizontal pass → f32 scratch buffer.
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + ki as isize - radius as isize).clamp(0, w as isize - 1) as usize;
                acc += src[y * w + sx] as f32 * kv;
            }
            tmp[y * w + x] = acc;
        }
    }

    // Vertical pass → output bytes.
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize + ki as isize - radius as isize).clamp(0, h as isize - 1) as usize;
                acc += tmp[sy * w + x] * kv;
            }
            out[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Unsharp mask: `output = clamp(src + strength * (src - blurred), 0, 255)`.
fn apply_unsharp_mask(src: &[u8], width: u32, height: u32, strength: f32) -> Vec<u8> {
    let blurred = gaussian_blur(src, width, height, 2.0);
    src.iter()
        .zip(blurred.iter())
        .map(|(&orig, &blur)| {
            let v = orig as f32 + strength * (orig as f32 - blur as f32);
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

async fn generate_video_from_fits_files(
    filenames: &[String],
    output_path: &Path,
    unsharp_strength: Option<f32>,
) -> Result<(), DynError> {
    let frame_dir = output_path.with_extension("");
    if fs::try_exists(&frame_dir).await? {
        fs::remove_dir_all(&frame_dir).await?;
    }
    fs::create_dir_all(&frame_dir).await?;

    log_step("Computing global intensity range...");
    let (global_min, global_max) = compute_global_range(filenames, 1.0, 99.0).await?;
    log_step(&format!("Global range: {} to {}", global_min, global_max));

    let mapping = AsinhMapping::from_range(global_min, global_max);
    log_step(&format!(
        "Asinh beta: {:.4} (softening = 2 % of range)",
        mapping.beta
    ));

    log_step("Building fixed CLAHE LUT from stacked histograms...");
    let clahe = Arc::new(build_clahe_lut(filenames, mapping).await?);
    log_step(&format!(
        "CLAHE LUT ready ({} tiles, clip limit {}x)",
        CLAHE_TILES_X * CLAHE_TILES_Y,
        CLAHE_CLIP_LIMIT
    ));

    if let Some(s) = unsharp_strength {
        println!("Unsharp masking enabled (strength {s:.2}, sigma 2.0 px)");
    }
    println!("Converting FITS files to PNG frames...");
    convert_fits_files_to_png_frames(filenames, &frame_dir, mapping, clahe, unsharp_strength).await?;

    log_step("All frames processed, generating video...");

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

    log_step(&format!("Generated video at {}", output_path.display()));
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

    log_step("Computing per-frame intensity percentiles...");
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

/// Read one FITS file and accumulate a per-tile histogram in asinh-normalised
/// space.  Returns `(width, height, histograms[tile_idx][bin])`.
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

/// Stack per-frame tile histograms across all files and compute one fixed set
/// of CLAHE LUTs.  Every frame sees the same LUTs → no flicker.
async fn build_clahe_lut(
    filenames: &[String],
    mapping: AsinhMapping,
) -> Result<ClaheLut, DynError> {
    let concurrency = processing_concurrency();
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set: JoinSet<Result<(u32, u32, Vec<Vec<u64>>), DynError>> = JoinSet::new();

    log_step("Computing stacked tile histograms for CLAHE...");
    for filename in filenames {
        let fits_path = Path::new(DATA_DIR).join(filename);
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
    unsharp_strength: Option<f32>,
) -> Result<(), DynError> {
    let concurrency = processing_concurrency();
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();

    for (index, filename) in filenames.iter().enumerate() {
        let fits_path = Path::new(DATA_DIR).join(filename);
        let frame_path = frame_dir.join(format!("frame_{:05}.png", index));
        let semaphore = Arc::clone(&semaphore);
        let clahe = Arc::clone(&clahe);

        join_set.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .map_err(|error| -> DynError { Box::new(error) })?;

            spawn_blocking(move || fits_to_png_frame(&fits_path, &frame_path, mapping, &clahe, unsharp_strength))
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
    unsharp_strength: Option<f32>,
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

    let bytes = match unsharp_strength {
        Some(strength) => apply_unsharp_mask(&bytes, width, height, strength),
        None => bytes,
    };

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
                            generate_video_from_fits_files(&fits_files, &video_path, args.unsharp).await
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
