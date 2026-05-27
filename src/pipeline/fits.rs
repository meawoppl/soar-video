use std::{collections::BTreeMap, path::Path, sync::Arc};

use fitsio::FitsFile;
use image::GrayImage;
use tokio::{
    fs,
    process::Command,
    sync::Semaphore,
    task::{JoinSet, spawn_blocking},
};

use crate::{
    pipeline::{
        asinh::AsinhMapping,
        blur::apply_unsharp_mask,
        clahe::{CLAHE_CLIP_LIMIT, CLAHE_TILES_X, CLAHE_TILES_Y, ClaheLut, build_clahe_lut},
        compute_global_range, find_image_hdu,
    },
    processing_concurrency,
    util::{DynError, log_progress, log_step},
};

pub async fn convert_fits_files_to_png_frames(
    filenames: &[String],
    frame_dir: &Path,
    mapping: AsinhMapping,
    clahe: Arc<ClaheLut>,
    data_dir: &Path,
    unsharp_strength: Option<f32>,
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

            spawn_blocking(move || {
                fits_to_png_frame(&fits_path, &frame_path, mapping, &clahe, unsharp_strength)
            })
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

pub fn fits_to_png_frame(
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

pub fn group_fits_files(files: &[String]) -> Vec<(String, Vec<String>)> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in files {
        let group = strip_timestamp_version(file).to_string();
        groups.entry(group).or_default().push(file.clone());
    }
    groups.into_iter().collect()
}

pub async fn generate_video_from_fits_files(
    filenames: &[String],
    output_path: &Path,
    data_dir: &Path,
    unsharp_strength: Option<f32>,
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
        mapping.beta()
    ));

    log_step("Building fixed CLAHE LUT from stacked histograms...");
    let clahe = Arc::new(build_clahe_lut(filenames, mapping, data_dir).await?);
    log_step(&format!(
        "CLAHE LUT ready ({} tiles, clip limit {}x)",
        CLAHE_TILES_X * CLAHE_TILES_Y,
        CLAHE_CLIP_LIMIT
    ));

    if let Some(s) = unsharp_strength {
        log_step(&format!(
            "Unsharp masking enabled (strength {s:.2}, sigma 2.0 px)"
        ));
    }
    log_step("Converting FITS files to PNG frames...");
    convert_fits_files_to_png_frames(
        filenames,
        &frame_dir,
        mapping,
        clahe,
        data_dir,
        unsharp_strength,
    )
    .await?;

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

pub fn strip_timestamp_version(filename: &str) -> &str {
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
