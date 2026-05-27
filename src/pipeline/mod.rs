use std::{path::Path, sync::Arc};

use fitsio::{FitsFile, hdu::HduInfo};
use tokio::{
    sync::Semaphore,
    task::{JoinSet, spawn_blocking},
};

use crate::util::{DynError, log_progress, log_step, percentile_index, processing_concurrency};

pub mod asinh;
pub mod blur;
pub mod clahe;
pub mod fits;

pub mod download;

pub fn find_image_hdu(file: &mut FitsFile) -> Result<(usize, Vec<usize>), DynError> {
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
