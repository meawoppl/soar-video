use std::{path::Path, sync::Arc};

use fitsio::FitsFile;
use tokio::{
    sync::Semaphore,
    task::{JoinSet, spawn_blocking},
};

use crate::{
    pipeline::{asinh::AsinhMapping, find_image_hdu},
    processing_concurrency,
    util::{DynError, log_progress, log_step},
};

pub const CLAHE_TILES_X: usize = 8;
pub const CLAHE_TILES_Y: usize = 8;
pub const CLAHE_CLIP_LIMIT: f64 = 3.0;
pub struct ClaheLut {
    luts: Vec<[u8; 256]>,
    tiles_x: usize,
    tiles_y: usize,
    width: u32,
    height: u32,
}

impl ClaheLut {
    pub fn build(
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
    pub fn apply(&self, x: u32, y: u32, bin: usize) -> u8 {
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

pub async fn build_clahe_lut(
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
