use std::{error::Error, io::Write};

pub const BASE_URL: &str = "https://www.sidc.be/EUI/data/L2";
pub const DATA_DIR: &str = "data";
pub const MAX_CONCURRENT_DOWNLOADS: usize = 4;

pub type DynError = Box<dyn Error + Send + Sync>;
pub fn log_step(message: &str) {
    println!("{message}");
}

pub fn log_progress(prefix: &str, current: usize, total: usize) -> Result<(), DynError> {
    print!("\r{} {}/{}", prefix, current, total);
    std::io::stdout().flush()?;
    Ok(())
}

pub fn processing_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4)
}

pub fn percentile_index(len: usize, percentile: f64) -> usize {
    (((len - 1) as f64 * percentile / 100.0).round() as usize).clamp(0, len - 1)
}
