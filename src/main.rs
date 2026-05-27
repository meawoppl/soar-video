use clap::Parser;
use std::path::Path;

use crate::{
    pipeline::{
        download::{
            download_fits_files, extract_fits_files, fetch_directory_listing,
            prompt_group_selection,
        },
        fits::generate_video_from_fits_files,
    },
    util::{BASE_URL, DATA_DIR, DynError, build_url, processing_concurrency},
};

// import lib.rs
mod pipeline;
mod util;

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
    /// Apply unsharp masking after CLAHE.  Pass alone for the default strength
    /// of 1.0 (`--unsharp`), or with a value to tune it (`--unsharp 0.5`).
    #[arg(long, default_missing_value = "1.0", num_args = 0..=1, require_equals = false)]
    unsharp: Option<f32>,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
    }
}

async fn run() -> Result<(), DynError> {
    let args = Args::parse();
    let url = build_url(&args);
    println!("Fetching data from: {}", url);

    let client = reqwest::Client::builder()
        .user_agent("sun/0.1.0")
        .build()
        .expect("failed to build HTTP client");

    let html = fetch_directory_listing(&client, &url).await?;
    let all_fits_files = extract_fits_files(&html);

    if all_fits_files.is_empty() {
        eprintln!("No FITS files found for this date.");
        return Ok(());
    }

    let groups = crate::pipeline::fits::group_fits_files(&all_fits_files);
    let selected_indices = prompt_group_selection(&groups).await?;

    if selected_indices.is_empty() {
        println!("No datasets selected. Exiting.");
        return Ok(());
    }

    for &idx in &selected_indices {
        process_group(&args, &url, &groups[idx]).await?;
    }

    Ok(())
}

async fn process_group(
    args: &Args,
    url: &str,
    group: &(String, Vec<String>),
) -> Result<(), DynError> {
    let (group_name, files) = group;
    let data_dir = Path::new(DATA_DIR).join(group_name);

    println!(
        "\n--- Processing group: {} ({} files) ---",
        group_name,
        files.len()
    );

    download_fits_files(url, files, &data_dir).await?;
    crate::pipeline::download::ensure_all_files_are_present(files, &data_dir).await?;

    let video_name = format!("{}-{:02}-{:02}.mp4", args.year, args.month, args.day);
    let video_path = data_dir.join(video_name);

    generate_video_from_fits_files(files, &video_path, &data_dir, args.unsharp).await?;
    Ok(())
}
