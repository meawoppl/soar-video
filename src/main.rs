use clap::Parser;
use std::path::Path;

use crate::{
    pipeline::{download::ensure_all_files_are_present, fits::group_fits_files},
    util::processing_concurrency,
};
use crate::{
    pipeline::{
        download::{download_fits_files, extract_fits_files, prompt_group_selection},
        fits::generate_video_from_fits_files,
    },
    util::{BASE_URL, DATA_DIR},
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

                            if let Err(error) = ensure_all_files_are_present(files, &data_dir).await
                            {
                                eprintln!("Data verification failed: {}", error);
                                continue;
                            }

                            let video_name =
                                format!("{}-{:02}-{:02}.mp4", args.year, args.month, args.day);
                            let video_path = data_dir.join(video_name);

                            if let Err(error) = generate_video_from_fits_files(
                                files,
                                &video_path,
                                &data_dir,
                                args.unsharp,
                            )
                            .await
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
