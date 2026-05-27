/// Build a 1-D Gaussian kernel with the given sigma.
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
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
pub fn gaussian_blur(src: &[u8], width: u32, height: u32, sigma: f32) -> Vec<u8> {
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
                let sx =
                    (x as isize + ki as isize - radius as isize).clamp(0, w as isize - 1) as usize;
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
                let sy =
                    (y as isize + ki as isize - radius as isize).clamp(0, h as isize - 1) as usize;
                acc += tmp[sy * w + x] * kv;
            }
            out[y * w + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Unsharp mask: `output = clamp(src + strength * (src - blurred), 0, 255)`.
pub fn apply_unsharp_mask(src: &[u8], width: u32, height: u32, strength: f32) -> Vec<u8> {
    let blurred = gaussian_blur(src, width, height, 2.0);
    src.iter()
        .zip(blurred.iter())
        .map(|(&orig, &blur)| {
            let v = orig as f32 + strength * (orig as f32 - blur as f32);
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}
