use std::collections::VecDeque;

use slouch_domain::{ported::messages::schemas::ImageData, CameraSettings};

/// Stateful native preprocessing applied to raw camera RGBA before inference.
#[derive(Debug, Default)]
pub struct NativePreprocessor {
    frames: VecDeque<ImageData>,
    smoothing_frames: u8,
}

impl NativePreprocessor {
    /// Ingests one camera-cadence sample into the temporal ring without
    /// performing inference-time spatial transforms.
    pub fn ingest_camera_frame(
        &mut self,
        image: ImageData,
        settings: &CameraSettings,
    ) -> Result<(), String> {
        settings.validate()?;
        validate_image(&image)?;
        if self.smoothing_frames != settings.smoothing_frames
            || self
                .frames
                .front()
                .is_some_and(|frame| frame.width != image.width || frame.height != image.height)
        {
            self.frames.clear();
            self.smoothing_frames = settings.smoothing_frames;
        }
        self.frames.push_back(image);
        while self.frames.len() > usize::from(settings.smoothing_frames) {
            self.frames.pop_front();
        }
        Ok(())
    }

    /// Returns the latest camera-rate moving average with the configured
    /// spatial transforms, without appending another inference-cadence frame.
    pub fn process_latest(&self, settings: &CameraSettings) -> Result<ImageData, String> {
        settings.validate()?;
        let first = self
            .frames
            .front()
            .ok_or_else(|| "native preprocessing buffer is empty".to_owned())?;
        let mut data = vec![0_u8; first.data.len()];
        for (index, value) in data.iter_mut().enumerate() {
            let sum = self
                .frames
                .iter()
                .try_fold(0_u32, |total, frame| {
                    total.checked_add(u32::from(frame.data[index]))
                })
                .ok_or_else(|| "temporal smoothing accumulator overflowed".to_owned())?;
            *value = clamp_u8_ties_even(f64::from(sum) / self.frames.len() as f64);
        }

        if settings.gaussian_blur_kernel > 0 {
            data = gaussian_blur_rgba(
                &data,
                first.width as usize,
                first.height as usize,
                usize::from(settings.gaussian_blur_kernel),
            );
        }
        if settings.clahe_strength > 0.0 {
            apply_clahe_lab(
                &mut data,
                first.width as usize,
                first.height as usize,
                settings.clahe_strength,
            );
        }
        // OpenCV's RGBA2RGB/RGB2RGBA round trip discards source alpha and
        // recreates an opaque channel whenever either transform is active.
        if settings.gaussian_blur_kernel > 0 || settings.clahe_strength > 0.0 {
            for alpha in data.iter_mut().skip(3).step_by(4) {
                *alpha = 255;
            }
        }
        Ok(ImageData {
            data,
            width: first.width,
            height: first.height,
        })
    }

    /// Backward-compatible one-shot entry. Camera integrations should call
    /// `ingest_camera_frame` for every sample and `process_latest` only when an
    /// inference request is due.
    pub fn process(
        &mut self,
        image: ImageData,
        settings: &CameraSettings,
    ) -> Result<ImageData, String> {
        self.ingest_camera_frame(image, settings)?;
        self.process_latest(settings)
    }

    pub fn reset(&mut self) {
        self.frames.clear();
        self.smoothing_frames = 0;
    }
}

const MAX_IMAGE_WIDTH: u32 = 1920;
const MAX_IMAGE_HEIGHT: u32 = 1080;
const MAX_RGBA_BYTES: usize = 8_294_400;

fn validate_image(image: &ImageData) -> Result<(), String> {
    if image.width == 0 || image.height == 0 {
        return Err("RGBA image dimensions must be positive".to_owned());
    }
    if image.width > MAX_IMAGE_WIDTH || image.height > MAX_IMAGE_HEIGHT {
        return Err(format!(
            "RGBA image dimensions exceed {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"
        ));
    }
    let expected = usize::try_from(image.width)
        .ok()
        .and_then(|width| {
            usize::try_from(image.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "RGBA image dimensions overflow".to_owned())?;
    if expected > MAX_RGBA_BYTES {
        return Err(format!(
            "RGBA image is larger than the {MAX_RGBA_BYTES}-byte limit"
        ));
    }
    if expected != image.data.len() {
        return Err(format!(
            "RGBA image has {} bytes, expected {expected}",
            image.data.len()
        ));
    }
    Ok(())
}

fn clamp_u8_ties_even(value: f64) -> u8 {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction > 0.5 || (fraction == 0.5 && (floor as u64) % 2 == 1) {
        floor + 1.0
    } else {
        floor
    };
    rounded.clamp(0.0, 255.0) as u8
}

fn gaussian_kernel(kernel_size: usize) -> Vec<f64> {
    // OpenCV uses these exact bit-stable kernels when sigma is zero.
    match kernel_size {
        1 => return vec![1.0],
        3 => return vec![0.25, 0.5, 0.25],
        5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
        7 => {
            return vec![
                0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
            ];
        }
        _ => {}
    }
    let radius = kernel_size / 2;
    let sigma = 0.3 * (radius as f64 - 1.0) + 0.8;
    let mut kernel = (0..kernel_size)
        .map(|index| {
            let distance = index as f64 - radius as f64;
            (-distance * distance / (2.0 * sigma * sigma)).exp()
        })
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f64>();
    for weight in &mut kernel {
        *weight /= sum;
    }
    kernel
}

fn reflect_101(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let limit = length as isize;
    let mut index = index;
    while index < 0 || index >= limit {
        index = if index < 0 {
            -index
        } else {
            limit * 2 - index - 2
        };
    }
    index as usize
}

fn gaussian_blur_rgba(source: &[u8], width: usize, height: usize, kernel_size: usize) -> Vec<u8> {
    let radius = (kernel_size / 2) as isize;
    let kernel = gaussian_kernel(kernel_size);
    let mut horizontal = vec![0_f64; source.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                horizontal[(y * width + x) * 4 + channel] = kernel
                    .iter()
                    .enumerate()
                    .map(|(index, weight)| {
                        let sample_x = reflect_101(x as isize + index as isize - radius, width);
                        *weight * f64::from(source[(y * width + sample_x) * 4 + channel])
                    })
                    .sum();
            }
        }
    }

    let mut output = source.to_vec();
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let value = kernel
                    .iter()
                    .enumerate()
                    .map(|(index, weight)| {
                        let sample_y = reflect_101(y as isize + index as isize - radius, height);
                        *weight * horizontal[(sample_y * width + x) * 4 + channel]
                    })
                    .sum();
                output[(y * width + x) * 4 + channel] = clamp_u8_ties_even(value);
            }
        }
    }
    output
}

fn srgb_to_linear(value: u8) -> f64 {
    let value = f64::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn lab_component(value: f64) -> f64 {
    if value > 216.0 / 24_389.0 {
        value.cbrt()
    } else {
        (24_389.0 / 27.0 * value + 16.0) / 116.0
    }
}

fn inverse_lab_component(value: f64) -> f64 {
    let cube = value * value * value;
    if cube > 216.0 / 24_389.0 {
        cube
    } else {
        (116.0 * value - 16.0) / (24_389.0 / 27.0)
    }
}

fn rgb_to_lab(pixel: &[u8]) -> (u8, u8, u8) {
    let r = srgb_to_linear(pixel[0]);
    let g = srgb_to_linear(pixel[1]);
    let b = srgb_to_linear(pixel[2]);
    let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / 0.950_47;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b) / 1.088_83;
    let fx = lab_component(x);
    let fy = lab_component(y);
    let fz = lab_component(z);
    let l = 116.0 * fy - 16.0;
    (
        clamp_u8_ties_even(l * 255.0 / 100.0),
        clamp_u8_ties_even(500.0 * (fx - fy) + 128.0),
        clamp_u8_ties_even(200.0 * (fy - fz) + 128.0),
    )
}

fn linear_to_srgb(value: f64) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let srgb = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    clamp_u8_ties_even(srgb * 255.0)
}

fn lab_to_rgb(l_byte: u8, a_byte: u8, b_byte: u8) -> [u8; 3] {
    let l = f64::from(l_byte) * 100.0 / 255.0;
    let a = f64::from(a_byte) - 128.0;
    let b = f64::from(b_byte) - 128.0;
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let x = 0.950_47 * inverse_lab_component(fx);
    let y = inverse_lab_component(fy);
    let z = 1.088_83 * inverse_lab_component(fz);
    [
        linear_to_srgb(3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z),
        linear_to_srgb(-0.969_266 * x + 1.876_010_8 * y + 0.041_556 * z),
        linear_to_srgb(0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z),
    ]
}

fn apply_clahe_lab(data: &mut [u8], width: usize, height: usize, strength: f64) {
    if width == 0 || height == 0 {
        return;
    }
    const TILES: usize = 8;
    const BINS: usize = 256;
    let tile_width = width.div_ceil(TILES).max(1);
    let tile_height = height.div_ceil(TILES).max(1);
    let extended_width = tile_width * TILES;
    let extended_height = tile_height * TILES;
    let mut l_values = vec![0_u8; extended_width * extended_height];
    let mut chroma = vec![(0_u8, 0_u8); width * height];
    for y in 0..extended_height {
        let source_y = reflect_101(y as isize, height);
        for x in 0..extended_width {
            let source_x = reflect_101(x as isize, width);
            let source_index = (source_y * width + source_x) * 4;
            let (l, a, b) = rgb_to_lab(&data[source_index..source_index + 4]);
            l_values[y * extended_width + x] = l;
            if y < height && x < width {
                chroma[y * width + x] = (a, b);
            }
        }
    }

    let tile_pixels = tile_width * tile_height;
    let clip_limit = ((strength * tile_pixels as f64 / BINS as f64).floor() as usize).max(1);
    let mut lookup = vec![[0_u8; BINS]; TILES * TILES];
    for tile_y in 0..TILES {
        for tile_x in 0..TILES {
            let mut histogram = [0_usize; BINS];
            for y in tile_y * tile_height..(tile_y + 1) * tile_height {
                for x in tile_x * tile_width..(tile_x + 1) * tile_width {
                    histogram[usize::from(l_values[y * extended_width + x])] += 1;
                }
            }
            let mut clipped = 0_usize;
            for count in &mut histogram {
                if *count > clip_limit {
                    clipped += *count - clip_limit;
                    *count = clip_limit;
                }
            }
            let batch = clipped / BINS;
            let residual = clipped - batch * BINS;
            for count in &mut histogram {
                *count += batch;
            }
            if residual > 0 {
                let step = (BINS / residual).max(1);
                for index in (0..BINS).step_by(step).take(residual) {
                    histogram[index] += 1;
                }
            }
            let mut cumulative = 0_usize;
            let table = &mut lookup[tile_y * TILES + tile_x];
            for (index, count) in histogram.iter().enumerate() {
                cumulative += count;
                table[index] = clamp_u8_ties_even(cumulative as f64 * 255.0 / tile_pixels as f64);
            }
        }
    }

    for y in 0..height {
        let tile_y = y as f64 / tile_height as f64 - 0.5;
        let lower_y = tile_y.floor() as isize;
        let upper_y = lower_y + 1;
        let y_weight = tile_y - lower_y as f64;
        let lower_y = lower_y.clamp(0, TILES as isize - 1) as usize;
        let upper_y = upper_y.clamp(0, TILES as isize - 1) as usize;
        for x in 0..width {
            let tile_x = x as f64 / tile_width as f64 - 0.5;
            let lower_x = tile_x.floor() as isize;
            let upper_x = lower_x + 1;
            let x_weight = tile_x - lower_x as f64;
            let lower_x = lower_x.clamp(0, TILES as isize - 1) as usize;
            let upper_x = upper_x.clamp(0, TILES as isize - 1) as usize;
            let source_l = usize::from(l_values[y * extended_width + x]);
            let top = (1.0 - x_weight) * f64::from(lookup[lower_y * TILES + lower_x][source_l])
                + x_weight * f64::from(lookup[lower_y * TILES + upper_x][source_l]);
            let bottom = (1.0 - x_weight) * f64::from(lookup[upper_y * TILES + lower_x][source_l])
                + x_weight * f64::from(lookup[upper_y * TILES + upper_x][source_l]);
            let mapped_l = clamp_u8_ties_even((1.0 - y_weight) * top + y_weight * bottom);
            let (a, b) = chroma[y * width + x];
            let rgb = lab_to_rgb(mapped_l, a, b);
            let index = (y * width + x) * 4;
            data[index..index + 3].copy_from_slice(&rgb);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageData, NativePreprocessor, MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, MAX_RGBA_BYTES};
    use slouch_domain::CameraSettings;

    fn gray(values: &[u8], width: u32, height: u32) -> ImageData {
        ImageData {
            data: values
                .iter()
                .flat_map(|value| [*value, *value, *value, 255])
                .collect(),
            width,
            height,
        }
    }

    #[test]
    fn temporal_smoothing_matches_uint8_clamped_moving_average() {
        let settings = CameraSettings {
            smoothing_frames: 3,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        assert_eq!(
            processor.process(gray(&[0], 1, 1), &settings).unwrap().data[0],
            0
        );
        assert_eq!(
            processor
                .process(gray(&[101], 1, 1), &settings)
                .unwrap()
                .data[0],
            50
        );
        assert_eq!(
            processor
                .process(gray(&[202], 1, 1), &settings)
                .unwrap()
                .data[0],
            101
        );
        assert_eq!(
            processor
                .process(gray(&[100], 1, 1), &settings)
                .unwrap()
                .data[0],
            134
        );
    }

    #[test]
    fn gaussian_blur_matches_frozen_native_fixture_and_preserves_alpha() {
        let settings = CameraSettings {
            smoothing_frames: 1,
            gaussian_blur_kernel: 3,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        let result = processor
            .process(gray(&[0, 64, 128, 255], 2, 2), &settings)
            .unwrap();
        assert_eq!(
            result.data,
            vec![112, 112, 112, 255, 112, 112, 112, 255, 112, 112, 112, 255, 112, 112, 112, 255]
        );
    }

    #[test]
    fn gaussian_kernel_size_nine_uses_computed_gaussian_not_hardcoded_table() {
        // OpenCV's small_gaussian_tab only covers ksize <= 7; ksize == 9 must use
        // the true computed Gaussian (sigma = 0.3*(radius-1)+0.8 = 1.7).
        let kernel = super::gaussian_kernel(9);
        assert_eq!(kernel.len(), 9);

        let radius = 4isize;
        let sigma = 1.7f64;
        let mut expected: Vec<f64> = (0..9)
            .map(|index| {
                let distance = index as isize - radius;
                (-(distance * distance) as f64 / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let sum: f64 = expected.iter().sum();
        for weight in &mut expected {
            *weight /= sum;
        }

        for (actual, want) in kernel.iter().zip(expected.iter()) {
            assert!(
                (actual - want).abs() < 1e-9,
                "kernel weight {actual} diverged from computed Gaussian {want}"
            );
        }

        let center = kernel[4];
        assert!(
            (center - 0.23639).abs() < 1e-4,
            "center weight {center} should match sigma=1.7 Gaussian (~0.2364)"
        );
        assert!(
            (center - 0.234375).abs() > 1e-4,
            "center weight {center} must not equal the spurious hardcoded 60/256"
        );
    }

    #[test]
    fn active_transforms_recreate_opaque_alpha_while_zero_transform_preserves_it() {
        let source = ImageData {
            data: vec![
                10, 20, 30, 0, 40, 50, 60, 127, 70, 80, 90, 1, 100, 110, 120, 255,
            ],
            width: 2,
            height: 2,
        };
        let no_transform = CameraSettings {
            smoothing_frames: 1,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        assert_eq!(
            NativePreprocessor::default()
                .process(source.clone(), &no_transform)
                .unwrap()
                .data
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            vec![0, 127, 1, 255]
        );

        let blur = CameraSettings {
            gaussian_blur_kernel: 1,
            ..no_transform.clone()
        };
        assert!(NativePreprocessor::default()
            .process(source.clone(), &blur)
            .unwrap()
            .data
            .chunks_exact(4)
            .all(|pixel| pixel[3] == 255));

        let clahe = CameraSettings {
            clahe_strength: 3.5,
            ..no_transform
        };
        assert!(NativePreprocessor::default()
            .process(source, &clahe)
            .unwrap()
            .data
            .chunks_exact(4)
            .all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn image_bounds_reject_zero_and_oversized_inputs_without_mutating_history() {
        let settings = CameraSettings {
            smoothing_frames: 2,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        processor.process(gray(&[10], 1, 1), &settings).unwrap();

        assert!(processor
            .process(
                ImageData {
                    data: Vec::new(),
                    width: 0,
                    height: 1,
                },
                &settings,
            )
            .is_err());
        assert!(processor
            .process(
                ImageData {
                    data: vec![0; MAX_RGBA_BYTES + 1],
                    width: MAX_IMAGE_WIDTH,
                    height: MAX_IMAGE_HEIGHT,
                },
                &settings,
            )
            .is_err());

        let output = processor.process(gray(&[30], 1, 1), &settings).unwrap();
        assert_eq!(output.data[0], 20);
    }

    #[test]
    fn clahe_matches_frozen_lab_tile_fixture_and_changes_contrast() {
        let settings = CameraSettings {
            smoothing_frames: 1,
            gaussian_blur_kernel: 0,
            clahe_strength: 3.5,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        let result = processor
            .process(gray(&[16, 32, 64, 128], 2, 2), &settings)
            .unwrap();
        assert_eq!(
            result.data,
            vec![255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        );
    }

    #[test]
    fn combined_preprocessing_matches_frozen_full_frame_digest() {
        let source = ImageData {
            data: (0_u16..256)
                .flat_map(|index| {
                    let x = index % 16;
                    let y = index / 16;
                    [
                        ((x * 13 + y * 3) % 256) as u8,
                        ((x * 5 + y * 17) % 256) as u8,
                        ((x * 19 + y * 7) % 256) as u8,
                        255,
                    ]
                })
                .collect(),
            width: 16,
            height: 16,
        };
        let settings = CameraSettings {
            smoothing_frames: 1,
            gaussian_blur_kernel: 5,
            clahe_strength: 3.5,
            ..CameraSettings::default()
        };
        let output = NativePreprocessor::default()
            .process(source, &settings)
            .unwrap();
        let digest = output
            .data
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            });
        assert_eq!(digest, 15_123_802_116_041_745_608);
    }

    #[test]
    fn changing_each_persisted_setting_changes_native_preprocessing() {
        let source = gray(&[0, 64, 128, 255], 2, 2);
        let baseline_settings = CameraSettings {
            smoothing_frames: 1,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let baseline = NativePreprocessor::default()
            .process(source.clone(), &baseline_settings)
            .unwrap();

        let mut blur_settings = baseline_settings.clone();
        blur_settings.gaussian_blur_kernel = 5;
        assert_ne!(
            NativePreprocessor::default()
                .process(source.clone(), &blur_settings)
                .unwrap()
                .data,
            baseline.data
        );

        let mut clahe_settings = baseline_settings.clone();
        clahe_settings.clahe_strength = 3.5;
        assert_ne!(
            NativePreprocessor::default()
                .process(source.clone(), &clahe_settings)
                .unwrap()
                .data,
            baseline.data
        );

        let mut smooth_settings = baseline_settings;
        smooth_settings.smoothing_frames = 3;
        let mut smooth = NativePreprocessor::default();
        smooth.process(source, &smooth_settings).unwrap();
        assert_ne!(
            smooth
                .process(gray(&[255, 255, 255, 255], 2, 2), &smooth_settings)
                .unwrap()
                .data,
            baseline.data
        );
    }

    #[test]
    fn dimension_and_window_changes_reset_temporal_history() {
        let mut processor = NativePreprocessor::default();
        let mut settings = CameraSettings {
            smoothing_frames: 3,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        processor.process(gray(&[0], 1, 1), &settings).unwrap();
        let changed_dimensions = processor
            .process(gray(&[200, 200], 2, 1), &settings)
            .unwrap();
        assert_eq!(changed_dimensions.data[0], 200);

        settings.smoothing_frames = 1;
        let changed_window = processor.process(gray(&[25, 25], 2, 1), &settings).unwrap();
        assert_eq!(changed_window.data[0], 25);
    }

    #[test]
    fn invalid_rgba_layout_does_not_mutate_valid_smoothing_history() {
        let settings = CameraSettings {
            smoothing_frames: 2,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        processor.process(gray(&[10], 1, 1), &settings).unwrap();
        assert!(processor
            .process(
                ImageData {
                    data: vec![1, 2, 3],
                    width: 1,
                    height: 1,
                },
                &settings,
            )
            .is_err());
        let output = processor.process(gray(&[30], 1, 1), &settings).unwrap();
        assert_eq!(output.data[0], 20);
    }

    #[test]
    fn explicit_reset_discards_temporal_history() {
        let settings = CameraSettings {
            smoothing_frames: 3,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        processor.process(gray(&[0], 1, 1), &settings).unwrap();
        processor.reset();
        let output = processor.process(gray(&[201], 1, 1), &settings).unwrap();
        assert_eq!(output.data[0], 201);
    }

    #[test]
    fn sparse_inference_consumes_camera_rate_smoothing_without_double_ingestion() {
        let settings = CameraSettings {
            smoothing_frames: 3,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        for value in [0, 30, 60] {
            processor
                .ingest_camera_frame(gray(&[value], 1, 1), &settings)
                .unwrap();
        }
        assert_eq!(processor.process_latest(&settings).unwrap().data[0], 30);
        assert_eq!(processor.process_latest(&settings).unwrap().data[0], 30);

        for value in [90, 120, 150] {
            processor
                .ingest_camera_frame(gray(&[value], 1, 1), &settings)
                .unwrap();
        }
        assert_eq!(processor.process_latest(&settings).unwrap().data[0], 120);
    }
}
