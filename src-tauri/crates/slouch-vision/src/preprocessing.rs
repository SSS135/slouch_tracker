use slouch_domain::{ported::messages::schemas::ImageData, CameraSettings};

/// Motion-tile edge length in pixels for the temporal accumulator gate.
const TILE_PX: usize = 16;
/// Diff subsampling stride inside each motion tile. This is the cheapness lever
/// that keeps ingest roughly one pass at camera rate: SAD reads 1/16th of pixels.
const SUBSAMPLE: usize = 4;
/// Normalized L-histogram L1 distance (in [0,1]) that declares a scene cut and
/// snaps the CLAHE tone curves; a raised enter/lowered exit pair latches so a
/// distance hovering near the threshold cannot flap between snap and EMA.
const SCENE_CHANGE_ENTER: f64 = 0.35;
const SCENE_CHANGE_EXIT: f64 = 0.20;

/// Which frame the NLF-L square crop is cut from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlfCropSource {
    /// The motion-gated temporal accumulator (default; already ghost-free).
    Accumulated,
    /// The raw current frame, so the crop is a single uniform noise source.
    Raw,
}

/// Bbox area-fraction of moving tiles at or above which the NLF crop switches to
/// the raw current frame. The accumulator is already ghost-free, so this only
/// refines crop uniformity — a conservative threshold keeps the default path.
pub const NLF_CROP_MOVING_FRACTION_MAX: f64 = 0.15;

/// Snapshot of the per-tile motion grid for the most recent ingest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileMotion {
    pub tile_px: usize,
    pub tiles_x: usize,
    pub tiles_y: usize,
    pub moving: Vec<bool>,
}

/// Decides the NLF crop source from the person bbox and the motion-tile grid.
///
/// Keeps the NLF crop a single uniform noise source: the accumulator is already
/// ghost-free, so this is a uniformity refinement. The threshold is the
/// area-fraction of the bbox that overlaps moving tiles — per overlapping tile
/// the weight is `area(tile ∩ bbox)`, and `fraction = movingWeight / totalWeight`
/// (0 when there is no overlap). At or above [`NLF_CROP_MOVING_FRACTION_MAX`] the
/// crop is cut from the raw frame; otherwise from the accumulator. Any degenerate
/// input (empty grid, mismatched bitmap, non-finite bbox, no overlap) defaults to
/// [`NlfCropSource::Accumulated`].
// Grid geometry (3), the motion bitmap, and the four bbox scalars are all
// distinct inputs; grouping them would only obscure a pure geometry helper.
#[allow(clippy::too_many_arguments)]
pub fn nlf_crop_source_for(
    tile_px: usize,
    tiles_x: usize,
    tiles_y: usize,
    moving: &[bool],
    bbox_x: f64,
    bbox_y: f64,
    bbox_w: f64,
    bbox_h: f64,
) -> NlfCropSource {
    if tile_px == 0 || tiles_x == 0 || tiles_y == 0 || moving.len() != tiles_x * tiles_y {
        return NlfCropSource::Accumulated;
    }
    if ![bbox_x, bbox_y, bbox_w, bbox_h]
        .iter()
        .all(|value| value.is_finite())
    {
        return NlfCropSource::Accumulated;
    }
    let left = bbox_x;
    let top = bbox_y;
    let right = bbox_x + bbox_w.max(0.0);
    let bottom = bbox_y + bbox_h.max(0.0);
    let tile = tile_px as f64;
    let tx_start = (left / tile).floor().max(0.0) as usize;
    let ty_start = (top / tile).floor().max(0.0) as usize;
    let tx_end = ((right / tile).ceil() as isize).clamp(0, tiles_x as isize) as usize;
    let ty_end = ((bottom / tile).ceil() as isize).clamp(0, tiles_y as isize) as usize;
    let mut total_area = 0.0;
    let mut moving_area = 0.0;
    for ty in ty_start.min(tiles_y)..ty_end {
        let tile_top = ty as f64 * tile;
        let overlap_h = (bottom.min(tile_top + tile) - top.max(tile_top)).max(0.0);
        if overlap_h <= 0.0 {
            continue;
        }
        for tx in tx_start.min(tiles_x)..tx_end {
            let tile_left = tx as f64 * tile;
            let overlap_w = (right.min(tile_left + tile) - left.max(tile_left)).max(0.0);
            if overlap_w <= 0.0 {
                continue;
            }
            let area = overlap_w * overlap_h;
            total_area += area;
            if moving[ty * tiles_x + tx] {
                moving_area += area;
            }
        }
    }
    if total_area <= 0.0 {
        return NlfCropSource::Accumulated;
    }
    if moving_area / total_area >= NLF_CROP_MOVING_FRACTION_MAX {
        NlfCropSource::Raw
    } else {
        NlfCropSource::Accumulated
    }
}

/// Stateful native preprocessing applied to raw camera RGBA before inference.
///
/// The temporal stage is a per-tile motion-gated running-mean/EMA accumulator
/// (ghost-free: moving tiles snap to the current frame), followed by optional
/// OpenCV-parity CLAHE whose per-tile tone curves are temporally EMA-filtered
/// to remove frame-to-frame flicker.
#[derive(Debug, Default)]
pub struct NativePreprocessor {
    width: usize,
    height: usize,
    smoothing_frames: u8,
    /// Per-pixel running-mean accumulator, RGBA, len = width*height*4. This is
    /// the denoised buffer that `process_latest` consumes.
    accum: Vec<f32>,
    /// Accumulation depth per motion tile (row-major, tiles_x * tiles_y); a tile
    /// with depth 1 was just reset (moving), higher depths are accumulating.
    counts: Vec<u16>,
    /// EMA-filtered CLAHE tone curves, one 256-entry LUT per 8x8 CLAHE tile.
    lut_ema: Vec<[f32; 256]>,
    lut_ema_valid: bool,
    /// Previous frame's normalized global L histogram, for scene-change detection.
    prev_l_hist: Option<[f32; 256]>,
    scene_change_latched: bool,
    /// Bumped on every ingest; pairs with `last_clahe_gen` so the CLAHE EMA
    /// advances at most once per ingested frame (repeated `process_latest`
    /// without a new frame returns identical bytes).
    generation: u64,
    last_clahe_gen: u64,
    /// Per-tile motion-reset bitmap for the most recent ingest (row-major,
    /// tiles_x * tiles_y). `true` means the tile was reset by motion THIS frame
    /// (SAD over threshold); the first frame after a reset is all `false`.
    /// Distinct from `counts == 1`, which is also true on the fresh first frame.
    moving: Vec<bool>,
    /// Raw RGBA bytes of the most recently ingested frame, before accumulation.
    /// The NLF crop-uniformity rule cuts its crop from these when the person
    /// overlaps enough moving tiles.
    latest_raw: Vec<u8>,
}

impl NativePreprocessor {
    /// Ingests one camera-cadence sample into the motion-gated accumulator. Cheap:
    /// a subsampled per-tile SAD plus one full blend pass. Invalid input never
    /// mutates state.
    pub fn ingest_camera_frame(
        &mut self,
        image: ImageData,
        settings: &CameraSettings,
    ) -> Result<(), String> {
        settings.validate()?;
        validate_image(&image)?;
        let width = image.width as usize;
        let height = image.height as usize;

        // Dimension or smoothing-window changes invalidate every temporal buffer.
        if self.width != width
            || self.height != height
            || self.smoothing_frames != settings.smoothing_frames
        {
            self.reset_state();
        }

        let tiles_x = width.div_ceil(TILE_PX);
        let tiles_y = height.div_ceil(TILE_PX);

        if self.accum.is_empty() {
            // First frame after any reset: nothing to blend against, so snap the
            // accumulator to the frame and mark every tile fresh (depth 1).
            self.width = width;
            self.height = height;
            self.smoothing_frames = settings.smoothing_frames;
            self.accum = image.data.iter().map(|&byte| f32::from(byte)).collect();
            self.counts = vec![1; tiles_x * tiles_y];
            // No prior frame to diff against, so no tile is "reset by motion" yet.
            self.moving = vec![false; tiles_x * tiles_y];
            self.generation = self.generation.wrapping_add(1);
            self.latest_raw = image.data;
            return Ok(());
        }

        let saturation = u16::from(settings.smoothing_frames);
        let mut alpha_tiles = vec![0_f64; tiles_x * tiles_y];
        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let tile = tile_y * tiles_x + tile_x;
                let sad = tile_sad(&image.data, &self.accum, width, height, tile_x, tile_y);
                let is_moving = sad > settings.tile_motion_threshold;
                self.moving[tile] = is_moving;
                if is_moving {
                    // Motion inside the tile: reset depth so the output snaps to
                    // the current frame (alpha == 1) and never ghosts.
                    self.counts[tile] = 1;
                } else {
                    self.counts[tile] = self.counts[tile].saturating_add(1).min(saturation);
                }
                // alpha = 1/depth: exact running mean while depth < N, EMA of depth
                // N once saturated. depth 1 => alpha 1 => pure current frame.
                alpha_tiles[tile] = 1.0 / f64::from(self.counts[tile]);
            }
        }

        // Feather alpha to per-pixel by bilinear interpolation of tile-center
        // values (same convention CLAHE uses) so the noise-level transition across
        // a static/moving boundary is seamless. At a moving tile's interior all
        // corners are alpha 1, so accum == current exactly (ghost-free); only the
        // ~half-tile band at a static/moving boundary blends.
        for y in 0..height {
            for x in 0..width {
                let alpha = bilinear_tile_value(&alpha_tiles, tiles_x, tiles_y, x, y) as f32;
                let base = (y * width + x) * 4;
                for channel in 0..4 {
                    let index = base + channel;
                    let current = f32::from(image.data[index]);
                    self.accum[index] = alpha * current + (1.0 - alpha) * self.accum[index];
                }
            }
        }

        self.generation = self.generation.wrapping_add(1);
        self.latest_raw = image.data;
        Ok(())
    }

    /// The raw RGBA bytes of the most recently ingested frame (before temporal
    /// accumulation), sized to the current dimensions. `None` before any ingest
    /// or after a reset.
    pub fn latest_raw_frame(&self) -> Option<ImageData> {
        if self.latest_raw.is_empty() {
            return None;
        }
        Some(ImageData {
            data: self.latest_raw.clone(),
            width: self.width as u32,
            height: self.height as u32,
        })
    }

    /// The per-tile motion grid from the most recent ingest. `None` before any
    /// ingest or after a reset.
    pub fn tile_motion(&self) -> Option<TileMotion> {
        if self.moving.is_empty() {
            return None;
        }
        Some(TileMotion {
            tile_px: TILE_PX,
            tiles_x: self.width.div_ceil(TILE_PX),
            tiles_y: self.height.div_ceil(TILE_PX),
            moving: self.moving.clone(),
        })
    }

    /// Convenience over [`nlf_crop_source_for`] using this preprocessor's own
    /// motion grid and dimensions; `(x, y, w, h)` is the person bbox in pixels.
    pub fn nlf_crop_source(&self, x: f64, y: f64, w: f64, h: f64) -> NlfCropSource {
        nlf_crop_source_for(
            TILE_PX,
            self.width.div_ceil(TILE_PX),
            self.height.div_ceil(TILE_PX),
            &self.moving,
            x,
            y,
            w,
            h,
        )
    }

    /// Returns the denoised accumulator with the configured spatial transforms.
    /// `&mut self` because the CLAHE tone-curve EMA advances here (at most once
    /// per ingested frame — repeated calls without a new frame are idempotent).
    pub fn process_latest(&mut self, settings: &CameraSettings) -> Result<ImageData, String> {
        settings.validate()?;
        if self.accum.is_empty() {
            return Err("native preprocessing buffer is empty".to_owned());
        }
        let width = self.width;
        let height = self.height;
        let mut data: Vec<u8> = self
            .accum
            .iter()
            .map(|&value| clamp_u8_ties_even(f64::from(value)))
            .collect();

        if settings.clahe_strength > 0.0 {
            let build = clahe_build(&data, width, height, settings.clahe_strength);
            let applied =
                self.advance_clahe_ema(&build, width, height, settings.clahe_temporal_alpha);
            clahe_apply(&mut data, width, height, &build, &applied);
        }
        // OpenCV's RGBA2RGB/RGB2RGBA round trip discards source alpha and
        // recreates an opaque channel whenever CLAHE is active.
        if settings.clahe_strength > 0.0 {
            for alpha in data.iter_mut().skip(3).step_by(4) {
                *alpha = 255;
            }
        }
        Ok(ImageData {
            data,
            width: width as u32,
            height: height as u32,
        })
    }

    /// One-shot entry used by tests and callers that infer on every frame. Camera
    /// integrations call `ingest_camera_frame` per sample and `process_latest`
    /// only when an inference request is due.
    pub fn process(
        &mut self,
        image: ImageData,
        settings: &CameraSettings,
    ) -> Result<ImageData, String> {
        self.ingest_camera_frame(image, settings)?;
        self.process_latest(settings)
    }

    /// Tints `base` by per-tile accumulation depth (red = just reset/moving, green
    /// = fully accumulated) using bilinearly-interpolated depth for a smooth
    /// heatmap alpha-blended over `base`, with thin tile grid lines for readability.
    pub fn render_debug_tiles(&self, base: &ImageData) -> Result<ImageData, String> {
        if base.width as usize != self.width || base.height as usize != self.height {
            return Err("debug base frame dimensions do not match preprocessor state".to_owned());
        }
        if self.counts.is_empty() {
            return Err("preprocessor has no accumulated frame to visualize".to_owned());
        }
        let width = self.width;
        let height = self.height;
        let tiles_x = width.div_ceil(TILE_PX);
        let tiles_y = height.div_ceil(TILE_PX);
        let denominator = (f64::from(self.smoothing_frames) - 1.0).max(1.0);
        let depth: Vec<f64> = self
            .counts
            .iter()
            .map(|&count| ((f64::from(count) - 1.0) / denominator).clamp(0.0, 1.0))
            .collect();
        let mut data = base.data.clone();
        const TINT: f64 = 0.4;
        for y in 0..height {
            let grid_y = y % TILE_PX == 0;
            for x in 0..width {
                let normalized =
                    bilinear_tile_value(&depth, tiles_x, tiles_y, x, y).clamp(0.0, 1.0);
                let index = (y * width + x) * 4;
                if grid_y || x % TILE_PX == 0 {
                    // Thin tile boundary line: darken so the grid reads over any base.
                    data[index] = (f64::from(data[index]) * 0.35) as u8;
                    data[index + 1] = (f64::from(data[index + 1]) * 0.35) as u8;
                    data[index + 2] = (f64::from(data[index + 2]) * 0.35) as u8;
                    data[index + 3] = 255;
                    continue;
                }
                let tint_r = (1.0 - normalized) * 255.0;
                let tint_g = normalized * 255.0;
                let tint_b = 40.0;
                data[index] =
                    clamp_u8_ties_even((1.0 - TINT) * f64::from(data[index]) + TINT * tint_r);
                data[index + 1] =
                    clamp_u8_ties_even((1.0 - TINT) * f64::from(data[index + 1]) + TINT * tint_g);
                data[index + 2] =
                    clamp_u8_ties_even((1.0 - TINT) * f64::from(data[index + 2]) + TINT * tint_b);
                data[index + 3] = 255;
            }
        }
        Ok(ImageData {
            data,
            width: base.width,
            height: base.height,
        })
    }

    pub fn reset(&mut self) {
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.width = 0;
        self.height = 0;
        self.smoothing_frames = 0;
        self.accum.clear();
        self.counts.clear();
        self.lut_ema.clear();
        self.lut_ema_valid = false;
        self.prev_l_hist = None;
        self.scene_change_latched = false;
        self.generation = 0;
        self.last_clahe_gen = 0;
        self.moving.clear();
        self.latest_raw.clear();
    }

    /// Advances the per-tile CLAHE tone-curve EMA at most once per ingested frame
    /// and returns the LUTs to apply this frame. EMA-ing the LUT (not the raw
    /// histogram) is deliberate: the tone curve is the quantity that flickers, an
    /// EMA of monotone LUTs stays monotone (convex combination), and it decouples
    /// the smoothing from the clip-redistribute nonlinearity.
    fn advance_clahe_ema(
        &mut self,
        build: &ClaheBuild,
        width: usize,
        height: usize,
        alpha: f64,
    ) -> Vec<[u8; 256]> {
        if !self.lut_ema_valid {
            // First CLAHE frame after a reset: seed the EMA with the current tone
            // curves so the applied LUT is byte-identical to per-frame CLAHE
            // (preserves every frozen parity fixture; alpha == 1 stays exact too).
            self.lut_ema = build.luts.iter().map(lut_to_f32).collect();
            self.lut_ema_valid = true;
            self.prev_l_hist = Some(clahe_l_histogram(
                &build.l_values,
                build.extended_width,
                width,
                height,
            ));
            self.last_clahe_gen = self.generation;
        } else if self.generation != self.last_clahe_gen {
            let current_hist =
                clahe_l_histogram(&build.l_values, build.extended_width, width, height);
            let distance = match &self.prev_l_hist {
                Some(previous) => l_histogram_distance(previous, &current_hist),
                None => 1.0,
            };
            let weight = alpha as f32;
            if self.scene_change(distance) {
                // A real lighting change must not lag: snap the tone curves.
                for (tile, lut) in build.luts.iter().enumerate() {
                    self.lut_ema[tile] = lut_to_f32(lut);
                }
            } else {
                for (tile, lut) in build.luts.iter().enumerate() {
                    let ema = &mut self.lut_ema[tile];
                    for (smoothed, &value) in ema.iter_mut().zip(lut.iter()) {
                        *smoothed = weight * f32::from(value) + (1.0 - weight) * *smoothed;
                    }
                }
            }
            self.prev_l_hist = Some(current_hist);
            self.last_clahe_gen = self.generation;
        }
        self.lut_ema
            .iter()
            .map(|lut| {
                let mut applied = [0_u8; 256];
                for (destination, &value) in applied.iter_mut().zip(lut.iter()) {
                    *destination = clamp_u8_ties_even(f64::from(value));
                }
                applied
            })
            .collect()
    }

    fn scene_change(&mut self, distance: f64) -> bool {
        if self.scene_change_latched {
            if distance < SCENE_CHANGE_EXIT {
                self.scene_change_latched = false;
                return false;
            }
            return true;
        }
        if distance >= SCENE_CHANGE_ENTER {
            self.scene_change_latched = true;
            return true;
        }
        false
    }
}

fn luma_u8(pixel: &[u8]) -> f64 {
    0.299 * f64::from(pixel[0]) + 0.587 * f64::from(pixel[1]) + 0.114 * f64::from(pixel[2])
}

fn luma_f32(pixel: &[f32]) -> f64 {
    0.299 * f64::from(pixel[0]) + 0.587 * f64::from(pixel[1]) + 0.114 * f64::from(pixel[2])
}

/// Per-tile DC-subtracted luminance SAD over a subsampled grid. Subtracting each
/// tile's own mean makes auto-exposure / brightness drift NOT read as motion;
/// only a change in the tile's spatial pattern does.
fn tile_sad(
    current: &[u8],
    accum: &[f32],
    width: usize,
    height: usize,
    tile_x: usize,
    tile_y: usize,
) -> f64 {
    let x0 = tile_x * TILE_PX;
    let x1 = ((tile_x + 1) * TILE_PX).min(width);
    let y0 = tile_y * TILE_PX;
    let y1 = ((tile_y + 1) * TILE_PX).min(height);
    let mut samples = 0usize;
    let mut current_sum = 0.0;
    let mut accum_sum = 0.0;
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            let index = (y * width + x) * 4;
            current_sum += luma_u8(&current[index..]);
            accum_sum += luma_f32(&accum[index..]);
            samples += 1;
            x += SUBSAMPLE;
        }
        y += SUBSAMPLE;
    }
    if samples == 0 {
        return 0.0;
    }
    let current_mean = current_sum / samples as f64;
    let accum_mean = accum_sum / samples as f64;
    let mut sad = 0.0;
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            let index = (y * width + x) * 4;
            let current_dev = luma_u8(&current[index..]) - current_mean;
            let accum_dev = luma_f32(&accum[index..]) - accum_mean;
            sad += (current_dev - accum_dev).abs();
            x += SUBSAMPLE;
        }
        y += SUBSAMPLE;
    }
    sad / samples as f64
}

/// Bilinear interpolation of a per-tile scalar grid to pixel `(x, y)` using the
/// tile-center convention `apply`/CLAHE share: `t = coord/tile_px - 0.5`.
fn bilinear_tile_value(values: &[f64], tiles_x: usize, tiles_y: usize, x: usize, y: usize) -> f64 {
    let fx = x as f64 / TILE_PX as f64 - 0.5;
    let lower_x = fx.floor();
    let x_weight = fx - lower_x;
    let left = (lower_x as isize).clamp(0, tiles_x as isize - 1) as usize;
    let right = (lower_x as isize + 1).clamp(0, tiles_x as isize - 1) as usize;
    let fy = y as f64 / TILE_PX as f64 - 0.5;
    let lower_y = fy.floor();
    let y_weight = fy - lower_y;
    let top_row = (lower_y as isize).clamp(0, tiles_y as isize - 1) as usize;
    let bottom_row = (lower_y as isize + 1).clamp(0, tiles_y as isize - 1) as usize;
    let top = (1.0 - x_weight) * values[top_row * tiles_x + left]
        + x_weight * values[top_row * tiles_x + right];
    let bottom = (1.0 - x_weight) * values[bottom_row * tiles_x + left]
        + x_weight * values[bottom_row * tiles_x + right];
    (1.0 - y_weight) * top + y_weight * bottom
}

fn lut_to_f32(lut: &[u8; 256]) -> [f32; 256] {
    let mut out = [0_f32; 256];
    for (destination, &value) in out.iter_mut().zip(lut.iter()) {
        *destination = f32::from(value);
    }
    out
}

/// Normalized (sum ≈ 1) global histogram of the L channel over the real image
/// region, used for scene-change detection.
fn clahe_l_histogram(
    l_values: &[u8],
    extended_width: usize,
    width: usize,
    height: usize,
) -> [f32; 256] {
    let mut histogram = [0_f32; 256];
    for y in 0..height {
        for x in 0..width {
            histogram[usize::from(l_values[y * extended_width + x])] += 1.0;
        }
    }
    let total = (width * height) as f32;
    if total > 0.0 {
        for bin in &mut histogram {
            *bin /= total;
        }
    }
    histogram
}

/// L1 distance between two normalized histograms, mapped to [0,1] (half the raw
/// L1, whose maximum is 2 for disjoint distributions).
fn l_histogram_distance(a: &[f32; 256], b: &[f32; 256]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .sum();
    sum * 0.5
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

struct ClaheBuild {
    l_values: Vec<u8>,
    extended_width: usize,
    chroma: Vec<(u8, u8)>,
    luts: Vec<[u8; 256]>,
}

/// Builds the CLAHE per-tile 256-entry tone-curve LUTs (clip / redistribute /
/// CDF), exactly as the OpenCV-parity pipeline, without applying them.
fn clahe_build(data: &[u8], width: usize, height: usize, strength: f64) -> ClaheBuild {
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
    let mut luts = vec![[0_u8; BINS]; TILES * TILES];
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
            let table = &mut luts[tile_y * TILES + tile_x];
            for (index, count) in histogram.iter().enumerate() {
                cumulative += count;
                table[index] = clamp_u8_ties_even(cumulative as f64 * 255.0 / tile_pixels as f64);
            }
        }
    }
    ClaheBuild {
        l_values,
        extended_width,
        chroma,
        luts,
    }
}

/// Applies the given per-tile LUTs to the L channel with bilinear LUT
/// interpolation across pixels, recombining with the stored chroma.
fn clahe_apply(
    data: &mut [u8],
    width: usize,
    height: usize,
    build: &ClaheBuild,
    applied: &[[u8; 256]],
) {
    const TILES: usize = 8;
    let tile_width = width.div_ceil(TILES).max(1);
    let tile_height = height.div_ceil(TILES).max(1);
    let extended_width = build.extended_width;
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
            let source_l = usize::from(build.l_values[y * extended_width + x]);
            let top = (1.0 - x_weight) * f64::from(applied[lower_y * TILES + lower_x][source_l])
                + x_weight * f64::from(applied[lower_y * TILES + upper_x][source_l]);
            let bottom = (1.0 - x_weight) * f64::from(applied[upper_y * TILES + lower_x][source_l])
                + x_weight * f64::from(applied[upper_y * TILES + upper_x][source_l]);
            let mapped_l = clamp_u8_ties_even((1.0 - y_weight) * top + y_weight * bottom);
            let (a, b) = build.chroma[y * width + x];
            let rgb = lab_to_rgb(mapped_l, a, b);
            let index = (y * width + x) * 4;
            data[index..index + 3].copy_from_slice(&rgb);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        nlf_crop_source_for, ImageData, NativePreprocessor, NlfCropSource, MAX_IMAGE_HEIGHT,
        MAX_IMAGE_WIDTH, MAX_RGBA_BYTES, NLF_CROP_MOVING_FRACTION_MAX,
    };
    use slouch_domain::CameraSettings;
    use std::time::Instant;

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

    fn solid(value: u8, width: u32, height: u32) -> ImageData {
        ImageData {
            data: (0..width * height)
                .flat_map(|_| [value, value, value, 255])
                .collect(),
            width,
            height,
        }
    }

    #[test]
    fn motion_gated_ema_matches_running_mean_then_saturates() {
        // Semantic upgrade from a sliding-window SMA to a depth-N motion-gated EMA:
        // a 1x1 image has no spatial pattern, so DC-subtracted SAD is always ~0 and
        // the single tile stays static. counts climb 1,2,3 -> alpha 1,1/2,1/3, which
        // is the exact running mean for the first N frames (0,50,101). The FOURTH
        // frame saturates at depth 3: 100 blended with alpha 1/3 over the prior mean
        // 101 gives ~100.67 -> 101 (the old windowed SMA produced 134).
        let settings = CameraSettings {
            smoothing_frames: 3,
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
            101
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
    fn clahe_full_frame_matches_frozen_digest() {
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
        assert_eq!(digest, 3_257_242_674_026_246_332);
    }

    #[test]
    fn changing_each_persisted_setting_changes_native_preprocessing() {
        let source = gray(&[0, 64, 128, 255], 2, 2);
        let baseline_settings = CameraSettings {
            smoothing_frames: 1,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let baseline = NativePreprocessor::default()
            .process(source.clone(), &baseline_settings)
            .unwrap();

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
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        for value in [0, 30, 60] {
            processor
                .ingest_camera_frame(gray(&[value], 1, 1), &settings)
                .unwrap();
        }
        // 1x1 tile stays static: counts 1,2,3 -> exact running mean of 0,30,60 = 30.
        assert_eq!(processor.process_latest(&settings).unwrap().data[0], 30);
        // Idempotent: no new ingest, so a repeated call returns identical bytes.
        assert_eq!(processor.process_latest(&settings).unwrap().data[0], 30);

        for value in [90, 120, 150] {
            processor
                .ingest_camera_frame(gray(&[value], 1, 1), &settings)
                .unwrap();
        }
        // Saturated depth-3 EMA (alpha 1/3 each step over 30): 50, 73.3, 98.9 -> 99.
        // The old sliding-window SMA produced 120.
        assert_eq!(processor.process_latest(&settings).unwrap().data[0], 99);
    }

    #[test]
    fn static_region_converges_reduces_noise_variance() {
        // A spatially-uniform image alternating in brightness never trips the
        // DC-subtracted gate, so tiles stay static and the depth-N EMA damps the
        // ±20 swing toward the mean.
        let settings = CameraSettings {
            smoothing_frames: 5,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        let mut outputs = Vec::new();
        for frame in 0..24 {
            let value = if frame % 2 == 0 { 100 } else { 140 };
            outputs.push(
                processor
                    .process(solid(value, 4, 4), &settings)
                    .unwrap()
                    .data[0],
            );
        }
        let tail = &outputs[16..];
        let range = tail.iter().max().unwrap() - tail.iter().min().unwrap();
        assert!(
            range < 20,
            "smoothed output range {range} should be far below the raw 40 swing"
        );
    }

    #[test]
    fn moving_block_center_snaps_byte_exact_to_current_frame() {
        // A 3x3 moving tile block surrounded by static tiles: the center tile (2,2)
        // has only moving neighbors, so its feathered alpha is 1 everywhere and its
        // output equals the current frame byte-for-byte (the ghost-free guarantee).
        let settings = CameraSettings {
            smoothing_frames: 4,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let width = 80usize;
        let height = 80usize;
        let mut processor = NativePreprocessor::default();
        processor
            .ingest_camera_frame(solid(100, width as u32, height as u32), &settings)
            .unwrap();

        // Frame 2: uniform 100 except a strong checker in the central 3x3 tile
        // block. The checker cell is 4px so the pattern is visible at the stride-4
        // SAD subsample grid (a period-1 checker would alias to a constant sample).
        let mut data = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let in_block = (16..64).contains(&x) && (16..64).contains(&y);
                let value = if in_block {
                    if (x / 4 + y / 4) % 2 == 0 {
                        200
                    } else {
                        50
                    }
                } else {
                    100
                };
                let index = (y * width + x) * 4;
                data[index] = value;
                data[index + 1] = value;
                data[index + 2] = value;
                data[index + 3] = 255;
            }
        }
        let current = ImageData {
            data,
            width: width as u32,
            height: height as u32,
        };
        let output = processor.process(current.clone(), &settings).unwrap();

        let mut differs_from_prior = false;
        for y in 32..48 {
            for x in 32..48 {
                let index = (y * width + x) * 4;
                assert_eq!(
                    output.data[index..index + 3],
                    current.data[index..index + 3],
                    "center-tile pixel ({x},{y}) must equal the current frame exactly"
                );
                if current.data[index] != 100 {
                    differs_from_prior = true;
                }
            }
        }
        assert!(
            differs_from_prior,
            "center tile must actually differ from the prior accumulator"
        );
    }

    #[test]
    fn tile_motion_threshold_gates_accumulate_versus_reset() {
        // Build a single 16x16 tile whose subsampled SAD lands just below vs just
        // above the default threshold of 1.5. A static tile blends with alpha 0.5
        // over the uniform-128 accumulator; a moving tile resets to alpha 1 and
        // snaps the output to the current frame exactly.
        let settings = CameraSettings {
            smoothing_frames: 3,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };

        // Sampled columns 0,4 fall in the left (high) half, 8,12 in the right (low)
        // half, so SAD = (high - low)/2 against a uniform accumulator.
        let split_tile = |high: u8, low: u8| {
            let mut data = vec![0u8; 16 * 16 * 4];
            for y in 0..16 {
                for x in 0..16 {
                    let value = if x < 8 { high } else { low };
                    let index = (y * 16 + x) * 4;
                    data[index] = value;
                    data[index + 1] = value;
                    data[index + 2] = value;
                    data[index + 3] = 255;
                }
            }
            ImageData {
                data,
                width: 16,
                height: 16,
            }
        };

        // SAD = 1.0 (< 1.5): tile stays static -> output[0] = 0.5*130 + 0.5*128 = 129.
        let mut below = NativePreprocessor::default();
        below
            .ingest_camera_frame(solid(128, 16, 16), &settings)
            .unwrap();
        let below_out = below.process(split_tile(130, 128), &settings).unwrap();
        assert_eq!(below_out.data[0], 129);

        // SAD = 2.0 (> 1.5): tile resets (moving) -> output[0] = current pixel 131.
        let mut above = NativePreprocessor::default();
        above
            .ingest_camera_frame(solid(128, 16, 16), &settings)
            .unwrap();
        let above_out = above.process(split_tile(131, 127), &settings).unwrap();
        assert_eq!(above_out.data[0], 131);
    }

    #[test]
    fn accumulation_depth_saturates_at_smoothing_frames() {
        // With smoothing_frames = 2, a static tile's depth may not climb past 2, so
        // a fifth static frame still blends with alpha 1/2 (not a growing 1/k mean).
        let settings = CameraSettings {
            smoothing_frames: 2,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        for _ in 0..4 {
            processor
                .ingest_camera_frame(gray(&[0], 1, 1), &settings)
                .unwrap();
        }
        let output = processor.process(gray(&[60], 1, 1), &settings).unwrap();
        // Depth-2 EMA: 0.5*60 + 0.5*0 = 30. An unbounded running mean would give 12.
        assert_eq!(output.data[0], 30);
    }

    #[test]
    fn smoothing_frames_one_is_exact_passthrough() {
        // smoothing_frames = 1 forces every tile's depth to 1 -> alpha 1 -> the
        // output equals the current frame exactly, regardless of motion.
        let settings = CameraSettings {
            smoothing_frames: 1,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        let frames = [
            solid(10, 32, 32),
            solid(200, 32, 32),
            gray(
                &(0..1024).map(|i| (i % 256) as u8).collect::<Vec<_>>(),
                32,
                32,
            ),
            solid(77, 32, 32),
        ];
        for frame in frames {
            let output = processor.process(frame.clone(), &settings).unwrap();
            assert_eq!(output.data, frame.data, "N=1 must be exact passthrough");
        }
    }

    #[test]
    fn resolution_change_resets_accumulation_and_clahe_ema() {
        let settings = CameraSettings {
            smoothing_frames: 3,
            clahe_strength: 3.5,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        for _ in 0..3 {
            processor.process(solid(120, 32, 32), &settings).unwrap();
        }
        // Switch resolution: accumulator and CLAHE EMA must both reset so the first
        // frame at the new size equals a fresh single-frame CLAHE of it.
        let resized = solid(90, 16, 16);
        let streamed = processor.process(resized.clone(), &settings).unwrap();
        let fresh = NativePreprocessor::default()
            .process(resized, &settings)
            .unwrap();
        assert_eq!(streamed.data, fresh.data);
    }

    #[test]
    fn feather_makes_static_to_moving_seam_gradual() {
        // 64x16 = four 16px tiles. Left half is uniform 160 over a uniform-100
        // accumulator (DC shift only -> static, alpha 0.5); right half is a strong
        // stripe pattern (moving, alpha 1). In the left region the current frame is
        // uniform, so output = 100 + 60*alpha directly encodes the feathered alpha.
        let settings = CameraSettings {
            smoothing_frames: 4,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let width = 64usize;
        let height = 16usize;
        let mut processor = NativePreprocessor::default();
        processor
            .ingest_camera_frame(solid(100, width as u32, height as u32), &settings)
            .unwrap();

        // Right half is 4px stripes (visible at the stride-4 SAD grid) so its tiles
        // register as moving; left half is uniform 160 over the uniform-100
        // accumulator (DC shift only -> static).
        let mut data = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let value = if x < 32 {
                    160
                } else if (x / 4) % 2 == 0 {
                    200
                } else {
                    50
                };
                let index = (y * width + x) * 4;
                data[index] = value;
                data[index + 1] = value;
                data[index + 2] = value;
                data[index + 3] = 255;
            }
        }
        let output = processor
            .process(
                ImageData {
                    data,
                    width: width as u32,
                    height: height as u32,
                },
                &settings,
            )
            .unwrap();

        // Read the left (uniform-current) region on one row; its values rise as the
        // moving tile is approached. The per-pixel step is tiny relative to the
        // hard-edge gap (60 * 0.5 = 30) a non-feathered gate would produce.
        let row = 8usize;
        let mut previous = None::<i32>;
        let mut max_step = 0i32;
        let mut min_value = 255i32;
        let mut max_value = 0i32;
        for x in 0..32 {
            let value = output.data[(row * width + x) * 4] as i32;
            min_value = min_value.min(value);
            max_value = max_value.max(value);
            if let Some(previous) = previous {
                max_step = max_step.max((value - previous).abs());
            }
            previous = Some(value);
        }
        assert!(
            max_step <= 4,
            "seam transition step {max_step} should be gradual, far below the hard-edge 30"
        );
        assert!(
            max_value - min_value >= 6,
            "the feather band should span a visible range, got {}",
            max_value - min_value
        );
    }

    #[test]
    fn clahe_ema_converges_on_constant_input() {
        let settings = CameraSettings {
            smoothing_frames: 3,
            clahe_strength: 3.5,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        let source = ImageData {
            data: (0..16 * 16)
                .flat_map(|i| {
                    let value = ((i * 7) % 200) as u8;
                    [value, value / 2, 255 - value, 255]
                })
                .collect(),
            width: 16,
            height: 16,
        };
        let mut last = Vec::new();
        let mut previous = Vec::new();
        for _ in 0..20 {
            previous = std::mem::take(&mut last);
            last = processor.process(source.clone(), &settings).unwrap().data;
        }
        assert_eq!(
            last, previous,
            "constant input must converge to identical output"
        );
    }

    #[test]
    fn clahe_ema_lowers_flicker_versus_per_frame() {
        // Two 4px-wide tile columns (Q, R) swap content between two brightness
        // levels every frame while the rest of the frame stays a constant 128. The
        // swap leaves the GLOBAL L histogram unchanged (so no scene snap) but flips
        // each column's per-tile histogram, so their tone-curve LUTs flicker. The
        // constant-128 background borders those columns, so its CLAHE output
        // flickers only through the bilinear LUT interpolation — pure tone-curve
        // flicker with zero content change. The LUT EMA (alpha 0.15) must damp it
        // below the per-frame curve (alpha 1.0). Content-flicker columns Q and R
        // are excluded from the measurement so only tone-curve flicker is scored.
        let base_settings = CameraSettings {
            smoothing_frames: 1,
            clahe_strength: 3.5,
            ..CameraSettings::default()
        };
        let width = 32usize;
        let height = 32usize;
        let in_q = |x: usize| (4..8).contains(&x);
        let in_r = |x: usize| (24..28).contains(&x);
        let frame = |phase_b: bool| {
            let mut data = vec![0u8; width * height * 4];
            for y in 0..height {
                for x in 0..width {
                    let value = if in_q(x) {
                        if phase_b {
                            200
                        } else {
                            40
                        }
                    } else if in_r(x) {
                        if phase_b {
                            40
                        } else {
                            200
                        }
                    } else {
                        128
                    };
                    let index = (y * width + x) * 4;
                    data[index] = value;
                    data[index + 1] = value;
                    data[index + 2] = value;
                    data[index + 3] = 255;
                }
            }
            ImageData {
                data,
                width: width as u32,
                height: height as u32,
            }
        };

        let variance = |alpha: f64| {
            let settings = CameraSettings {
                clahe_temporal_alpha: alpha,
                ..base_settings.clone()
            };
            let mut processor = NativePreprocessor::default();
            let mut frames = Vec::new();
            for i in 0..16 {
                let output = processor.process(frame(i % 2 == 1), &settings).unwrap();
                if i >= 2 {
                    frames.push(output.data);
                }
            }
            let mut total = 0.0;
            for y in 0..height {
                for x in 0..width {
                    if in_q(x) || in_r(x) {
                        continue;
                    }
                    let base = (y * width + x) * 4;
                    for channel in 0..3 {
                        let index = base + channel;
                        let values: Vec<f64> = frames.iter().map(|f| f64::from(f[index])).collect();
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        total += values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
                    }
                }
            }
            total
        };

        let smoothed = variance(0.15);
        let per_frame = variance(1.0);
        assert!(
            smoothed < per_frame,
            "LUT EMA (alpha 0.15) variance {smoothed} should be below per-frame {per_frame}"
        );
    }

    #[test]
    fn clahe_alpha_one_reproduces_per_frame_across_multiple_frames() {
        // alpha == 1.0 must equal per-frame CLAHE on EVERY frame, not just the
        // first. smoothing_frames = 1 makes the accumulator equal the current
        // frame, so each streamed output must match a fresh single-frame CLAHE.
        let settings = CameraSettings {
            smoothing_frames: 1,
            clahe_strength: 3.5,
            clahe_temporal_alpha: 1.0,
            ..CameraSettings::default()
        };
        let mut streaming = NativePreprocessor::default();
        let frames = [
            solid(30, 24, 24),
            gray(
                &(0..576).map(|i| (i % 256) as u8).collect::<Vec<_>>(),
                24,
                24,
            ),
            solid(180, 24, 24),
            gray(
                &(0..576).map(|i| ((i * 5) % 256) as u8).collect::<Vec<_>>(),
                24,
                24,
            ),
        ];
        for frame in frames {
            let streamed = streaming.process(frame.clone(), &settings).unwrap();
            let fresh = NativePreprocessor::default()
                .process(frame, &settings)
                .unwrap();
            assert_eq!(streamed.data, fresh.data);
        }
    }

    #[test]
    fn clahe_scene_change_snaps_without_lag() {
        // A large global luminance jump must snap the tone curves so the output
        // immediately equals a fresh per-frame CLAHE of the new scene.
        let settings = CameraSettings {
            smoothing_frames: 1,
            clahe_strength: 3.5,
            clahe_temporal_alpha: 0.15,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        processor.process(solid(30, 24, 24), &settings).unwrap();
        processor.process(solid(30, 24, 24), &settings).unwrap();
        let bright = gray(
            &(0..576).map(|i| (180 + (i % 40)) as u8).collect::<Vec<_>>(),
            24,
            24,
        );
        let streamed = processor.process(bright.clone(), &settings).unwrap();
        let fresh = NativePreprocessor::default()
            .process(bright, &settings)
            .unwrap();
        assert_eq!(
            streamed.data, fresh.data,
            "scene change must not lag behind the new scene"
        );
    }

    #[test]
    fn render_debug_tiles_heatmaps_accumulation_depth() {
        // The debug heatmap requires an accumulated frame and matching base dims,
        // and tints toward green as tiles saturate.
        let settings = CameraSettings {
            smoothing_frames: 4,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        assert!(processor.render_debug_tiles(&solid(128, 32, 32)).is_err());

        for _ in 0..3 {
            processor.process(solid(128, 32, 32), &settings).unwrap();
        }
        let base = processor.process(solid(128, 32, 32), &settings).unwrap();
        // A fully static run drives every tile to full depth -> green-dominant tint.
        let heatmap = processor.render_debug_tiles(&base).unwrap();
        assert_eq!(heatmap.width, 32);
        assert_eq!(heatmap.height, 32);
        // Interior pixel (not on a grid line): green channel exceeds red.
        let index = (10 * 32 + 10) * 4;
        assert!(heatmap.data[index + 1] > heatmap.data[index]);

        // Mismatched base dimensions are rejected.
        assert!(processor.render_debug_tiles(&solid(128, 16, 16)).is_err());
    }

    #[test]
    fn perf_sanity_720p_ingest_and_full_pipeline() {
        let settings = CameraSettings {
            smoothing_frames: 3,
            clahe_strength: 3.5,
            ..CameraSettings::default()
        };
        let width = 1280usize;
        let height = 720usize;
        let make_frame = |seed: u32| {
            let mut data = vec![0u8; width * height * 4];
            let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            for byte in data.chunks_exact_mut(4) {
                for channel in byte.iter_mut().take(3) {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *channel = (state >> 24) as u8;
                }
                byte[3] = 255;
            }
            ImageData {
                data,
                width: width as u32,
                height: height as u32,
            }
        };

        let mut processor = NativePreprocessor::default();
        processor
            .ingest_camera_frame(make_frame(1), &settings)
            .unwrap();

        let ingest_start = Instant::now();
        processor
            .ingest_camera_frame(make_frame(2), &settings)
            .unwrap();
        let ingest_elapsed = ingest_start.elapsed();

        let full_start = Instant::now();
        processor
            .ingest_camera_frame(make_frame(3), &settings)
            .unwrap();
        processor.process_latest(&settings).unwrap();
        let full_elapsed = full_start.elapsed();

        println!("perf_sanity 720p ingest-only: {ingest_elapsed:?}");
        println!("perf_sanity 720p ingest+process_latest: {full_elapsed:?}");

        assert!(
            ingest_elapsed.as_millis() < 400,
            "ingest {ingest_elapsed:?} exceeded the loose 400ms bound"
        );
        assert!(
            full_elapsed.as_millis() < 2000,
            "full pipeline {full_elapsed:?} exceeded the loose 2000ms bound"
        );
    }

    #[test]
    fn nlf_crop_source_all_static_bbox_uses_accumulator() {
        // 4x4 tile grid, no tile moving: the crop stays on the ghost-free accumulator.
        let moving = vec![false; 16];
        assert_eq!(
            nlf_crop_source_for(16, 4, 4, &moving, 0.0, 0.0, 64.0, 64.0),
            NlfCropSource::Accumulated
        );
    }

    #[test]
    fn nlf_crop_source_moving_dominated_bbox_uses_raw() {
        // Every tile the bbox covers is moving -> fraction 1.0 -> raw current frame.
        let moving = vec![true; 16];
        assert_eq!(
            nlf_crop_source_for(16, 4, 4, &moving, 0.0, 0.0, 64.0, 64.0),
            NlfCropSource::Raw
        );
    }

    #[test]
    fn nlf_crop_source_threshold_is_area_fraction_boundary() {
        // 20 equal tiles fully covered by the bbox. 3/20 = 0.15 hits the >= MAX
        // boundary exactly (raw); 2/20 = 0.10 stays on the accumulator.
        assert!((NLF_CROP_MOVING_FRACTION_MAX - 0.15).abs() < 1e-12);
        let mut three_moving = vec![false; 20];
        for flag in three_moving.iter_mut().take(3) {
            *flag = true;
        }
        assert_eq!(
            nlf_crop_source_for(5, 20, 1, &three_moving, 0.0, 0.0, 100.0, 5.0),
            NlfCropSource::Raw
        );
        let mut two_moving = vec![false; 20];
        for flag in two_moving.iter_mut().take(2) {
            *flag = true;
        }
        assert_eq!(
            nlf_crop_source_for(5, 20, 1, &two_moving, 0.0, 0.0, 100.0, 5.0),
            NlfCropSource::Accumulated
        );
    }

    #[test]
    fn nlf_crop_source_no_overlap_and_bad_input_default_to_accumulator() {
        let moving = vec![true; 16];
        // Bbox entirely outside the grid: no overlap -> accumulator.
        assert_eq!(
            nlf_crop_source_for(16, 4, 4, &moving, 1000.0, 1000.0, 10.0, 10.0),
            NlfCropSource::Accumulated
        );
        // Mismatched bitmap length -> accumulator.
        assert_eq!(
            nlf_crop_source_for(16, 4, 4, &[true, false], 0.0, 0.0, 64.0, 64.0),
            NlfCropSource::Accumulated
        );
        // Non-finite bbox -> accumulator.
        assert_eq!(
            nlf_crop_source_for(16, 4, 4, &moving, f64::NAN, 0.0, 64.0, 64.0),
            NlfCropSource::Accumulated
        );
    }

    #[test]
    fn latest_raw_frame_returns_exact_last_ingested_bytes() {
        // With smoothing the accumulator diverges from the current frame, but
        // latest_raw_frame must echo the last ingested RAW bytes verbatim.
        let settings = CameraSettings {
            smoothing_frames: 3,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut processor = NativePreprocessor::default();
        assert!(processor.latest_raw_frame().is_none());
        processor
            .ingest_camera_frame(solid(10, 4, 4), &settings)
            .unwrap();
        let last = gray(&(0..16).map(|i| (i * 15) as u8).collect::<Vec<_>>(), 4, 4);
        processor
            .ingest_camera_frame(last.clone(), &settings)
            .unwrap();
        let raw = processor.latest_raw_frame().expect("raw frame present");
        assert_eq!(raw.width, 4);
        assert_eq!(raw.height, 4);
        assert_eq!(raw.data, last.data);
        // The accumulated view diverges from the raw bytes under smoothing.
        let processed = processor.process_latest(&settings).unwrap();
        assert_ne!(processed.data, last.data);
        processor.reset();
        assert!(processor.latest_raw_frame().is_none());
    }

    #[test]
    fn tile_motion_reports_grid_dims_and_gated_flags() {
        let settings = CameraSettings {
            smoothing_frames: 3,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let width = 32u32;
        let height = 48u32;
        let mut processor = NativePreprocessor::default();
        assert!(processor.tile_motion().is_none());

        processor
            .ingest_camera_frame(solid(120, width, height), &settings)
            .unwrap();
        let first = processor.tile_motion().expect("motion after first ingest");
        // 32x48 over 16px tiles -> 2 x 3 grid; the first frame flags nothing moving.
        assert_eq!((first.tile_px, first.tiles_x, first.tiles_y), (16, 2, 3));
        assert_eq!(first.moving.len(), 6);
        assert!(first.moving.iter().all(|&flag| !flag));

        // Second frame: strong 4px stripes in the top-left tile trip its SAD gate
        // while the rest of the frame stays uniform (static).
        let mut data = vec![0u8; (width * height * 4) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let in_moving_tile = x < 16 && y < 16;
                let value = if in_moving_tile {
                    if (x / 4) % 2 == 0 {
                        220
                    } else {
                        30
                    }
                } else {
                    120
                };
                let index = (y * width as usize + x) * 4;
                data[index] = value;
                data[index + 1] = value;
                data[index + 2] = value;
                data[index + 3] = 255;
            }
        }
        processor
            .ingest_camera_frame(
                ImageData {
                    data,
                    width,
                    height,
                },
                &settings,
            )
            .unwrap();
        let second = processor.tile_motion().expect("motion after second ingest");
        assert!(second.moving[0], "top-left tile must read as moving");
        assert!(
            second.moving[1..].iter().all(|&flag| !flag),
            "only the striped tile should be moving"
        );
    }
}
