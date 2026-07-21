//! Cross-validation utilities for model evaluation.
//!
//! This is a mechanical Rust port of `src/services/ml/crossValidation.ts`.
//! Class labels remain integer values and timestamps remain `f64` values.

use std::collections::{BTreeMap, HashSet};

/// One train/test split produced by cross-validation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CvFold {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

/// A contiguous temporal run of samples with one label.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalBlock {
    pub indices: Vec<usize>,
    pub label: i32,
    pub start_time: f64,
    pub end_time: f64,
}

/// The default number of cross-validation folds used by the source module.
pub const DEFAULT_N_FOLDS: usize = 5;

/// The default temporal gap threshold used by the source module.
pub const DEFAULT_GAP_THRESHOLD_MS: f64 = 15_000.0;

/// The default random seed used by the source module.
pub const DEFAULT_RANDOM_SEED: f64 = 42.0;

const TRAINING_CATEGORY: &str = "training";

/// Invalid input observed by the cross-validation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossValidationError {
    InvalidFoldCount,
    InvalidLabel,
    InvalidIndex,
    LengthMismatch,
    NonFiniteInput,
}

/// Levels understood by the cross-validation logging boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossValidationLogLevel {
    Info,
    Warn,
}

/// Logging boundary for cross-validation diagnostics.
///
/// The default implementation below suppresses INFO and emits WARN, matching
/// the TypeScript logger's production-safe configuration. Callers that expose
/// a configured native logger can use the `*_with_logger` helpers.
pub trait CrossValidationLogger {
    fn is_enabled(&self, category: &str, level: CrossValidationLogLevel) -> bool;

    fn log(&self, category: &str, level: CrossValidationLogLevel, message: &str);
}

#[derive(Debug, Default, Clone, Copy)]
struct DefaultCrossValidationLogger;

impl CrossValidationLogger for DefaultCrossValidationLogger {
    fn is_enabled(&self, _category: &str, level: CrossValidationLogLevel) -> bool {
        matches!(level, CrossValidationLogLevel::Warn)
    }

    fn log(&self, _category: &str, level: CrossValidationLogLevel, message: &str) {
        if !self.is_enabled(TRAINING_CATEGORY, level) {
            return;
        }
        match level {
            CrossValidationLogLevel::Info => eprintln!("{message}"),
            CrossValidationLogLevel::Warn => eprintln!("{message}"),
        }
    }
}

fn emit_log<L: CrossValidationLogger + ?Sized>(
    logger: &L,
    level: CrossValidationLogLevel,
    message: &str,
) {
    if logger.is_enabled(TRAINING_CATEGORY, level) {
        logger.log(TRAINING_CATEGORY, level, message);
    }
}

fn valid_seed(seed: f64) -> bool {
    seed.is_finite()
}

fn valid_labels(labels: &[i32]) -> bool {
    // The source imposes no upper bound on label values: `numClasses = max(labels) + 1`
    // sizes the class buckets, so sparse labels (max >= sample_count) yield empty
    // buckets, drive `minClassSize` to 0, and fall through to the single all-data fold
    // fallback. Only reject negative labels, which have no valid bucket index.
    labels.iter().all(|label| *label >= 0)
}

/// A small implementation of the default `seedrandom` ARC4 generator.
///
/// `seedrandom(seed.toString())` is used by the TypeScript source. Keeping the
/// same key mixing, ARC4 warm-up, and 52-bit output construction preserves the
/// source's deterministic fold assignments without adding a random-number
/// dependency to the ML crate.
struct SeedRandom {
    state: [u8; 256],
    index: usize,
    j: usize,
}

impl SeedRandom {
    fn from_seed(seed: f64) -> Self {
        let seed_string = super::js_number::to_string(seed);
        let mut key = Vec::with_capacity(seed_string.len());
        let mut smear = 0_i32;

        for (position, code_point) in seed_string.encode_utf16().enumerate() {
            let slot = position & 255;
            let previous = i32::from(*key.get(slot).unwrap_or(&0));
            smear ^= previous * 19;
            key.push_or_set(slot, ((smear + i32::from(code_point)) & 255) as u8);
        }

        if key.is_empty() {
            key.push(0);
        }

        let mut state = [0_u8; 256];
        for (index, value) in state.iter_mut().enumerate() {
            *value = index as u8;
        }

        let mut j = 0_usize;
        for index in 0..256 {
            let value = state[index];
            j = (j + usize::from(value) + usize::from(key[index % key.len()])) & 255;
            state[index] = state[j];
            state[j] = value;
        }

        let mut generator = Self {
            state,
            index: 0,
            j: 0,
        };
        generator.generate(256);
        generator
    }

    fn generate(&mut self, count: usize) -> f64 {
        let mut result = 0_u64;
        for _ in 0..count {
            self.index = (self.index + 1) & 255;
            let value = self.state[self.index];
            self.j = (self.j + usize::from(value)) & 255;
            let other = self.state[self.j];
            self.state[self.index] = other;
            self.state[self.j] = value;
            let output = self.state[(usize::from(other) + usize::from(value)) & 255];
            result = result.wrapping_mul(256).wrapping_add(u64::from(output));
        }
        result as f64
    }

    fn next_f64(&mut self) -> f64 {
        const WIDTH: f64 = 256.0;
        const CHUNKS: usize = 6;
        const START_DENOMINATOR: f64 = 281_474_976_710_656.0;
        const SIGNIFICANCE: f64 = 4_503_599_627_370_496.0;
        const OVERFLOW: f64 = 9_007_199_254_740_992.0;

        let mut numerator = self.generate(CHUNKS);
        let mut denominator = START_DENOMINATOR;
        let mut extra = 0.0;

        while numerator < SIGNIFICANCE {
            numerator = (numerator + extra) * WIDTH;
            denominator *= WIDTH;
            extra = self.generate(1);
        }

        while numerator >= OVERFLOW {
            numerator /= 2.0;
            denominator /= 2.0;
            extra = (extra / 2.0).floor();
        }

        (numerator + extra) / denominator
    }
}

trait PushOrSet<T> {
    fn push_or_set(&mut self, index: usize, value: T);
}

impl<T> PushOrSet<T> for Vec<T> {
    fn push_or_set(&mut self, index: usize, value: T) {
        if index == self.len() {
            self.push(value);
        } else {
            self[index] = value;
        }
    }
}

fn shuffle_array<T: Clone>(array: &[T], seed: f64) -> Vec<T> {
    let mut shuffled = array.to_vec();
    let mut rng = SeedRandom::from_seed(seed);

    for index in (1..shuffled.len()).rev() {
        let swap_index = (rng.next_f64() * (index + 1) as f64).floor() as usize;
        shuffled.swap(index, swap_index);
    }

    shuffled
}

fn all_indices(length: usize) -> Vec<usize> {
    (0..length).collect()
}

fn class_indices(labels: &[i32]) -> Option<Vec<Vec<usize>>> {
    if labels.is_empty() || labels.iter().any(|label| *label < 0) {
        return None;
    }

    let num_classes = labels.iter().copied().max()? as usize + 1;
    let mut indices = vec![Vec::new(); num_classes];
    for (index, &label) in labels.iter().enumerate() {
        indices[label as usize].push(index);
    }
    Some(indices)
}

/// Creates stratified K-fold cross-validation splits.
///
/// Each class is shuffled independently using `random_seed + class_index`,
/// then distributed into folds in the same order as the TypeScript source.
/// When fewer than two samples are available in a class, a single all-data
/// train/test fold is returned.
pub fn create_stratified_k_fold(labels: &[i32], n_folds: usize, random_seed: f64) -> Vec<CvFold> {
    if !valid_seed(random_seed) || !valid_labels(labels) {
        return Vec::new();
    }
    // `Math.max(...new Int32Array())` is -Infinity in the source. That yields
    // zero class buckets and `minClassSize = Math.min(...[]) = Infinity`, so
    // `actualFolds = min(nFolds, Infinity) = nFolds`. When nFolds < 2 the source
    // takes the `actualFolds < 2` fallback and returns a single all-data fold
    // (empty indices here); otherwise the outer loop emits nFolds empty folds.
    if labels.is_empty() {
        if n_folds < 2 {
            return vec![CvFold::default()];
        }
        return vec![CvFold::default(); n_folds];
    }

    let Some(class_indices) = class_indices(labels) else {
        return Vec::new();
    };

    let shuffled_class_indices: Vec<Vec<usize>> = class_indices
        .iter()
        .enumerate()
        .map(|(class_index, indices)| shuffle_array(indices, random_seed + class_index as f64))
        .collect();

    let Some(min_class_size) = shuffled_class_indices.iter().map(Vec::len).min() else {
        return Vec::new();
    };
    let actual_folds = n_folds.min(min_class_size);

    if actual_folds < 2 {
        emit_log(
            &DefaultCrossValidationLogger,
            CrossValidationLogLevel::Warn,
            &format!(
                "Insufficient data for cross-validation: smallest class has only {min_class_size} sample(s). Falling back to single-fold training (no cross-validation)."
            ),
        );
        let indices = all_indices(labels.len());
        return vec![CvFold {
            train_indices: indices.clone(),
            test_indices: indices,
        }];
    }

    if actual_folds < n_folds {
        emit_log(
            &DefaultCrossValidationLogger,
            CrossValidationLogLevel::Warn,
            &format!(
                "Fold count limited from {n_folds} to {actual_folds} (smallest class size: {min_class_size})"
            ),
        );
    }

    let mut folds = Vec::with_capacity(actual_folds);
    for fold_index in 0..actual_folds {
        let mut test_indices = Vec::new();
        let mut train_indices = Vec::new();

        for shuffled_indices in &shuffled_class_indices {
            let test_size = shuffled_indices.len() / actual_folds;
            let test_start = fold_index * test_size;
            let test_end = if fold_index == actual_folds - 1 {
                shuffled_indices.len()
            } else {
                (fold_index + 1) * test_size
            };

            test_indices.extend_from_slice(&shuffled_indices[test_start..test_end]);
            train_indices.extend_from_slice(&shuffled_indices[..test_start]);
            train_indices.extend_from_slice(&shuffled_indices[test_end..]);
        }

        folds.push(CvFold {
            train_indices,
            test_indices,
        });
    }

    folds
}

/// Calculates class counts for the requested sample indices.
pub fn get_class_distribution(
    labels: &[i32],
    indices: &[usize],
) -> Result<BTreeMap<i32, usize>, CrossValidationError> {
    if !valid_labels(labels) {
        return Err(CrossValidationError::InvalidLabel);
    }
    if indices.iter().any(|index| *index >= labels.len()) {
        return Err(CrossValidationError::InvalidIndex);
    }

    let mut distribution = BTreeMap::new();
    if let Some(max_label) = labels.iter().copied().max() {
        for label in 0..=max_label {
            distribution.insert(label, 0);
        }
    }

    for &index in indices {
        let label = labels[index];
        *distribution.entry(label).or_insert(0) += 1;
    }

    Ok(distribution)
}

/// Validates that test folds cover every sample exactly once.
pub fn validate_folds(folds: &[CvFold], n_samples: usize) -> bool {
    let mut all_test_indices = HashSet::new();

    for fold in folds {
        for &index in &fold.test_indices {
            if !all_test_indices.insert(index) {
                return false;
            }
        }
    }

    all_test_indices.len() == n_samples
}

/// Groups frames into temporal blocks based on timestamps and labels.
pub fn detect_temporal_blocks(
    timestamps: &[f64],
    labels: &[i32],
    gap_threshold_ms: f64,
) -> Vec<TemporalBlock> {
    if timestamps.is_empty()
        || timestamps.len() != labels.len()
        || !timestamps.iter().all(|timestamp| timestamp.is_finite())
        || !gap_threshold_ms.is_finite()
        || gap_threshold_ms < 0.0
        || !valid_labels(labels)
    {
        return Vec::new();
    }

    let mut sorted_indices = all_indices(timestamps.len());
    sorted_indices.sort_by(|left, right| timestamps[*left].total_cmp(&timestamps[*right]));

    let first_index = sorted_indices[0];
    let mut blocks = Vec::new();
    let mut current_block = TemporalBlock {
        indices: vec![first_index],
        label: labels[first_index],
        start_time: timestamps[first_index],
        end_time: timestamps[first_index],
    };

    for window in sorted_indices.windows(2) {
        let previous_index = window[0];
        let index = window[1];
        let gap = timestamps[index] - timestamps[previous_index];
        let same_label = labels[index] == current_block.label;

        if same_label && gap <= gap_threshold_ms {
            current_block.indices.push(index);
            current_block.end_time = timestamps[index];
        } else {
            blocks.push(current_block);
            current_block = TemporalBlock {
                indices: vec![index],
                label: labels[index],
                start_time: timestamps[index],
                end_time: timestamps[index],
            };
        }
    }

    blocks.push(current_block);
    blocks
}

/// Logs diagnostic information about temporal correlation in the dataset.
pub fn log_temporal_correlation_diagnostics(
    timestamps: &[f64],
    labels: &[i32],
    gap_threshold_ms: f64,
) {
    if timestamps.is_empty() {
        return;
    }

    let blocks = detect_temporal_blocks(timestamps, labels, gap_threshold_ms);
    let frames_in_bursts: usize = blocks
        .iter()
        .filter(|block| block.indices.len() > 1)
        .map(|block| block.indices.len())
        .sum();
    let _single_frame_blocks = blocks
        .iter()
        .filter(|block| block.indices.len() == 1)
        .count();
    let burst_percentage = frames_in_bursts as f64 / timestamps.len() as f64 * 100.0;

    emit_log(
        &DefaultCrossValidationLogger,
        CrossValidationLogLevel::Info,
        &format!(
            "[CV_DIAG] Temporal analysis: {} blocks, {}/{} frames in bursts ({:.1}%)",
            blocks.len(),
            frames_in_bursts,
            timestamps.len(),
            burst_percentage
        ),
    );

    if burst_percentage > 50.0 {
        emit_log(
            &DefaultCrossValidationLogger,
            CrossValidationLogLevel::Warn,
            &format!(
                "[CV_DIAG] HIGH TEMPORAL CORRELATION: {:.0}% of frames in burst sequences (<{}s apart). Consider capturing frames more spread out over time.",
                burst_percentage,
                gap_threshold_ms / 1000.0
            ),
        );
    }
}

/// Creates stratified K-fold splits while keeping temporal blocks together.
pub fn create_temporal_block_k_fold(
    timestamps: &[f64],
    labels: &[i32],
    n_folds: usize,
    gap_threshold_ms: f64,
    random_seed: f64,
) -> Vec<CvFold> {
    if timestamps.is_empty()
        || timestamps.len() != labels.len()
        || !valid_seed(random_seed)
        || !timestamps.iter().all(|timestamp| timestamp.is_finite())
        || !gap_threshold_ms.is_finite()
        || gap_threshold_ms < 0.0
        || !valid_labels(labels)
    {
        return Vec::new();
    }

    log_temporal_correlation_diagnostics(timestamps, labels, gap_threshold_ms);
    let blocks = detect_temporal_blocks(timestamps, labels, gap_threshold_ms);
    if blocks.is_empty() {
        return Vec::new();
    }

    let Some(class_indices) = class_indices(labels) else {
        return Vec::new();
    };
    let num_classes = class_indices.len();
    let mut blocks_by_class: Vec<Vec<TemporalBlock>> = vec![Vec::new(); num_classes];
    for block in blocks {
        let class_index = block.label as usize;
        if class_index >= num_classes {
            return Vec::new();
        }
        blocks_by_class[class_index].push(block);
    }

    let shuffled_blocks_by_class: Vec<Vec<TemporalBlock>> = blocks_by_class
        .iter()
        .enumerate()
        .map(|(class_index, class_blocks)| {
            shuffle_array(class_blocks, random_seed + class_index as f64)
        })
        .collect();

    let non_empty_block_counts: Vec<usize> = shuffled_blocks_by_class
        .iter()
        .filter(|class_blocks| !class_blocks.is_empty())
        .map(Vec::len)
        .collect();
    let Some(min_block_count) = non_empty_block_counts.iter().copied().min() else {
        return Vec::new();
    };
    let actual_folds = n_folds.min(min_block_count);

    if min_block_count < 2 || actual_folds < 1 {
        emit_log(
            &DefaultCrossValidationLogger,
            CrossValidationLogLevel::Warn,
            &format!(
                "[CV] Only {min_block_count} block(s) in smallest class - cannot do proper CV, using all data for train/test"
            ),
        );
        let indices = all_indices(labels.len());
        return vec![CvFold {
            train_indices: indices.clone(),
            test_indices: indices,
        }];
    }

    if actual_folds < n_folds {
        emit_log(
            &DefaultCrossValidationLogger,
            CrossValidationLogLevel::Info,
            &format!(
                "[CV] Using {actual_folds}-fold CV (limited by {min_block_count} blocks in smallest class)"
            ),
        );
    }

    let mut fold_block_assignments: Vec<Vec<TemporalBlock>> = vec![Vec::new(); actual_folds];
    for class_blocks in &shuffled_blocks_by_class {
        for (block_index, block) in class_blocks.iter().enumerate() {
            let fold_index = block_index % actual_folds;
            fold_block_assignments[fold_index].push(block.clone());
        }
    }

    let mut folds = Vec::with_capacity(actual_folds);
    for fold_index in 0..actual_folds {
        let mut test_indices = Vec::new();
        let mut train_indices = Vec::new();

        for (other_fold_index, blocks_in_fold) in fold_block_assignments.iter().enumerate() {
            for block in blocks_in_fold {
                if other_fold_index == fold_index {
                    test_indices.extend_from_slice(&block.indices);
                } else {
                    train_indices.extend_from_slice(&block.indices);
                }
            }
        }

        folds.push(CvFold {
            train_indices,
            test_indices,
        });
    }

    folds
}

/// Creates forward-chaining, capture-time-ordered cross-validation splits.
///
/// Sample indices are sorted by `timestamp` and partitioned into contiguous,
/// chronologically-ordered segments. Fold `k` trains on the earliest cumulative
/// block and tests on the next segment, so every test index is captured strictly
/// later than all of that fold's train indices. Train frames within
/// `gap_threshold_ms` of the test segment's earliest timestamp are purged (an
/// embargo) so adjacent near-duplicate captures cannot leak across the split.
///
/// When there are too few frames for the requested fold count, this falls back to
/// a single earliest-train / latest-test (~67/33) holdout. When even a purged
/// holdout cannot be formed (e.g. every frame sits inside one sub-gap burst), it
/// returns the single all-data identity fold that callers treat as the
/// "insufficient data" sentinel.
pub fn create_time_ordered_folds(
    timestamps: &[f64],
    labels: &[i32],
    n_folds: usize,
    gap_threshold_ms: f64,
) -> Vec<CvFold> {
    if timestamps.is_empty()
        || timestamps.len() != labels.len()
        || !timestamps.iter().all(|timestamp| timestamp.is_finite())
        || !gap_threshold_ms.is_finite()
        || gap_threshold_ms < 0.0
        || !valid_labels(labels)
    {
        return Vec::new();
    }

    let mut sorted_indices = all_indices(timestamps.len());
    sorted_indices.sort_by(|left, right| timestamps[*left].total_cmp(&timestamps[*right]));

    let sample_count = sorted_indices.len();
    // `n_folds` expanding-window folds need `n_folds + 1` non-empty segments: one
    // leading train block plus one test block per fold.
    let segment_count = n_folds.saturating_add(1);
    let folds = if n_folds >= 2 && sample_count >= segment_count {
        build_expanding_window_folds(&sorted_indices, timestamps, n_folds, gap_threshold_ms)
    } else {
        build_holdout_fold(&sorted_indices, timestamps, gap_threshold_ms)
            .into_iter()
            .collect()
    };

    let has_usable_fold = folds
        .iter()
        .any(|fold| !fold.train_indices.is_empty() && !fold.test_indices.is_empty());
    if !has_usable_fold {
        emit_log(
            &DefaultCrossValidationLogger,
            CrossValidationLogLevel::Warn,
            "[CV] Insufficient frames for a purged time-ordered holdout - metrics will be optimistic; collect more frames spread over time.",
        );
        let indices = all_indices(labels.len());
        return vec![CvFold {
            train_indices: indices.clone(),
            test_indices: indices,
        }];
    }

    folds
}

fn build_expanding_window_folds(
    sorted_indices: &[usize],
    timestamps: &[f64],
    n_folds: usize,
    gap_threshold_ms: f64,
) -> Vec<CvFold> {
    let sample_count = sorted_indices.len();
    let segment_count = n_folds + 1;
    let base = sample_count / segment_count;
    let remainder = sample_count % segment_count;
    // Trailing segments absorb the remainder so every segment stays non-empty.
    let segment_len = |segment: usize| base + usize::from(segment >= segment_count - remainder);

    let mut boundaries = Vec::with_capacity(segment_count + 1);
    let mut cursor = 0;
    boundaries.push(0);
    for segment in 0..segment_count {
        cursor += segment_len(segment);
        boundaries.push(cursor);
    }

    let mut folds = Vec::with_capacity(n_folds);
    for fold in 0..n_folds {
        let test_start = boundaries[fold + 1];
        let test_end = boundaries[fold + 2];
        let boundary_time = timestamps[sorted_indices[test_start]];
        folds.push(CvFold {
            train_indices: collect_purged_train(
                sorted_indices,
                timestamps,
                boundary_time,
                gap_threshold_ms,
            ),
            test_indices: sorted_indices[test_start..test_end].to_vec(),
        });
    }
    folds
}

fn build_holdout_fold(
    sorted_indices: &[usize],
    timestamps: &[f64],
    gap_threshold_ms: f64,
) -> Option<CvFold> {
    let sample_count = sorted_indices.len();
    if sample_count < 2 {
        return None;
    }
    // ceil(2 * n / 3), kept in 1..=(n - 1) so both sides are non-empty.
    let split = (sample_count * 2).div_ceil(3).clamp(1, sample_count - 1);
    let boundary_time = timestamps[sorted_indices[split]];
    let train_indices =
        collect_purged_train(sorted_indices, timestamps, boundary_time, gap_threshold_ms);
    let test_indices = sorted_indices[split..].to_vec();
    if train_indices.is_empty() || test_indices.is_empty() {
        return None;
    }
    Some(CvFold {
        train_indices,
        test_indices,
    })
}

/// Collects the training indices that precede `boundary_time` by at least
/// `gap_threshold_ms`, preserving chronological order. The strict `<` guard keeps
/// tie-straddling frames out of the train side even when the gap is zero.
fn collect_purged_train(
    sorted_indices: &[usize],
    timestamps: &[f64],
    boundary_time: f64,
    gap_threshold_ms: f64,
) -> Vec<usize> {
    sorted_indices
        .iter()
        .copied()
        .filter(|&index| {
            let time = timestamps[index];
            time < boundary_time && boundary_time - time >= gap_threshold_ms
        })
        .collect()
}

impl Default for TemporalBlock {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            label: 0,
            start_time: 0.0,
            end_time: 0.0,
        }
    }
}
