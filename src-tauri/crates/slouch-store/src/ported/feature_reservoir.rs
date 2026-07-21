//! Durable feature reservoir for retaining representative inference samples.
//!
//! Each named reservoir is a SQLite file with atomic sample/state updates. The
//! implementation preserves the source Algorithm R semantics while validating
//! the stricter native feature, keypoint, and geometry boundary before any
//! mutation.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use slouch_domain::{BoundingBox, FeatureMap, Keypoint};

const MAX_SAMPLES: usize = 1_000;
const MAX_NATIVE_SAMPLES: usize = 1_000_000;
const DEFAULT_DB_NAME: &str = "slouch-tracker-reservoir";
const INITIAL_RNG_STATE: u64 = 0x9e37_79b9_7f4a_7c15;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS feature_reservoir_state (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  seen_count INTEGER NOT NULL,
  sample_count INTEGER NOT NULL,
  rng_state INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS feature_reservoir_samples (
  slot INTEGER PRIMARY KEY,
  payload BLOB NOT NULL
);
";

/// A retained inference sample. `features` holds the stored (GPU-produced,
/// persisted) feature vectors of one frame keyed by their registry id, so new
/// stored features flow through the reservoir with no shape change. The twin
/// `slouch_ml::ported::training_worker::ReservoirSample` is byte-identical under
/// `rmp_serde::to_vec_named` (fields `features`/`keypoints`/`bbox`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReservoirSample {
    pub features: FeatureMap,
    pub keypoints: Vec<Keypoint>,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservoirMeta {
    #[serde(rename = "totalSeen")]
    pub total_seen: usize,
    pub count: usize,
    #[serde(rename = "maxSamples")]
    pub max_samples: usize,
}

#[derive(Debug)]
pub enum ReservoirError {
    Storage(rusqlite::Error),
    Encoding(String),
    Validation(String),
    InvalidData(String),
}

impl std::fmt::Display for ReservoirError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(error) => write!(formatter, "feature reservoir SQLite error: {error}"),
            Self::Encoding(error) => write!(formatter, "feature reservoir encoding error: {error}"),
            Self::Validation(error) => {
                write!(formatter, "feature reservoir validation failed: {error}")
            }
            Self::InvalidData(error) => {
                write!(formatter, "invalid feature reservoir data: {error}")
            }
        }
    }
}

impl std::error::Error for ReservoirError {}

impl From<rusqlite::Error> for ReservoirError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Storage(error)
    }
}

/// A fixed-size, durable reservoir using the source implementation's
/// Algorithm R. Separate values of `db_name` select separate SQLite owners.
#[derive(Debug, Clone)]
pub struct FeatureReservoir {
    database_path: PathBuf,
    max_samples: usize,
    initial_rng_state: u64,
}

impl FeatureReservoir {
    pub fn new(max_samples: usize, db_name: impl AsRef<str>) -> Self {
        let name = db_name.as_ref();
        let path = named_database_path(name);
        Self {
            database_path: path,
            max_samples,
            initial_rng_state: seed_for_name(name),
        }
    }

    /// Opens a reservoir at an explicit application-owned path.
    pub fn at_path(max_samples: usize, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let initial_rng_state = seed_for_name(&path.to_string_lossy());
        Self {
            database_path: path,
            max_samples,
            initial_rng_state,
        }
    }

    pub fn with_default_max_samples() -> Self {
        Self::new(MAX_SAMPLES, DEFAULT_DB_NAME)
    }

    pub fn database_path(&self) -> &Path {
        &self.database_path
    }

    pub fn add(&self, sample: ReservoirSample) -> Result<(), ReservoirError> {
        self.validate_capacity()?;
        validate_sample(&sample)?;
        let payload = rmp_serde::to_vec_named(&sample)
            .map_err(|error| ReservoirError::Encoding(error.to_string()))?;
        if payload.is_empty() || payload.len() > 8 * 1024 * 1024 {
            return Err(ReservoirError::Validation(
                "encoded sample exceeds the 8 MiB limit".to_owned(),
            ));
        }

        let mut connection = self.open()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let state = transaction
            .query_row(
                "SELECT seen_count, sample_count, rng_state FROM feature_reservoir_state WHERE singleton = 1",
                [],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
            )
            .optional()?;
        let (seen, count, rng) =
            state.unwrap_or((0, 0, (self.initial_rng_state & i64::MAX as u64) as i64));
        if seen < 0 || count < 0 {
            return Err(ReservoirError::InvalidData(
                "negative persisted reservoir counters".to_owned(),
            ));
        }
        let next_seen = seen
            .checked_add(1)
            .ok_or_else(|| ReservoirError::InvalidData("seen count overflow".to_owned()))?;
        let next_rng = next_rng(rng as u64) & i64::MAX as u64;
        let capacity = i64::try_from(self.max_samples)
            .map_err(|_| ReservoirError::Validation("capacity is too large".to_owned()))?;
        let slot = if count < capacity {
            Some(count)
        } else {
            let candidate = (next_rng % next_seen as u64) as i64;
            (candidate < capacity).then_some(candidate)
        };
        if let Some(slot) = slot {
            transaction.execute(
                "INSERT INTO feature_reservoir_samples(slot, payload) VALUES (?, ?)
                 ON CONFLICT(slot) DO UPDATE SET payload = excluded.payload",
                params![slot, payload],
            )?;
        }
        let next_count = count.saturating_add(1).min(capacity);
        transaction.execute(
            "INSERT INTO feature_reservoir_state(singleton, seen_count, sample_count, rng_state)
             VALUES (1, ?, ?, ?)
             ON CONFLICT(singleton) DO UPDATE SET
               seen_count = excluded.seen_count,
               sample_count = excluded.sample_count,
               rng_state = excluded.rng_state",
            params![next_seen, next_count, next_rng as i64],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn get_all_samples(&self) -> Result<Vec<ReservoirSample>, ReservoirError> {
        self.validate_capacity()?;
        let connection = self.open()?;
        let expected_count = persisted_count(&connection)?;
        let mut statement = connection
            .prepare("SELECT slot, payload FROM feature_reservoir_samples ORDER BY slot")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        let mut samples = Vec::with_capacity(expected_count.min(self.max_samples));
        for (expected_slot, row) in rows.enumerate() {
            let (slot, payload) = row?;
            if slot != expected_slot as i64 || expected_slot >= expected_count {
                return Err(ReservoirError::InvalidData(
                    "sample slots are not contiguous".to_owned(),
                ));
            }
            let sample: ReservoirSample = rmp_serde::from_slice(&payload)
                .map_err(|error| ReservoirError::Encoding(error.to_string()))?;
            validate_sample(&sample)
                .map_err(|error| ReservoirError::InvalidData(error.to_string()))?;
            samples.push(sample);
        }
        if samples.len() != expected_count {
            return Err(ReservoirError::InvalidData(format!(
                "state records {expected_count} samples but {} were stored",
                samples.len()
            )));
        }
        Ok(samples)
    }

    pub fn get_count(&self) -> Result<usize, ReservoirError> {
        self.validate_capacity()?;
        let connection = self.open()?;
        persisted_count(&connection)
    }

    pub fn get_meta(&self) -> Result<ReservoirMeta, ReservoirError> {
        self.validate_capacity()?;
        let connection = self.open()?;
        let state = connection
            .query_row(
                "SELECT seen_count, sample_count FROM feature_reservoir_state WHERE singleton = 1",
                [],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()?;
        let (seen, count) = state.unwrap_or((0, 0));
        Ok(ReservoirMeta {
            total_seen: checked_counter(seen, "seen")?,
            count: checked_counter(count, "sample")?,
            max_samples: self.max_samples,
        })
    }

    pub fn clear(&self) -> Result<(), ReservoirError> {
        self.validate_capacity()?;
        let mut connection = self.open()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute("DELETE FROM feature_reservoir_samples", [])?;
        transaction.execute("DELETE FROM feature_reservoir_state", [])?;
        transaction.commit()?;
        Ok(())
    }

    fn validate_capacity(&self) -> Result<(), ReservoirError> {
        if self.max_samples == 0 || self.max_samples > MAX_NATIVE_SAMPLES {
            return Err(ReservoirError::Validation(format!(
                "maxSamples must be between 1 and {MAX_NATIVE_SAMPLES}"
            )));
        }
        Ok(())
    }

    fn open(&self) -> Result<Connection, ReservoirError> {
        if let Some(parent) = self.database_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|error| {
                    ReservoirError::Validation(format!(
                        "could not create reservoir directory: {error}"
                    ))
                })?;
            }
        }
        let connection = Connection::open(&self.database_path)?;
        connection.execute_batch("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;")?;
        connection.execute_batch(SCHEMA)?;
        Ok(connection)
    }
}

impl Default for FeatureReservoir {
    fn default() -> Self {
        Self::with_default_max_samples()
    }
}

static DEFAULT_RESERVOIR: OnceLock<FeatureReservoir> = OnceLock::new();

pub fn feature_reservoir() -> &'static FeatureReservoir {
    DEFAULT_RESERVOIR.get_or_init(FeatureReservoir::default)
}

fn validate_sample(sample: &ReservoirSample) -> Result<(), ReservoirError> {
    if sample.features.is_empty() {
        return Err(ReservoirError::Validation(
            "reservoir sample must contain at least one stored feature".to_owned(),
        ));
    }
    for (id, values) in &sample.features {
        let metadata = id.metadata();
        if metadata.computed {
            return Err(ReservoirError::Validation(format!(
                "reservoir feature {} is computed and cannot be stored",
                id.as_str()
            )));
        }
        if values.len() != metadata.dimensions || values.iter().any(|value| !value.is_finite()) {
            return Err(ReservoirError::Validation(format!(
                "{} feature must contain {} finite values",
                id.as_str(),
                metadata.dimensions
            )));
        }
    }
    if sample.keypoints.len() != 17
        || sample
            .keypoints
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.score.is_finite())
    {
        return Err(ReservoirError::Validation(
            "keypoints must contain 17 finite points".to_owned(),
        ));
    }
    slouch_domain::validate_bbox(&sample.bbox)
        .map_err(|error| ReservoirError::Validation(format!("bounding box is invalid: {error}")))?;
    Ok(())
}

fn persisted_count(connection: &Connection) -> Result<usize, ReservoirError> {
    let count = connection
        .query_row(
            "SELECT sample_count FROM feature_reservoir_state WHERE singleton = 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .unwrap_or(0);
    checked_counter(count, "sample")
}

fn checked_counter(value: i64, name: &str) -> Result<usize, ReservoirError> {
    usize::try_from(value).map_err(|_| {
        ReservoirError::InvalidData(format!("persisted {name} count is negative or too large"))
    })
}

fn next_rng(mut state: u64) -> u64 {
    if state == 0 {
        state = INITIAL_RNG_STATE;
    }
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn seed_for_name(name: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    let seed = hasher.finish() & i64::MAX as u64;
    if seed == 0 {
        INITIAL_RNG_STATE
    } else {
        seed
    }
}

fn named_database_path(name: &str) -> PathBuf {
    let path = Path::new(name);
    if path.is_absolute() || path.components().count() > 1 || path.extension().is_some() {
        path.to_path_buf()
    } else {
        let safe = name
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                    character
                } else {
                    '_'
                }
            })
            .collect::<String>();
        std::env::temp_dir().join(format!("slouch-feature-reservoir-{safe}.sqlite3"))
    }
}
