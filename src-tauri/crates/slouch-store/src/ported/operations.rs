//! Dataset business operations.
//!
//! This is the native counterpart of `src/services/dataset/operations.ts`.
//! Browser-only IndexedDB, ZIP/Blob, and download details are supplied by the
//! [`DatasetStore`] boundary during application integration; the operation
//! ordering, bulk-result semantics, and safe retraining fallback stay here.

use std::fmt;

use slouch_domain::{DatasetStats, FrameLabel, ImportResult, PostureDataset, PostureFrame};

use super::feature_reservoir::{feature_reservoir, FeatureReservoir, ReservoirError};

/// Result returned by bulk frame mutations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BulkOperationResult {
    pub deleted: usize,
    pub success: bool,
    pub error: Option<String>,
}

/// Archive bytes supplied by the native import boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportArchive<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
}

/// Storage/archive boundary required by [`DatasetOperations`].
///
/// The concrete SQLite/archive implementation belongs to the integration
/// layer. Keeping this trait narrow avoids making the business layer depend on
/// Tauri, browser `File`/`Blob` values, or a particular archive crate.
pub trait DatasetStore {
    type Error: fmt::Display + Send + Sync + 'static;

    fn update_frame_label(&self, id: &str, label: FrameLabel) -> Result<(), Self::Error>;
    fn remove_frames_by_label(&self, label: FrameLabel) -> Result<usize, Self::Error>;
    fn clear_dataset(&self) -> Result<(), Self::Error>;
    fn clear_training_settings(&self) -> Result<(), Self::Error>;
    fn clear_posture_model(&self) -> Result<(), Self::Error>;
    fn clear_presence_model(&self) -> Result<(), Self::Error>;

    fn get_stats(&self) -> Result<DatasetStats, Self::Error>;
    fn load_dataset(&self) -> Result<PostureDataset, Self::Error>;
    fn get_frames_by_label(&self, label: FrameLabel) -> Result<Vec<PostureFrame>, Self::Error>;
    fn get_frame_by_id(&self, id: &str) -> Result<Option<PostureFrame>, Self::Error>;
    fn needs_retraining(&self) -> Result<bool, Self::Error>;

    /// Export and persist/hand off the archive. Download UX is integration-owned.
    fn export_dataset(
        &self,
        dataset: &PostureDataset,
        filename: Option<&str>,
    ) -> Result<(), Self::Error>;

    /// Import archive bytes additively, preserving duplicate/partial-recovery
    /// behavior in the archive implementation.
    fn import_dataset(&self, archive: ImportArchive<'_>) -> Result<ImportResult, Self::Error>;
}

/// Logging boundary used by the service layer.
pub trait DatasetLogger {
    fn debug(&self, category: &str, message: &str);
    fn info(&self, category: &str, message: &str);
    fn warn(&self, category: &str, message: &str);
    fn error(&self, category: &str, message: &str);
}

/// Default native logger. The application can replace it with its structured
/// logger without changing dataset behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopDatasetLogger;

impl DatasetLogger for NoopDatasetLogger {
    fn debug(&self, _category: &str, _message: &str) {}
    fn info(&self, _category: &str, _message: &str) {}
    fn warn(&self, _category: &str, _message: &str) {}
    fn error(&self, _category: &str, _message: &str) {}
}

/// Errors raised by operations that throw in the TypeScript service.
#[derive(Debug)]
pub enum DatasetOperationError<E> {
    Storage {
        context: &'static str,
        source: E,
    },
    Reservoir {
        context: &'static str,
        source: ReservoirError,
    },
    InvalidInput(String),
}

impl<E: fmt::Display> fmt::Display for DatasetOperationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage { context, source } => write!(formatter, "{context}: {source}"),
            Self::Reservoir { context, source } => write!(formatter, "{context}: {source}"),
            Self::InvalidInput(message) => formatter.write_str(message),
        }
    }
}

impl<E> std::error::Error for DatasetOperationError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage { source, .. } => Some(source),
            Self::Reservoir { source, .. } => Some(source),
            Self::InvalidInput(_) => None,
        }
    }
}

/// Business logic for dataset CRUD, reset, query, and archive operations.
pub struct DatasetOperations<S, L = NoopDatasetLogger> {
    storage: S,
    reservoir: &'static FeatureReservoir,
    logger: L,
}

impl<S> DatasetOperations<S, NoopDatasetLogger>
where
    S: DatasetStore,
{
    pub fn new(storage: S) -> Self {
        Self::with_dependencies(storage, feature_reservoir(), NoopDatasetLogger)
    }
}

impl<S, L> DatasetOperations<S, L>
where
    S: DatasetStore,
    L: DatasetLogger,
{
    pub fn with_dependencies(storage: S, reservoir: &'static FeatureReservoir, logger: L) -> Self {
        Self {
            storage,
            reservoir,
            logger,
        }
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    pub fn update_frame_label(
        &self,
        id: &str,
        label: FrameLabel,
    ) -> Result<(), DatasetOperationError<S::Error>> {
        self.logger.debug(
            "storage",
            &format!("[DatasetOperations] Updating frame {id} label to: {label:?}"),
        );
        self.storage
            .update_frame_label(id, label)
            .map_err(|error| self.storage_error("Failed to update frame label", error))?;
        self.logger.debug(
            "storage",
            &format!("[DatasetOperations] Successfully updated frame {id}"),
        );
        Ok(())
    }

    pub fn delete_frame(&self, id: &str) -> Result<(), DatasetOperationError<S::Error>> {
        self.logger.debug(
            "storage",
            &format!("[DatasetOperations] Deleting frame: {id}"),
        );
        self.storage
            .update_frame_label(id, FrameLabel::Unused)
            .map_err(|error| self.storage_error("Failed to delete frame", error))?;
        self.logger.debug(
            "storage",
            &format!("[DatasetOperations] Successfully deleted frame {id}"),
        );
        Ok(())
    }

    pub fn delete_bulk<I, T>(
        &self,
        ids: I,
    ) -> Result<BulkOperationResult, DatasetOperationError<S::Error>>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let ids = ids
            .into_iter()
            .map(|id| id.as_ref().to_owned())
            .collect::<Vec<_>>();
        let mut deleted = 0;
        self.logger.info(
            "storage",
            &format!("[DatasetOperations] Bulk deleting {} frames", ids.len()),
        );

        for id in &ids {
            // The unused-relabel succeeds for absent ids (TS oracle semantics), but the
            // tightened native contract counts only frames that actually existed.
            let existed = self
                .storage
                .get_frame_by_id(id)
                .map(|frame| frame.is_some())
                .unwrap_or(false);
            match self.storage.update_frame_label(id, FrameLabel::Unused) {
                Ok(()) if existed => deleted += 1,
                Ok(()) => self.logger.warn(
                    "storage",
                    &format!("[DatasetOperations] Failed to delete frame {id}: frame not found"),
                ),
                Err(error) => self.logger.warn(
                    "storage",
                    &format!("[DatasetOperations] Failed to delete frame {id}: {error}"),
                ),
            }
        }

        self.logger.info(
            "storage",
            &format!(
                "[DatasetOperations] Bulk delete complete: {deleted}/{} frames deleted",
                ids.len()
            ),
        );
        Ok(BulkOperationResult {
            deleted,
            success: deleted > 0,
            error: None,
        })
    }

    pub fn cleanup_unused(&self) -> Result<BulkOperationResult, DatasetOperationError<S::Error>> {
        self.logger.info(
            "storage",
            "[DatasetOperations] Starting cleanup of unused frames",
        );
        match self.storage.remove_frames_by_label(FrameLabel::Unused) {
            Ok(deleted) => {
                self.logger.info(
                    "storage",
                    &format!(
                        "[DatasetOperations] Cleanup complete: removed {deleted} unused frames"
                    ),
                );
                Ok(BulkOperationResult {
                    deleted,
                    success: true,
                    error: None,
                })
            }
            Err(error) => {
                let message = format!("Cleanup failed: {error}");
                self.logger.error("storage", &message);
                Ok(BulkOperationResult {
                    deleted: 0,
                    success: false,
                    error: Some(message),
                })
            }
        }
    }

    pub fn delete_by_label(
        &self,
        label: FrameLabel,
    ) -> Result<BulkOperationResult, DatasetOperationError<S::Error>> {
        self.logger.info(
            "storage",
            &format!("[DatasetOperations] Deleting all frames with label: {label:?}"),
        );

        let frames = match self.storage.get_frames_by_label(label) {
            Ok(frames) => frames,
            Err(error) => {
                let message = format!("Delete by label failed: {error}");
                self.logger.error("storage", &message);
                return Ok(BulkOperationResult {
                    deleted: 0,
                    success: false,
                    error: Some(message),
                });
            }
        };
        if frames.iter().any(|frame| frame.label != label) {
            let message =
                "Delete by label failed: storage returned a frame with a mismatched label"
                    .to_owned();
            self.logger.error("storage", &message);
            return Ok(BulkOperationResult {
                deleted: 0,
                success: false,
                error: Some(message),
            });
        }
        let ids: Vec<String> = frames.into_iter().map(|frame| frame.id).collect();
        self.logger.info(
            "storage",
            &format!(
                "[DatasetOperations] Found {} frames with label '{label:?}'",
                ids.len()
            ),
        );

        if ids.is_empty() {
            return Ok(BulkOperationResult {
                deleted: 0,
                success: true,
                error: None,
            });
        }

        let result = self.delete_bulk(&ids)?;
        self.logger.info(
            "storage",
            &format!(
                "[DatasetOperations] Deleted {}/{} frames with label '{label:?}'",
                result.deleted,
                ids.len()
            ),
        );
        Ok(BulkOperationResult {
            deleted: result.deleted,
            success: result.success,
            error: result.error,
        })
    }

    pub fn reset_dataset(&self) -> Result<(), DatasetOperationError<S::Error>> {
        self.logger.info(
            "storage",
            "[DatasetOperations] Resetting dataset (keeping model + settings)",
        );

        // Both operations are started before returning the first failure, matching
        // Promise.all's eager execution in the source implementation.
        let storage_result = self.storage.clear_dataset();
        let reservoir_result = self.reservoir.clear();
        if let Err(error) = storage_result {
            let wrapped = self.storage_error("Dataset reset failed", error);
            self.logger.error("storage", &wrapped.to_string());
            return Err(wrapped);
        }
        if let Err(error) = reservoir_result {
            let wrapped = self.reservoir_error("Dataset reset failed", error);
            self.logger.error("storage", &wrapped.to_string());
            return Err(wrapped);
        }

        self.logger.info(
            "storage",
            "[DatasetOperations] Dataset reset complete (model + settings preserved)",
        );
        Ok(())
    }

    pub fn reset_all_data(&self) -> Result<(), DatasetOperationError<S::Error>> {
        self.logger
            .info("storage", "[DatasetOperations] Complete app wipe");

        // Keep invoking every clear operation after an earlier failure, as the
        // source Promise.all does, while returning the first error in call order.
        let mut first_error = self
            .storage
            .clear_dataset()
            .err()
            .map(|error| self.storage_error("App wipe failed", error));
        if let Err(error) = self.storage.clear_training_settings() {
            if first_error.is_none() {
                first_error = Some(self.storage_error("App wipe failed", error));
            }
        }
        if let Err(error) = self.storage.clear_posture_model() {
            if first_error.is_none() {
                first_error = Some(self.storage_error("App wipe failed", error));
            }
        }
        if let Err(error) = self.storage.clear_presence_model() {
            if first_error.is_none() {
                first_error = Some(self.storage_error("App wipe failed", error));
            }
        }
        if let Err(error) = self.reservoir.clear() {
            if first_error.is_none() {
                first_error = Some(self.reservoir_error("App wipe failed", error));
            }
        }

        if let Some(error) = first_error {
            self.logger.error("storage", &error.to_string());
            return Err(error);
        }

        self.logger
            .info("storage", "[DatasetOperations] Complete app wipe done");
        Ok(())
    }

    pub fn get_stats(&self) -> Result<DatasetStats, DatasetOperationError<S::Error>> {
        self.logger.debug(
            "storage",
            "[DatasetOperations] Computing dataset statistics",
        );
        let stats = self
            .storage
            .get_stats()
            .map_err(|error| self.storage_error("Failed to get stats", error))?;
        self.logger
            .info("storage", &format!("[DatasetOperations] Stats: {stats:?}"));
        Ok(stats)
    }

    pub fn load_dataset(&self) -> Result<PostureDataset, DatasetOperationError<S::Error>> {
        self.logger
            .info("storage", "[DatasetOperations] Loading complete dataset");
        let dataset = self
            .storage
            .load_dataset()
            .map_err(|error| self.storage_error("Failed to load dataset", error))?;
        self.logger.info(
            "storage",
            &format!("[DatasetOperations] Loaded {} frames", dataset.frames.len()),
        );
        Ok(dataset)
    }

    pub fn get_frames_by_label(
        &self,
        label: FrameLabel,
    ) -> Result<Vec<PostureFrame>, DatasetOperationError<S::Error>> {
        self.logger.info(
            "storage",
            &format!("[DatasetOperations] Loading frames with label: {label:?}"),
        );
        let frames = self
            .storage
            .get_frames_by_label(label)
            .map_err(|error| self.storage_error("Failed to get frames by label", error))?;
        self.logger.info(
            "storage",
            &format!(
                "[DatasetOperations] Found {} frames with label '{label:?}'",
                frames.len()
            ),
        );
        Ok(frames)
    }

    pub fn get_frame_by_id(
        &self,
        id: &str,
    ) -> Result<Option<PostureFrame>, DatasetOperationError<S::Error>> {
        self.logger.debug(
            "storage",
            &format!("[DatasetOperations] Loading frame: {id}"),
        );
        let frame = self
            .storage
            .get_frame_by_id(id)
            .map_err(|error| self.storage_error("Failed to get frame", error))?;
        if frame.is_some() {
            self.logger
                .debug("storage", &format!("[DatasetOperations] Found frame {id}"));
        } else {
            self.logger.warn(
                "storage",
                &format!("[DatasetOperations] Frame {id} not found"),
            );
        }
        Ok(frame)
    }

    pub fn needs_retraining(&self) -> Result<bool, DatasetOperationError<S::Error>> {
        match self.storage.needs_retraining() {
            Ok(needs_retraining) => {
                self.logger.debug(
                    "storage",
                    &format!("[DatasetOperations] Needs retraining: {needs_retraining}"),
                );
                Ok(needs_retraining)
            }
            Err(error) => {
                self.logger.error(
                    "storage",
                    &format!("[DatasetOperations] Failed to check retraining status: {error}"),
                );
                Ok(true)
            }
        }
    }

    pub fn export_dataset(
        &self,
        filename: Option<&str>,
    ) -> Result<(), DatasetOperationError<S::Error>> {
        self.logger
            .info("storage", "[DatasetOperations] Starting dataset export");
        let dataset = self
            .storage
            .load_dataset()
            .map_err(|error| self.storage_error("Export failed", error))?;
        if dataset.frames.is_empty() {
            let error = DatasetOperationError::InvalidInput(
                "Export failed: Cannot export empty dataset".to_owned(),
            );
            self.logger.error("storage", &error.to_string());
            return Err(error);
        }

        self.storage
            .export_dataset(&dataset, filename)
            .map_err(|error| self.storage_error("Export failed", error))?;
        self.logger.info(
            "storage",
            &format!(
                "[DatasetOperations] Export complete: {} frames",
                dataset.frames.len()
            ),
        );
        Ok(())
    }

    pub fn import_dataset(
        &self,
        archive: ImportArchive<'_>,
    ) -> Result<ImportResult, DatasetOperationError<S::Error>> {
        self.logger.info(
            "storage",
            &format!(
                "[DatasetOperations] Starting dataset import from: {}",
                archive.name
            ),
        );
        let result = self
            .storage
            .import_dataset(archive)
            .map_err(|error| self.storage_error("Import failed", error))?;
        self.logger.info(
            "storage",
            &format!(
                "[DatasetOperations] Import complete: {} imported, {} skipped",
                result.imported, result.skipped
            ),
        );
        Ok(result)
    }

    fn storage_error(
        &self,
        context: &'static str,
        source: S::Error,
    ) -> DatasetOperationError<S::Error> {
        DatasetOperationError::Storage { context, source }
    }

    fn reservoir_error(
        &self,
        context: &'static str,
        source: ReservoirError,
    ) -> DatasetOperationError<S::Error> {
        DatasetOperationError::Reservoir { context, source }
    }
}
