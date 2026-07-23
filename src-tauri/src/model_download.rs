//! First-run downloader for the NLF pose model.
//!
//! The NLF model is too large to ship in the installer, so on first launch the app
//! streams it once from a pinned release URL into `{data_dir}/models`, verifies its
//! SHA-256, and atomically renames it into place. A manually pre-placed file (either
//! in the packaged resources or dropped into `{data_dir}/models`) short-circuits the
//! whole thing: `find_resource` finds it, the status is `ready`, and no network call
//! is ever made.
//!
//! The network is abstracted behind [`ByteSource`] so the streaming/resume/verify
//! logic is exercised in tests without touching the wire.

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use serde::Serialize;
use sha2::{Digest, Sha256};

pub const POSE_MODEL_FILENAME: &str = "nlf_l_crop_fp16.onnx";

// Pinned to the `nlf-l-crop-fp16` entry in resource-lock.json. Held as constants
// (not parsed at runtime) so the download path has no startup parse/error branch;
// `lock_entry_matches_pinned_constants` asserts they stay in step with the lock.
pub const POSE_MODEL_URL: &str =
    "https://github.com/SSS135/slouch_tracker/releases/download/models-v1/nlf_l_crop_fp16.onnx";
pub const POSE_MODEL_SHA256: &str =
    "33bd300cd5a65681a5d671debd82a63f842c7420443cd9bb7424ca7aef82cca8";
pub const POSE_MODEL_BYTES: u64 = 244_722_283;

const READ_BUFFER_BYTES: usize = 64 * 1024;
// The UI wants smooth-but-cheap progress; ~4 events/second is plenty over IPC.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PoseModelStatus {
    Ready {
        path: String,
    },
    DownloadRequired {
        #[serde(rename = "totalBytes")]
        #[specta(type = specta_typescript::Number)]
        total_bytes: u64,
    },
    Downloading,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PoseModelDownloadEvent {
    Started {
        #[serde(rename = "totalBytes")]
        #[specta(type = specta_typescript::Number)]
        total_bytes: u64,
    },
    Progress {
        #[specta(type = specta_typescript::Number)]
        received: u64,
        #[specta(type = specta_typescript::Number)]
        total: u64,
    },
    Verifying,
    Ready,
    Failed {
        reason: String,
    },
}

#[derive(Debug)]
pub enum DownloadError {
    Io(String),
    Http(String),
    ChecksumMismatch { expected: String, actual: String },
    Cancelled,
}

impl fmt::Display for DownloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "file error: {message}"),
            Self::Http(message) => write!(formatter, "network error: {message}"),
            Self::ChecksumMismatch { expected, actual } => write!(
                formatter,
                "downloaded file failed checksum verification (expected {expected}, got {actual})"
            ),
            Self::Cancelled => formatter.write_str("download cancelled"),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<io::Error> for DownloadError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// A byte stream for the remote resource, starting at some offset. `start_offset`
/// is the offset the stream actually begins at: it equals the requested offset when
/// the server honored a Range request (HTTP 206), and `0` when it served the whole
/// body instead (HTTP 200), which tells the caller to restart rather than append.
pub struct RemoteStream {
    pub total_len: u64,
    pub start_offset: u64,
    pub reader: Box<dyn Read + Send>,
}

/// Abstracts fetching the resource so the download logic can be tested off-wire.
pub trait ByteSource {
    fn open(&self, offset: u64) -> Result<RemoteStream, DownloadError>;
}

/// The production [`ByteSource`]: a blocking ureq GET with HTTP Range resume.
pub struct HttpByteSource {
    url: String,
}

impl HttpByteSource {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }
}

impl ByteSource for HttpByteSource {
    fn open(&self, offset: u64) -> Result<RemoteStream, DownloadError> {
        let mut request = ureq::get(&self.url);
        if offset > 0 {
            request = request.set("Range", &format!("bytes={offset}-"));
        }
        let response = match request.call() {
            Ok(response) => response,
            Err(ureq::Error::Status(code, _)) => {
                return Err(DownloadError::Http(format!("server returned HTTP {code}")))
            }
            Err(error) => return Err(DownloadError::Http(error.to_string())),
        };
        let (total_len, start_offset) = if response.status() == 206 {
            let total = response
                .header("Content-Range")
                .and_then(content_range_total)
                .ok_or_else(|| {
                    DownloadError::Http("partial response missing a Content-Range total".into())
                })?;
            (total, offset)
        } else {
            let total = response
                .header("Content-Length")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            (total, 0)
        };
        Ok(RemoteStream {
            total_len,
            start_offset,
            reader: Box::new(response.into_reader()),
        })
    }
}

/// The sibling `<name>.partial` path a download streams into before verification.
pub fn partial_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(".partial");
    PathBuf::from(name)
}

/// Classifies the pose model for the status command from three cheap signals.
pub fn pose_model_status(
    resolved: Option<PathBuf>,
    running: bool,
    partial_present: bool,
) -> PoseModelStatus {
    if let Some(path) = resolved {
        PoseModelStatus::Ready {
            path: path.to_string_lossy().into_owned(),
        }
    } else if running || partial_present {
        PoseModelStatus::Downloading
    } else {
        PoseModelStatus::DownloadRequired {
            total_bytes: POSE_MODEL_BYTES,
        }
    }
}

/// Streams `source` into `target` with resume, SHA-256 verification, and an atomic
/// rename. Emits `Started`/`Progress`/`Verifying` through `on_progress`; the caller
/// emits the terminal `Ready`/`Failed`. If `target` already exists no network call
/// is made. A checksum mismatch deletes the `.partial` so the next attempt restarts.
pub fn run_pose_download(
    source: &dyn ByteSource,
    target: &Path,
    expected_sha256: &str,
    expected_len: u64,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(PoseModelDownloadEvent),
) -> Result<(), DownloadError> {
    if target.is_file() {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let partial = partial_path(target);

    let existing_len = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
    // Only resume a partial that is a strict prefix; anything else (oversized or
    // exactly full-but-unverified) is discarded so the fetch starts clean.
    let resume_from = if existing_len > 0 && existing_len < expected_len {
        existing_len
    } else {
        if existing_len > 0 {
            let _ = fs::remove_file(&partial);
        }
        0
    };

    let stream = source.open(resume_from)?;
    let total = stream.total_len.max(expected_len);
    on_progress(PoseModelDownloadEvent::Started { total_bytes: total });

    let mut file = if stream.start_offset > 0 {
        OpenOptions::new().append(true).open(&partial)?
    } else {
        File::create(&partial)?
    };

    let mut received = stream.start_offset;
    let mut reader = stream.reader;
    let mut buffer = [0u8; READ_BUFFER_BYTES];
    let mut last_emit = Instant::now();
    on_progress(PoseModelDownloadEvent::Progress { received, total });
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        received += read as u64;
        if last_emit.elapsed() >= PROGRESS_INTERVAL {
            on_progress(PoseModelDownloadEvent::Progress { received, total });
            last_emit = Instant::now();
        }
    }
    file.flush()?;
    drop(file);
    on_progress(PoseModelDownloadEvent::Progress { received, total });

    on_progress(PoseModelDownloadEvent::Verifying);
    let actual = hex_encode(&sha256_file(&partial)?);
    if actual != expected_sha256.to_ascii_lowercase() {
        let _ = fs::remove_file(&partial);
        return Err(DownloadError::ChecksumMismatch {
            expected: expected_sha256.to_ascii_lowercase(),
            actual,
        });
    }

    // Same-directory rename over NTFS is atomic; readers see the whole verified file
    // or nothing, never a half-written one.
    fs::rename(&partial, target)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<[u8; 32], DownloadError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; READ_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn hex_encode(bytes: &[u8]) -> String {
    use fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn content_range_total(header: &str) -> Option<u64> {
    header.rsplit('/').next()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::Mutex;

    struct StubSource {
        bytes: Vec<u8>,
        honor_range: bool,
        requested_offset: Mutex<Option<u64>>,
    }

    impl StubSource {
        fn new(bytes: Vec<u8>, honor_range: bool) -> Self {
            Self {
                bytes,
                honor_range,
                requested_offset: Mutex::new(None),
            }
        }
    }

    impl ByteSource for StubSource {
        fn open(&self, offset: u64) -> Result<RemoteStream, DownloadError> {
            *self.requested_offset.lock().unwrap() = Some(offset);
            if self.honor_range && offset > 0 {
                let remaining = self.bytes[offset as usize..].to_vec();
                Ok(RemoteStream {
                    total_len: self.bytes.len() as u64,
                    start_offset: offset,
                    reader: Box::new(Cursor::new(remaining)),
                })
            } else {
                Ok(RemoteStream {
                    total_len: self.bytes.len() as u64,
                    start_offset: 0,
                    reader: Box::new(Cursor::new(self.bytes.clone())),
                })
            }
        }
    }

    struct PanickingSource;
    impl ByteSource for PanickingSource {
        fn open(&self, _offset: u64) -> Result<RemoteStream, DownloadError> {
            panic!("network must not be touched when the model is already present");
        }
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("slouch-model-download-{tag}-{nonce}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sha_hex(bytes: &[u8]) -> String {
        hex_encode(&Sha256::digest(bytes))
    }

    #[test]
    fn download_verifies_and_atomically_renames_on_success() {
        let dir = scratch_dir("ok");
        let target = dir.join("model.bin");
        let bytes = vec![7u8; 4096];
        let source = StubSource::new(bytes.clone(), false);

        run_pose_download(
            &source,
            &target,
            &sha_hex(&bytes),
            bytes.len() as u64,
            &AtomicBool::new(false),
            |_| {},
        )
        .expect("download succeeds");

        assert_eq!(fs::read(&target).unwrap(), bytes);
        assert!(!partial_path(&target).exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn checksum_mismatch_deletes_partial_and_never_creates_target() {
        let dir = scratch_dir("bad-sha");
        let target = dir.join("model.bin");
        let bytes = vec![1u8; 2048];
        let source = StubSource::new(bytes.clone(), false);

        let wrong_sha = sha_hex(&vec![2u8; 2048]);
        let error = run_pose_download(
            &source,
            &target,
            &wrong_sha,
            bytes.len() as u64,
            &AtomicBool::new(false),
            |_| {},
        )
        .expect_err("checksum mismatch fails");

        assert!(matches!(error, DownloadError::ChecksumMismatch { .. }));
        assert!(!target.exists());
        assert!(!partial_path(&target).exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resumes_from_partial_fetching_only_the_remainder() {
        let dir = scratch_dir("resume");
        let target = dir.join("model.bin");
        let bytes: Vec<u8> = (0..3000u32).map(|value| value as u8).collect();
        let prefix = 1200usize;
        fs::write(partial_path(&target), &bytes[..prefix]).unwrap();

        let source = StubSource::new(bytes.clone(), true);
        run_pose_download(
            &source,
            &target,
            &sha_hex(&bytes),
            bytes.len() as u64,
            &AtomicBool::new(false),
            |_| {},
        )
        .expect("resume succeeds");

        assert_eq!(
            *source.requested_offset.lock().unwrap(),
            Some(prefix as u64),
            "resume must request the byte offset of the existing partial"
        );
        assert_eq!(fs::read(&target).unwrap(), bytes);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restarts_when_server_ignores_range() {
        let dir = scratch_dir("restart");
        let target = dir.join("model.bin");
        let bytes: Vec<u8> = (0..3000u32).map(|value| (value * 3) as u8).collect();
        // A stale/corrupt partial with wrong bytes; server ignores Range (200).
        fs::write(partial_path(&target), vec![0xFFu8; 1200]).unwrap();

        let source = StubSource::new(bytes.clone(), false);
        run_pose_download(
            &source,
            &target,
            &sha_hex(&bytes),
            bytes.len() as u64,
            &AtomicBool::new(false),
            |_| {},
        )
        .expect("restart succeeds");

        assert_eq!(fs::read(&target).unwrap(), bytes);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_network_when_target_already_present() {
        let dir = scratch_dir("present");
        let target = dir.join("model.bin");
        fs::write(&target, b"already here").unwrap();

        run_pose_download(
            &PanickingSource,
            &target,
            POSE_MODEL_SHA256,
            POSE_MODEL_BYTES,
            &AtomicBool::new(false),
            |_| panic!("no progress events when the file is present"),
        )
        .expect("present file short-circuits");

        assert_eq!(fs::read(&target).unwrap(), b"already here");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cancellation_stops_the_stream() {
        let dir = scratch_dir("cancel");
        let target = dir.join("model.bin");
        let bytes = vec![9u8; 8192];
        let source = StubSource::new(bytes.clone(), false);

        let error = run_pose_download(
            &source,
            &target,
            &sha_hex(&bytes),
            bytes.len() as u64,
            &AtomicBool::new(true),
            |_| {},
        )
        .expect_err("pre-set cancel aborts");

        assert!(matches!(error, DownloadError::Cancelled));
        assert!(!target.exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn status_reflects_resolved_running_and_partial_signals() {
        assert!(matches!(
            pose_model_status(None, false, false),
            PoseModelStatus::DownloadRequired { .. }
        ));
        assert!(matches!(
            pose_model_status(None, true, false),
            PoseModelStatus::Downloading
        ));
        assert!(matches!(
            pose_model_status(None, false, true),
            PoseModelStatus::Downloading
        ));
        match pose_model_status(Some(PathBuf::from("/models/nlf.onnx")), false, false) {
            PoseModelStatus::Ready { path } => assert!(path.contains("nlf.onnx")),
            other => panic!("expected ready, got {other:?}"),
        }
    }

    #[test]
    fn content_range_total_parses_trailing_size() {
        assert_eq!(content_range_total("bytes 200-1023/1024"), Some(1024));
        assert_eq!(content_range_total("bytes 0-0/500"), Some(500));
        assert_eq!(content_range_total("bytes */*"), None);
    }

    #[test]
    fn lock_entry_matches_pinned_constants() {
        let lock: serde_json::Value =
            serde_json::from_str(include_str!("../resource-lock.json")).expect("lock parses");
        let entry = lock["resources"]
            .as_array()
            .expect("resources array")
            .iter()
            .find(|entry| entry["id"] == "nlf-l-crop-fp16")
            .expect("nlf lock entry present");
        assert_eq!(entry["sha256"], POSE_MODEL_SHA256);
        assert_eq!(entry["bytes"].as_u64(), Some(POSE_MODEL_BYTES));
    }
}
