PRAGMA application_id = 1397510219; -- SLPK
PRAGMA user_version = 1;

CREATE TABLE archive_meta (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  format_version INTEGER NOT NULL CHECK (format_version = 1),
  app_schema_version INTEGER NOT NULL CHECK (app_schema_version = 1),
  created_at_ms INTEGER NOT NULL CHECK (created_at_ms > 0),
  exporter_version TEXT NOT NULL CHECK (length(exporter_version) BETWEEN 1 AND 64),
  source_dataset_version INTEGER NOT NULL CHECK (source_dataset_version >= 0)
) STRICT;

CREATE TABLE app_meta (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  dataset_version INTEGER NOT NULL CHECK (dataset_version >= 0),
  last_modified_ms INTEGER NOT NULL CHECK (last_modified_ms > 0)
) STRICT;

CREATE TABLE frames (
  id TEXT PRIMARY KEY CHECK (length(id) BETWEEN 1 AND 128),
  captured_at_ms INTEGER NOT NULL CHECK (captured_at_ms > 0),
  label TEXT NOT NULL CHECK (label IN ('good', 'bad', 'away', 'unused')),
  bbox_x1 REAL NOT NULL CHECK (bbox_x1 = bbox_x1 AND abs(bbox_x1) <= 1000000000),
  bbox_y1 REAL NOT NULL CHECK (bbox_y1 = bbox_y1 AND abs(bbox_y1) <= 1000000000),
  bbox_x2 REAL NOT NULL CHECK (bbox_x2 = bbox_x2 AND abs(bbox_x2) <= 1000000000),
  bbox_y2 REAL NOT NULL CHECK (bbox_y2 = bbox_y2 AND abs(bbox_y2) <= 1000000000),
  bbox_score REAL NOT NULL CHECK (bbox_score = bbox_score AND bbox_score BETWEEN 0 AND 1),
  CHECK (bbox_x1 <= bbox_x2 AND bbox_y1 <= bbox_y2)
) STRICT;

CREATE INDEX frames_captured_at_idx ON frames(captured_at_ms DESC, id);
CREATE INDEX frames_label_idx ON frames(label, captured_at_ms DESC);

CREATE TABLE frame_keypoints (
  frame_id TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
  keypoint_index INTEGER NOT NULL CHECK (keypoint_index BETWEEN 0 AND 16),
  x REAL NOT NULL CHECK (x = x AND abs(x) <= 1000000000),
  y REAL NOT NULL CHECK (y = y AND abs(y) <= 1000000000),
  score REAL NOT NULL CHECK (score = score AND abs(score) <= 1000000000),
  PRIMARY KEY (frame_id, keypoint_index)
) WITHOUT ROWID, STRICT;

CREATE TABLE frame_features (
  frame_id TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
  feature_type TEXT NOT NULL CHECK (length(feature_type) BETWEEN 1 AND 64),
  dimension INTEGER NOT NULL CHECK (dimension BETWEEN 1 AND 1048576),
  values_le_f32 BLOB NOT NULL CHECK (length(values_le_f32) = dimension * 4),
  payload_sha256 BLOB NOT NULL CHECK (length(payload_sha256) = 32),
  PRIMARY KEY (frame_id, feature_type)
) WITHOUT ROWID, STRICT;

CREATE TABLE thumbnails (
  frame_id TEXT PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
  mime_type TEXT NOT NULL CHECK (mime_type IN ('image/jpeg', 'image/png', 'image/webp')),
  bytes BLOB NOT NULL CHECK (length(bytes) BETWEEN 1 AND 2097152),
  payload_sha256 BLOB NOT NULL CHECK (length(payload_sha256) = 32)
) STRICT;

CREATE TABLE settings (
  key TEXT PRIMARY KEY CHECK (length(key) BETWEEN 1 AND 64),
  schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
  json TEXT NOT NULL CHECK (length(json) BETWEEN 2 AND 1048576)
) STRICT;

CREATE TABLE model_generations (
  id INTEGER PRIMARY KEY,
  created_at_ms INTEGER NOT NULL CHECK (created_at_ms > 0),
  dataset_version INTEGER NOT NULL CHECK (dataset_version >= 0),
  dataset_identity_sha256 BLOB NOT NULL CHECK (length(dataset_identity_sha256) = 32),
  training_config_sha256 BLOB NOT NULL CHECK (length(training_config_sha256) = 32),
  active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1))
) STRICT;

CREATE UNIQUE INDEX one_active_model_generation_idx
  ON model_generations(active) WHERE active = 1;

CREATE TABLE models (
  generation_id INTEGER NOT NULL REFERENCES model_generations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('presence', 'posture')),
  envelope_version INTEGER NOT NULL CHECK (envelope_version = 1),
  payload BLOB NOT NULL CHECK (length(payload) BETWEEN 1 AND 268435456),
  payload_sha256 BLOB NOT NULL CHECK (length(payload_sha256) = 32),
  PRIMARY KEY (generation_id, role)
) WITHOUT ROWID, STRICT;

CREATE TABLE reservoir_state (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  capacity INTEGER NOT NULL CHECK (capacity BETWEEN 1 AND 1000000),
  seen_count INTEGER NOT NULL CHECK (seen_count >= 0),
  sample_count INTEGER NOT NULL CHECK (sample_count BETWEEN 0 AND capacity),
  rng_state INTEGER NOT NULL CHECK (rng_state >= 0),
  last_sampled_ms INTEGER NOT NULL CHECK (last_sampled_ms >= 0)
) STRICT;

CREATE TABLE reservoir_samples (
  slot INTEGER PRIMARY KEY CHECK (slot >= 0),
  payload BLOB NOT NULL CHECK (length(payload) BETWEEN 1 AND 8388608),
  payload_sha256 BLOB NOT NULL CHECK (length(payload_sha256) = 32)
) STRICT;
