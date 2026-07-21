import type {
  BoundingBox,
  ClassifierConfig,
  DatasetStats,
  DimensionalityReductionConfig,
  FrameLabel as NativeFrameLabel,
  InferenceUiResult,
  Keypoint,
  NormalizationMode,
  TrainingResultResponse_Serialize,
  TrainingResult_Serialize,
  TrainingSettings_Serialize,
} from '@generated/bindings';

export enum FrameLabel {
  GOOD = 'good',
  BAD = 'bad',
  AWAY = 'away',
  UNUSED = 'unused',
}

export type ClassificationResult = NonNullable<InferenceUiResult['classification']>;
export type InferenceResult = InferenceUiResult;
export type TrainingResult = TrainingResult_Serialize;
export type TrainingSettings = TrainingSettings_Serialize;
export type { ClassifierConfig, DatasetStats, DimensionalityReductionConfig, NormalizationMode };

export interface PostureFrame {
  id: string;
  timestamp: number;
  thumbnail?: Blob;
  thumbnailMimeType: string;
  keypoints: Array<{ x: number; y: number; score: number }>;
  bbox: BoundingBox;
  label: FrameLabel;
}

export interface PostureDataset {
  frames: PostureFrame[];
  version: number;
  lastModified: number;
}

export interface CaptureAction {
  frameId: string;
  timestamp: number;
  label: FrameLabel;
  thumbnailUrl: string;
}

export interface ImportResult {
  imported: number;
  skipped: number;
  errors: string[];
}

export type NativeTrainingResult = TrainingResultResponse_Serialize;
export type NativeKeypoint = Keypoint;
export type FrameLabelValue = NativeFrameLabel;
