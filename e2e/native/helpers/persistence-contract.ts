/**
 * Shared contract between the persistence pair
 * (30-persistence-setup / 31-persistence-restart-verify).
 * Lives outside the spec files so importing it never registers foreign tests.
 */
export const PERSISTED_FRAME_ID = 'e2e-native-frame-1';

/** Distinctive non-default values; every field passes CameraSettings::validate. */
export const PERSISTED_CAMERA_SETTINGS = {
  cameraWidth: 1024,
  cameraHeight: 576,
  captureIntervalSeconds: 0.25,
  autoCaptureEnabled: false,
  autoCaptureIntervalSeconds: 7.5,
  privacyMode: false,
  claheStrength: 1.25,
  smoothingFrames: 6,
  showDetectionOverlay: true,
  // get_camera_settings serializes the FULL CameraSettings struct, so the strict
  // toEqual in 31-persistence-restart-verify needs every serde-defaulted field
  // present too. Values match slouch-domain's Rust defaults.
  tileMotionThreshold: 1.5,
  claheTemporalAlpha: 0.2,
  preprocessingDebugView: false,
  cameraIndex: 0,
};
