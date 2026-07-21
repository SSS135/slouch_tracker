use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CameraSettings {
    pub camera_width: u32,
    pub camera_height: u32,
    pub capture_interval_seconds: f64,
    pub auto_capture_enabled: bool,
    pub auto_capture_interval_seconds: f64,
    pub privacy_mode: bool,
    pub clahe_strength: f64,
    pub gaussian_blur_kernel: u8,
    pub smoothing_frames: u8,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            camera_width: 800,
            camera_height: 600,
            // 1 fps is the detection rate the app actually needs; 0.2 (5 fps)
            // was an over-aggressive speed-wave value that multiplied CPU.
            capture_interval_seconds: 1.0,
            auto_capture_enabled: true,
            auto_capture_interval_seconds: 2.0,
            privacy_mode: true,
            clahe_strength: 3.5,
            gaussian_blur_kernel: 5,
            smoothing_frames: 3,
        }
    }
}

impl CameraSettings {
    pub fn validate(&self) -> Result<(), String> {
        if !(160..=1920).contains(&self.camera_width) {
            return Err("cameraWidth must be between 160 and 1920".into());
        }
        if !(120..=1080).contains(&self.camera_height) {
            return Err("cameraHeight must be between 120 and 1080".into());
        }
        validate_finite_range(
            self.capture_interval_seconds,
            0.05,
            60.0,
            "captureIntervalSeconds",
        )?;
        validate_finite_range(
            self.auto_capture_interval_seconds,
            0.1,
            3600.0,
            "autoCaptureIntervalSeconds",
        )?;
        validate_finite_range(self.clahe_strength, 0.0, 10.0, "claheStrength")?;
        if self.gaussian_blur_kernel > 15
            || (self.gaussian_blur_kernel != 0 && self.gaussian_blur_kernel.is_multiple_of(2))
        {
            return Err(
                "gaussianBlurKernel must be zero or an odd integer between 1 and 15".into(),
            );
        }
        if !(1..=10).contains(&self.smoothing_frames) {
            return Err("smoothingFrames must be between 1 and 10".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UiSettings {
    pub alert_volume: f64,
    pub alert_delay_seconds: f64,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            alert_volume: 0.3,
            alert_delay_seconds: 5.0,
        }
    }
}

impl UiSettings {
    pub fn validate(&self) -> Result<(), String> {
        validate_finite_range(self.alert_volume, 0.0, 1.0, "alertVolume")?;
        validate_finite_range(self.alert_delay_seconds, 0.0, 3600.0, "alertDelaySeconds")
    }
}

fn validate_finite_range(value: f64, minimum: f64, maximum: f64, name: &str) -> Result<(), String> {
    if !value.is_finite() || value < minimum || value > maximum {
        return Err(format!(
            "{name} must be a finite number between {minimum} and {maximum}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CameraSettings, UiSettings};

    #[test]
    fn defaults_are_valid() {
        CameraSettings::default()
            .validate()
            .expect("camera defaults");
        UiSettings::default().validate().expect("UI defaults");
    }

    #[test]
    fn rejects_non_finite_and_noncanonical_values() {
        let camera = CameraSettings {
            capture_interval_seconds: f64::NAN,
            ..CameraSettings::default()
        };
        assert!(camera.validate().is_err());

        let camera = CameraSettings {
            gaussian_blur_kernel: 4,
            ..CameraSettings::default()
        };
        assert!(camera.validate().is_err());

        let ui = UiSettings {
            alert_volume: 1.01,
            ..UiSettings::default()
        };
        assert!(ui.validate().is_err());
    }
}
