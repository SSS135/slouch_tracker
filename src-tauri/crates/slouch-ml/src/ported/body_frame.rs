//! Shared body-intrinsic 3D frame primitives.
//!
//! The 3D vector type, midpoint/mean helpers, and the orthonormal `(T̂, Ŝ, F̂)` body-frame
//! construction were lifted verbatim out of `nlf_features` so both the NLF depth feature
//! and the computed `keypoints_3d_features` build the same frame from a single source. The
//! numerics are unchanged; only the visibility was widened to `pub(crate)`.

/// Body-frame degeneracy floor in the model's metric units (meters). Real seated torso
/// lengths are ~0.4-0.7 m; anything at or below this is a collapsed/absent torso.
pub(crate) const MIN_TORSO_LEN: f64 = 1e-3;
/// Minimum length of a body axis after normalization/orthogonalization. Below this the
/// shoulder line is parallel to the trunk (or the shoulders coincide) and no frame exists.
pub(crate) const MIN_AXIS_LEN: f64 = 1e-6;

#[derive(Clone, Copy)]
pub(crate) struct Vec3 {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
}

impl Vec3 {
    pub(crate) fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub(crate) fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub(crate) fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub(crate) fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub(crate) fn scale(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    pub(crate) fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
}

pub(crate) fn midpoint(first: Vec3, second: Vec3) -> Vec3 {
    first.add(second).scale(0.5)
}

pub(crate) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub(crate) struct BodyFrame {
    pub(crate) trunk_hat: Vec3,
    pub(crate) shoulder_hat: Vec3,
    pub(crate) forward_hat: Vec3,
}

/// Constructs the orthonormal body frame or returns `None` when the geometry is
/// degenerate (collapsed torso, shoulder line parallel to the trunk, or zero shoulder
/// width). `Ŝ` is Gram-Schmidt-orthogonalized against `T̂` before `F̂ = T̂ × Ŝ`, so the
/// returned basis is exactly orthonormal.
pub(crate) fn build_body_frame(
    trunk: Vec3,
    torso_len: f64,
    shoulder_axis: Vec3,
    shoulder_width: f64,
) -> Option<BodyFrame> {
    if torso_len < MIN_TORSO_LEN || shoulder_width < MIN_AXIS_LEN {
        return None;
    }
    let trunk_hat = trunk.scale(1.0 / torso_len);

    let shoulder_perp = shoulder_axis.sub(trunk_hat.scale(shoulder_axis.dot(trunk_hat)));
    let shoulder_perp_len = shoulder_perp.norm();
    if shoulder_perp_len < MIN_AXIS_LEN {
        return None;
    }
    let shoulder_hat = shoulder_perp.scale(1.0 / shoulder_perp_len);

    let forward = trunk_hat.cross(shoulder_hat);
    let forward_len = forward.norm();
    if forward_len < MIN_AXIS_LEN {
        return None;
    }
    let forward_hat = forward.scale(1.0 / forward_len);

    Some(BodyFrame {
        trunk_hat,
        shoulder_hat,
        forward_hat,
    })
}
