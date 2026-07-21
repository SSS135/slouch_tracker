//! COCO keypoint indices shared with the canonical domain boundary.

pub use crate::keypoint::{
    COCO_KEYPOINT_COUNT, LEFT_ANKLE, LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_KNEE,
    LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW, RIGHT_EYE, RIGHT_HIP,
    RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_coco_indices_match_typescript_source() {
        assert_eq!(NOSE, 0);
        assert_eq!(LEFT_EYE, 1);
        assert_eq!(RIGHT_EYE, 2);
        assert_eq!(LEFT_EAR, 3);
        assert_eq!(RIGHT_EAR, 4);
        assert_eq!(LEFT_SHOULDER, 5);
        assert_eq!(RIGHT_SHOULDER, 6);
        assert_eq!(LEFT_ELBOW, 7);
        assert_eq!(RIGHT_ELBOW, 8);
        assert_eq!(LEFT_WRIST, 9);
        assert_eq!(RIGHT_WRIST, 10);
        assert_eq!(LEFT_HIP, 11);
        assert_eq!(RIGHT_HIP, 12);
        assert_eq!(LEFT_KNEE, 13);
        assert_eq!(RIGHT_KNEE, 14);
        assert_eq!(LEFT_ANKLE, 15);
        assert_eq!(RIGHT_ANKLE, 16);
        assert_eq!(COCO_KEYPOINT_COUNT, 17);
    }
}
