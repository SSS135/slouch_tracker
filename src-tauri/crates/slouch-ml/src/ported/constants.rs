#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtmposeModelConfig {
    pub name: &'static str,
    pub path: &'static str,
    pub backbone_channels: usize,
    pub gau_features: usize,
}

pub const RTMPOSE_MODEL_CONFIG: RtmposeModelConfig = RtmposeModelConfig {
    name: "RTMPose-M",
    path: "rtmpose-m.onnx",
    backbone_channels: 768,
    gau_features: 256,
};

pub const RTMPOSE_BACKBONE_SHAPE: [usize; 4] = [1, RTMPOSE_MODEL_CONFIG.backbone_channels, 8, 6];
pub const RTMPOSE_GAU_SHAPE: [usize; 3] = [1, 17, RTMPOSE_MODEL_CONFIG.gau_features];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

pub const RTMPOSE_INPUT_SIZE: ImageSize = ImageSize {
    width: 192,
    height: 256,
};

pub const RTMPOSE_MEAN_RGB: [f32; 3] = [123.675, 116.28, 103.53];
pub const RTMPOSE_STD_RGB: [f32; 3] = [58.395, 57.12, 57.375];
pub const RTMPOSE_SIMCC_SPLIT_RATIO: f64 = 2.0;
pub const RTMPOSE_NUM_KEYPOINTS: usize = 17;

pub const RTMPOSE_BACKBONE_RAW_DIMS: usize =
    RTMPOSE_BACKBONE_SHAPE[1] * RTMPOSE_BACKBONE_SHAPE[2] * RTMPOSE_BACKBONE_SHAPE[3];
pub const RTMPOSE_BACKBONE_POOLED_DIMS: usize = RTMPOSE_BACKBONE_SHAPE[1];
pub const RTMPOSE_GAU_RAW_DIMS: usize = RTMPOSE_GAU_SHAPE[1] * RTMPOSE_GAU_SHAPE[2];
pub const RTMPOSE_GAU_POOLED_DIMS: usize = RTMPOSE_GAU_SHAPE[2];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtmDetShape {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtmDetOutputNames {
    pub cls_p5: &'static str,
    pub reg_p5: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtmDetModel {
    pub name: &'static str,
    pub path: &'static str,
    pub shape: RtmDetShape,
    pub output_names: RtmDetOutputNames,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtmDetModels {
    pub nano: RtmDetModel,
    pub tiny: RtmDetModel,
    pub s: RtmDetModel,
}

pub const RTMDET_MODELS: RtmDetModels = RtmDetModels {
    nano: RtmDetModel {
        name: "RTMDet-Nano",
        path: "rtmdet-nano.onnx",
        shape: RtmDetShape {
            batch: 1,
            channels: 64,
            height: 10,
            width: 10,
        },
        output_names: RtmDetOutputNames {
            cls_p5: "/bbox_head/cls_convs.2.1/pointwise_conv/activate/Mul_output_0",
            reg_p5: "/bbox_head/reg_convs.2.1/pointwise_conv/activate/Mul_output_0",
        },
    },
    tiny: RtmDetModel {
        name: "RTMDet-Tiny",
        path: "rtmdet_tiny_320.onnx",
        shape: RtmDetShape {
            batch: 1,
            channels: 96,
            height: 10,
            width: 10,
        },
        output_names: RtmDetOutputNames {
            cls_p5: "/bbox_head/cls_convs.2.1/activate/Mul_output_0",
            reg_p5: "/bbox_head/reg_convs.2.1/activate/Mul_output_0",
        },
    },
    s: RtmDetModel {
        name: "RTMDet-S",
        path: "rtmdet_s_320.onnx",
        shape: RtmDetShape {
            batch: 1,
            channels: 128,
            height: 10,
            width: 10,
        },
        output_names: RtmDetOutputNames {
            cls_p5: "/bbox_head/cls_convs.2.1/activate/Mul_output_0",
            reg_p5: "/bbox_head/reg_convs.2.1/activate/Mul_output_0",
        },
    },
};

pub const RTMDET_MODEL: RtmDetModel = RTMDET_MODELS.nano;
pub const RTMDET_SHAPE: RtmDetShape = RTMDET_MODEL.shape;
pub const RTMDET_OUTPUT_NAMES: RtmDetOutputNames = RTMDET_MODEL.output_names;
pub const RTMDET_RAW_DIMS: usize = RTMDET_SHAPE.channels * RTMDET_SHAPE.height * RTMDET_SHAPE.width;
pub const RTMDET_EXTRACTED_DIMS: usize = 2 * 3 * RTMDET_SHAPE.channels;
pub const RTMDET_INPUT_SIZE: ImageSize = ImageSize {
    width: 320,
    height: 320,
};
pub const PERSON_DETECTION_CONFIDENCE: f64 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraResolution {
    pub width: usize,
    pub height: usize,
    pub frame_rate: usize,
}

pub const CAMERA_RESOLUTION: CameraResolution = CameraResolution {
    width: 800,
    height: 600,
    frame_rate: 30,
};

pub const THUMBNAIL_RESOLUTION: ImageSize = ImageSize {
    width: 640,
    height: 480,
};

pub const FLOAT32_BYTES: usize = 4;
pub const RTMPOSE_BACKBONE_POOLED_STORAGE_COST: usize =
    RTMPOSE_BACKBONE_POOLED_DIMS * FLOAT32_BYTES;
pub const RTMPOSE_GAU_POOLED_STORAGE_COST: usize = RTMPOSE_GAU_POOLED_DIMS * FLOAT32_BYTES;
pub const RTMDET_EXTRACTED_STORAGE_COST: usize = RTMDET_EXTRACTED_DIMS * FLOAT32_BYTES;

pub const EPSILON_STABLE: f32 = 1e-6;

pub const SOFT_BIN_PERCENTILES: [usize; 9] = [10, 20, 30, 40, 50, 60, 70, 80, 90];
pub const NUM_SOFT_BINS: usize = SOFT_BIN_PERCENTILES.len();

pub const ENGINEERED_FEATURES_LIST: [&str; 6] = [
    "neck_shoulder_ratio",
    "neck_eye_ratio",
    "neck_ear_ratio",
    "ear_eye_vertical",
    "head_rotation",
    "neck_length",
];

pub const NUM_SOFT_BINS_5: usize = 5;
pub const ENGINEERED_1D_DIMS: usize = ENGINEERED_FEATURES_LIST.len() * NUM_SOFT_BINS;
pub const ENGINEERED_2D_DIMS: usize = NUM_SOFT_BINS * NUM_SOFT_BINS;
pub const ENGINEERED_3D_DIMS: usize = NUM_SOFT_BINS_5 * NUM_SOFT_BINS_5 * NUM_SOFT_BINS_5;
pub const NUM_3D_BINS: usize = NUM_SOFT_BINS_5;
pub const JOINT_2D_DIMS: usize = ENGINEERED_2D_DIMS;
pub const JOINT_3D_DIMS: usize = ENGINEERED_3D_DIMS;
pub const ENGINEERED_4D_DIMS: usize =
    NUM_SOFT_BINS_5 * NUM_SOFT_BINS_5 * NUM_SOFT_BINS_5 * NUM_SOFT_BINS_5;
pub const JOINT_4D_DIMS: usize = ENGINEERED_4D_DIMS;
pub const RTMDET_ENGINEERED_DIMS: usize = 81 + (6 * NUM_SOFT_BINS);
pub const POSTURE_RAW_DIMS: usize = 5;
pub const POSTURE_GEOMETRY_DIMS: usize = 10;
pub const TORSO_INVARIANT_DIMS: usize = 7;
pub const KEYPOINT_SCORES_DIMS: usize = 17;
pub const RAW_KEYPOINTS_DIMS: usize = 34;

pub const KEYPOINT_RENDER_MIN_CONFIDENCE: f64 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlfOutputNames {
    pub coords2d: &'static str,
    pub coords3d_rel: &'static str,
    pub uncertainty: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlfModelConfig {
    pub name: &'static str,
    pub path: &'static str,
    pub input_side: usize,
    pub num_canonical: usize,
    pub output_names: NlfOutputNames,
}

/// NLF-L crop model (EfficientNetV2-L backbone). fp16 weights with fp32 IO
/// (`keep_io_types=True`), so the existing `Array4<f32>` session seam is unchanged.
/// Input `image` is `[1, 3, 384, 384]` RGB in `[0, 1]`; outputs are all fp32.
pub const NLF_MODEL_CONFIG: NlfModelConfig = NlfModelConfig {
    name: "NLF-L",
    path: "nlf_l_crop_fp16.onnx",
    input_side: 384,
    num_canonical: 867,
    output_names: NlfOutputNames {
        coords2d: "coords2d",
        coords3d_rel: "coords3d_rel",
        uncertainty: "uncertainty",
    },
};

/// Number of baked canonical joints; `coords3d_rel` is `[1, 867, 3]`, `uncertainty` `[1, 867]`.
pub const NLF_NUM_CANONICAL: usize = NLF_MODEL_CONFIG.num_canonical;

// coco_19 posture-joint output indices into the 867 canonical points (from
// `models/nlf_joint_map.json`: output_index == canonical index).
pub const NLF_JOINT_NECK: usize = 75;
pub const NLF_JOINT_NOSE: usize = 76;
pub const NLF_JOINT_LSHO: usize = 77;
pub const NLF_JOINT_RSHO: usize = 83;
pub const NLF_JOINT_LHIP: usize = 80;
pub const NLF_JOINT_RHIP: usize = 86;
pub const NLF_JOINT_LEAR: usize = 90;
pub const NLF_JOINT_REAR: usize = 92;
pub const NLF_JOINT_LEYE: usize = 89;
pub const NLF_JOINT_REYE: usize = 91;
pub const NLF_JOINT_PELV: usize = 93;
// Lower-body joints used only for the truncation-uncertainty signal.
pub const NLF_JOINT_LKNE: usize = 81;
pub const NLF_JOINT_LANK: usize = 82;
pub const NLF_JOINT_RKNE: usize = 87;
pub const NLF_JOINT_RANK: usize = 88;

/// FROZEN feature width of the NLF depth feature. Changing it would make every
/// already-stored `nlf_depth` frame fail the storage dimension check. Kept in sync
/// with the literal `14` in `slouch_domain::FeatureId::NlfDepth` metadata.
pub const NLF_DEPTH_DIMS: usize = 14;
pub const NLF_DEPTH_STORAGE_COST: usize = NLF_DEPTH_DIMS * FLOAT32_BYTES;
