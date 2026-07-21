use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct BoundingBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub score: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct ExpandedBbox {
    pub original: BoundingBox,
    pub expanded: BoundingBox,
}

pub trait BboxAccessor {
    fn original_bbox(&self) -> &BoundingBox;
}

impl BboxAccessor for BoundingBox {
    fn original_bbox(&self) -> &BoundingBox {
        self
    }
}

impl BboxAccessor for ExpandedBbox {
    fn original_bbox(&self) -> &BoundingBox {
        &self.original
    }
}
