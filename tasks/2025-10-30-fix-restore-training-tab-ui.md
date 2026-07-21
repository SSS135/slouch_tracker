# Task 2025-10-30: Restore Training Tab UI After Migration
**STATUS:** COMPLETED

## User Request
"we recently migrated to react dom + mantine. the training tab looks similar, but dataset box and normalisation feature are different. make them same as before. also move train box just below dataset box. you replaced dataset collection, with some post-migration half assed shit. you did not restore dataset statistics tab as it was before. also completely remove current training dataset view you used, it should not exist"

Follow-up: "do feature normalisation, random projection, classifier use same control element? if not, unify it"

Follow-up 2: "inline dim reduction selector into training tab"

## Critical Discoveries (Non-Obvious)

**1. Separate Overview section was missing**
Pre-migration had dedicated "Overview" section for statistics only. Post-migration crammed stats into "Dataset" section with frame management, breaking visual separation.

**2. DatasetStatsCard was gutted**
Pre-migration: rich visual design (progress bars, colored badges, warning icons). Post-migration: plain text list. Lost all visual feedback for data quality.

**3. Wrong component used**
Migration used DatasetFrameTable (plain table) instead of UnifiedFrameGrid (visual grid with drag-and-drop). Grid is needed for dataset frame management.

**4. Section order wrong**
Pre-migration: Overview → Feature Types → Preprocessing → Classifier → Train → Dataset. Post-migration jumbled this, breaking workflow logic.

**5. ClassifierSelector used inconsistent UI**
Feature Normalization and Dimensionality Reduction used RadioGroup component, but Classifier Selection used custom Mantine Radio.Group + Paper. No badge support, inconsistent styling.

**6. DimReductionSelector component created unnecessary indirection**
Separate component for dimensionality reduction added complexity. All other training config controls (normalization, classifier) were inline in TrainingTab. Inlining dim reduction simplified component hierarchy.

## Solution

**DatasetStatsCard.tsx** - Restored visual features:
- Large colored stats grid (Total: white, Good: green, Bad: red, Away: blue, Unused: gray)
- Progress bar with colored segments showing class distribution
- Warning badges with icons: Red (⚠️ no data), Yellow (⚠️ imbalance >35%), Yellow (ℹ️ no away frames), Green (✅ ready)
- Mantine components: Alert, Progress.Root, Progress.Section

**TrainingTab.tsx** - Complete restructure:
1. Replaced DatasetFrameTable import → UnifiedFrameGrid
2. Created separate "Overview" section at top (DatasetStatsCard only)
3. Reordered sections: Overview → Feature Types → Preprocessing → Classifier → Train Model → Dataset (collapsible)
4. Made Dataset section collapsible with UnifiedFrameGrid
5. Restored handleFrameDrag handler (needed by UnifiedFrameGrid)
6. Removed handleUpdateLabel (not needed)

**Normalization UI Restoration:**
- Replaced SegmentedControl → RadioGroup for normalization options
- Added full descriptions: "No feature normalization" / "Normalize each sample independently" / "Normalize each feature dimension"
- Added "Recommended" badge (green) on Per-Feature Normalization
- Added HelpText: "Normalization improves model training stability. Per-feature normalization is recommended."
- Changed section title: "Normalization & Dimensionality Reduction" → "Preprocessing" with subtitle
- Changed label: "Normalization" → "Feature Normalization"

**ClassifierSelector.tsx** - Unified UI controls:
- Replaced custom Radio.Group + Paper → RadioGroup component (consistent with Feature Normalization and Dimensionality Reduction)
- Added "Recommended" badge (green) to Logistic Regression classifier
- Removed unused Paper import and custom styling (~30 lines)
- Achieved consistent appearance across all three control sections

**DimReductionSelector inlining** - Removed component, integrated into TrainingTab:
- Deleted DimReductionSelector component (separate file)
- Inlined dimensionality reduction UI directly into Preprocessing section (lines 296-330 of TrainingTab)
- Added handlers: handleDimReductionMethodChange, handleDimReductionComponentsChange
- Method selection: RadioGroup (Random Projection / None)
- Dimensions selection: SegmentedControl (64 / 256 / 1024) - shown only when method != 'none'
- Added SegmentedControl to Mantine imports
- All training configuration now visible in single file (simpler component hierarchy)

**Cleanup** - Removed unused files and props:
- Deleted src/components/dataset/DimReductionSelector.tsx (no longer needed after inlining)
- Deleted src/components/dataset/TrainingModal.tsx (unused legacy file)
- Removed `collapsible` and `defaultExpanded` props from Dataset section (not supported by Mantine Section component)

## Files Modified
- src/components/dataset/DatasetStatsCard.tsx
- src/components/unified/TrainingTab.tsx
- src/components/dataset/ClassifierSelector.tsx

## Files Deleted
- src/components/dataset/DimReductionSelector.tsx
- src/components/dataset/TrainingModal.tsx

## Test Results
All relevant tests PASSED: TrainingConfigContext (15/15), useDatasetOperations (21/21). No functionality broken.

## Impact
Restored rich visual feedback for dataset statistics, proper section organization (separate Overview), and drag-and-drop frame management (UnifiedFrameGrid). UI now matches pre-migration design with improved UX.

## Related
- tasks/2025-10-29-react-dom-vite-mantine-migration.md - Migration that introduced UI differences
