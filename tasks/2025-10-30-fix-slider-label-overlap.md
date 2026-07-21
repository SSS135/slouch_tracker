# Task 2025-10-30: Fix Slider Label Overlap & Unify Slider Components

**STATUS:** COMPLETED

## User Request
"every slider has min max values overlapping with text below [Image #1]"

Follow-up: "unify the duplicate slider components in the codebase"

## Critical Discoveries

**1. CSS overrides for Mantine marks don't work reliably:**
Initial approach used `styles` prop with `transform` CSS to align mark labels. This failed - Mantine's internal styling has higher specificity, making CSS overrides unreliable and brittle.

**2. Three separate slider implementations existed:**
- `src/components/ui/Slider.tsx` - Unused reusable component with basic features
- RuntimeTab `SliderField` - Inline component used 6 times for linear scales with help text
- ClassifierSelector `RangeParam` - Inline component with exponential scale support and click-to-edit functionality

**3. Architectural solution beats CSS hacks:**
Instead of fighting Mantine's mark positioning, remove `marks` prop entirely and use custom `Group` component with `justify="space-between"` for perfect left/right alignment.

**4. Mantine's label prop controls WHAT is displayed, not WHETHER:**
After unification, exponential sliders displayed raw position values (e.g., "0.429") instead of actual formatted values (e.g., "2.5"). Mantine shows a tooltip by default during interaction. When `label` prop is `undefined`, Mantine displays the raw internal slider value. For exponential scales using 0-1 internal range, this showed positions instead of converted values. The fix: always provide a `label` function (not conditionally) to ensure proper value conversion and formatting.

## Solution

**Phase 1: Fixed label overlap** - Removed Mantine marks, added custom label Groups with proper spacing.

**Phase 2: Unified all slider implementations** - Enhanced `src/components/ui/Slider.tsx` with all features from the three implementations:

```tsx
<Slider
  label="ML Confidence Threshold"
  value={0.75}
  minimumValue={0}
  maximumValue={1}
  step={0.01}
  onValueChange={onChange}
  scale="exponential"        // NEW: logarithmic mapping for ML hyperparameters
  editable={true}            // NEW: click value to enter precise number
  showTooltip={true}         // NEW: show formatted value on hover
  helpText="Threshold for..." // NEW: help text below slider
/>
```

**Key features added:**
1. **Exponential scale** - Logarithmic value mapping for ML hyperparameters (C, learning rate)
2. **Click-to-edit** - Click value to enter precise number via TextInput (Enter/Escape to submit/cancel)
3. **Help text** - Optional description below slider
4. **Tooltip** - Hover shows formatted value
5. **Min/max labels** - Conditional display (auto-hide for exponential scales)
6. **Accessibility** - aria-label support

**Migration:**
1. RuntimeTab (6 sliders) - Replaced inline `SliderField` with unified `Slider`, deleted `SliderField` function
2. ClassifierSelector - Replaced inline `RangeParam` with unified `Slider`, deleted `RangeParam`, `valueToPosition`, `positionToValue`, `formatParamValue` helpers

**Phase 3: Fixed exponential tooltip display** - Changed `label` prop to **always** provide a conversion function (not conditional on `showTooltip`). This ensures Mantine's default tooltip displays correctly converted values:

```typescript
label={sliderValue => {
  // Convert slider position to actual value (important for exponential scale)
  const actualValue = scale === 'exponential'
    ? positionToValue(sliderValue, minimumValue, maximumValue, scale)
    : sliderValue;
  return formatValue
    ? formatValue(actualValue)
    : formatParamValue(actualValue, scale, step);
}}
```

Initial fix was conditional on `showTooltip` prop, but this was incorrect - Mantine shows tooltips by default during interaction regardless of custom label function. The `label` prop controls **what** is displayed, not **whether** it's displayed. Always provide conversion logic to ensure correct values.

## Lessons

- CSS overrides via `styles` prop unreliable for Mantine components with complex internal styling
- Architectural changes (removing features, custom components) often cleaner than CSS hacks
- Component unification eliminates ~200 lines of duplicate code and ensures consistency
- Single source of truth for slider UI makes features easier to maintain and extend
- All sliders now have consistent behavior, styling, and accessibility support
- **Mantine's `label` prop controls WHAT is displayed, not WHETHER** - Always provide conversion logic; Mantine shows tooltips by default during interaction
- Conditional tooltip logic based on props was wrong approach - the library handles visibility, we just provide formatting

## Files Modified

- `src/components/ui/Slider.tsx` (enhanced to 273 lines with all features)
- `src/components/unified/RuntimeTab.tsx` (migrated to unified Slider, deleted SliderField)
- `src/components/dataset/ClassifierSelector.tsx` (migrated to unified Slider, deleted RangeParam and helpers)

## Impact

All sliders now use a single unified component with proper spacing, consistent features, and no visual overlap. Build successful with no TypeScript errors. All 6 Runtime tab sliders and classifier parameter sliders support tooltips with correctly formatted values, help text, exponential scales, and click-to-edit. Exponential sliders display actual values (e.g., "0.001", "100") in tooltips instead of raw positions.
