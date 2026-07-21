<script module lang="ts">
  let nextSliderId = 0;
</script>

<script lang="ts">
  import { untrack } from 'svelte';
  import { logger } from '@/services/logging/logger';

  export interface SliderProps {
    label: string;
    value: number;
    minimumValue?: number;
    maximumValue?: number;
    step?: number;
    onValueChange: (value: number) => void;
    disabled?: boolean;
    formatValue?: (value: number) => string;
    accessibilityLabel?: string;
    scale?: 'linear' | 'exponential';
    helpText?: string;
    showMinMax?: boolean;
    editable?: boolean;
    showTooltip?: boolean;
    fixedValues?: number[];
  }

  let {
    label,
    value,
    minimumValue = 0,
    maximumValue = 100,
    step = 1,
    onValueChange,
    disabled = false,
    formatValue,
    accessibilityLabel,
    scale = 'linear',
    helpText,
    showMinMax,
    editable = false,
    showTooltip = false,
    fixedValues,
  }: SliderProps = $props();

  function valueToPosition(
    currentValue: number,
    min: number,
    max: number,
    currentScale?: 'linear' | 'exponential',
  ): number {
    if (currentScale === 'exponential') {
      if (min <= 0) {
        logger.warn('debug', 'Exponential scale requires min > 0, falling back to linear');
        return (currentValue - min) / (max - min);
      }
      const logMin = Math.log(min);
      const logMax = Math.log(max);
      const logValue = Math.log(Math.max(min, Math.min(max, currentValue)));
      return (logValue - logMin) / (logMax - logMin);
    }
    return (currentValue - min) / (max - min);
  }

  function positionToValue(
    position: number,
    min: number,
    max: number,
    currentScale?: 'linear' | 'exponential',
  ): number {
    if (currentScale === 'exponential') {
      if (min <= 0) {
        logger.warn('debug', 'Exponential scale requires min > 0, falling back to linear');
        return min + position * (max - min);
      }
      const logMin = Math.log(min);
      const logMax = Math.log(max);
      return Math.exp(logMin + position * (logMax - logMin));
    }
    return min + position * (max - min);
  }

  function formatParamValue(
    currentValue: number,
    currentScale?: 'linear' | 'exponential',
    currentStep?: number,
  ): string {
    if (currentScale === 'exponential') {
      if (currentValue < 0.01) return currentValue.toFixed(4);
      if (currentValue < 1) return currentValue.toFixed(3);
      if (currentValue < 10) return currentValue.toFixed(2);
      if (currentValue < 100) return currentValue.toFixed(1);
      return currentValue.toFixed(0);
    }
    return currentStep && currentStep < 1 ? currentValue.toFixed(4) : currentValue.toFixed(0);
  }

  const useFixedValues = $derived(Boolean(fixedValues && fixedValues.length > 0));

  function findNearestIndex(currentValue: number): number {
    if (!useFixedValues) return 0;
    let nearestIndex = 0;
    let minimumDifference = Math.abs(fixedValues![0] - currentValue);
    for (let index = 1; index < fixedValues!.length; index += 1) {
      const difference = Math.abs(fixedValues![index] - currentValue);
      if (difference < minimumDifference) {
        minimumDifference = difference;
        nearestIndex = index;
      }
    }
    return nearestIndex;
  }

  // Match React's mount-time useState initializer: later prop updates do not
  // overwrite an in-progress local interaction.
  let localValue = $state(untrack(() => (
    useFixedValues
      ? findNearestIndex(value)
      : scale === 'exponential'
        ? valueToPosition(value, minimumValue, maximumValue, scale)
        : value
  )));
  let isEditing = $state(false);
  let inputValue = $state('');
  let valueInput = $state<HTMLInputElement | null>(null);
  let interactionTooltipVisible = $state(false);
  const tooltipId = `slider-tooltip-${++nextSliderId}`;

  $effect(() => {
    if (isEditing && valueInput) valueInput.focus();
  });

  const shouldShowMinMax = $derived(showMinMax !== undefined ? showMinMax : scale !== 'exponential');
  const displayValue = $derived(
    useFixedValues
      ? fixedValues![localValue]
      : scale === 'exponential'
        ? positionToValue(localValue, minimumValue, maximumValue, scale)
        : localValue,
  );
  const formattedValue = $derived(
    formatValue ? formatValue(displayValue) : formatParamValue(displayValue, scale, step),
  );
  const sliderMin = $derived(
    useFixedValues ? 0 : scale === 'exponential' ? 0 : minimumValue,
  );
  const sliderMax = $derived(
    useFixedValues
      ? fixedValues!.length - 1
      : scale === 'exponential'
        ? 1
        : maximumValue,
  );
  const sliderStep = $derived(
    useFixedValues ? 1 : scale === 'exponential' ? 0.001 : step,
  );
  const sliderPositionPercent = $derived(
    sliderMax === sliderMin ? 0 : ((localValue - sliderMin) / (sliderMax - sliderMin)) * 100,
  );
  const sliderValueText = $derived(
    formatValue
      ? formatValue(
          useFixedValues
            ? fixedValues![localValue]
            : scale === 'exponential'
              ? positionToValue(localValue, minimumValue, maximumValue, scale)
              : localValue,
        )
      : formatParamValue(
          useFixedValues
            ? fixedValues![localValue]
            : scale === 'exponential'
              ? positionToValue(localValue, minimumValue, maximumValue, scale)
              : localValue,
          scale,
          step,
        ),
  );

  function handleChange(event: Event): void {
    const newPosition = Number((event.currentTarget as HTMLInputElement).value);
    localValue = newPosition;
    const finalValue = useFixedValues
      ? fixedValues![newPosition]
      : scale === 'exponential'
        ? positionToValue(newPosition, minimumValue, maximumValue, scale)
        : newPosition;
    onValueChange(finalValue);
  }

  function handleValueClick(): void {
    if (!disabled && editable) {
      inputValue = formattedValue;
      isEditing = true;
    }
  }

  function handleInputChange(event: Event): void {
    inputValue = (event.currentTarget as HTMLInputElement).value;
  }

  function handleInputSubmit(): void {
    const numericValue = Number.parseFloat(inputValue);
    if (!Number.isNaN(numericValue)) {
      if (useFixedValues) {
        const nearestIndex = findNearestIndex(numericValue);
        localValue = nearestIndex;
        onValueChange(fixedValues![nearestIndex]);
      } else {
        const clampedValue = Math.max(minimumValue, Math.min(maximumValue, numericValue));
        const newPosition =
          scale === 'exponential'
            ? valueToPosition(clampedValue, minimumValue, maximumValue, scale)
            : clampedValue;
        localValue = newPosition;
        onValueChange(clampedValue);
      }
    }
    isEditing = false;
  }

  function handleInputKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') handleInputSubmit();
    if (event.key === 'Escape') isEditing = false;
  }
</script>

<div class="slider-stack">
  <div class="slider-label-row">
    <span class="slider-label">{label}</span>
    {#if isEditing}
      <input
        bind:this={valueInput}
        class="slider-value-input"
        type="text"
        inputmode="decimal"
        value={inputValue}
        oninput={handleInputChange}
        onblur={handleInputSubmit}
        onkeydown={handleInputKeydown}
        disabled={disabled}
        aria-label={`${label} value`}
      />
    {:else if editable}
      <button
        class:editable
        class="slider-value"
        type="button"
        onclick={handleValueClick}
        disabled={disabled}
        aria-label={`${label}: ${formattedValue}`}
      >
        {formattedValue}
      </button>
    {:else}
      <span class="slider-value" class:disabled aria-label={`${label}: ${formattedValue}`}>
        {formattedValue}
      </span>
    {/if}
  </div>

  <div
    class="slider-control-wrap"
    role="group"
    aria-label={`${label} slider control`}
    onmouseenter={() => { interactionTooltipVisible = true; }}
    onmouseleave={() => { interactionTooltipVisible = false; }}
  >
    {#if interactionTooltipVisible}
      <span
        id={tooltipId}
        class="slider-tooltip"
        role="tooltip"
        style:left={`${Math.max(0, Math.min(100, sliderPositionPercent))}%`}
      >{sliderValueText}</span>
    {/if}
    <input
      class="slider-control"
      type="range"
      min={sliderMin}
      max={sliderMax}
      step={sliderStep}
      value={localValue}
      oninput={handleChange}
      onfocus={() => { interactionTooltipVisible = true; }}
      onblur={() => { interactionTooltipVisible = false; }}
      disabled={disabled}
      aria-label={accessibilityLabel || label}
      aria-valuetext={sliderValueText}
      aria-describedby={interactionTooltipVisible ? tooltipId : undefined}
      title={sliderValueText}
      data-show-tooltip={showTooltip}
    />
  </div>

  {#if shouldShowMinMax}
    <div class="slider-min-max">
      <span>
        {useFixedValues
          ? formatValue
            ? formatValue(fixedValues![0])
            : String(fixedValues![0])
          : formatValue
            ? formatValue(minimumValue)
            : String(minimumValue)}
      </span>
      <span>
        {useFixedValues
          ? formatValue
            ? formatValue(fixedValues![fixedValues!.length - 1])
            : String(fixedValues![fixedValues!.length - 1])
          : formatValue
            ? formatValue(maximumValue)
            : String(maximumValue)}
      </span>
    </div>
  {/if}

  {#if helpText}
    <span class="slider-help-text">{helpText}</span>
  {/if}
</div>

<style>
  .slider-stack {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .slider-label-row,
  .slider-min-max {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .slider-label,
  .slider-value {
    font-size: 0.875rem;
    font-weight: 600;
  }

  .slider-value {
    border: 0;
    padding: 0;
    color: var(--slider-value-color, #228be6);
    background: transparent;
    cursor: default;
    font-family: inherit;
  }

  .slider-value.editable {
    cursor: pointer;
    text-decoration: underline dotted;
  }

  .slider-value.disabled,
  .slider-value:disabled {
    cursor: not-allowed;
  }

  .slider-value:disabled {
    color: var(--slider-value-color, #228be6);
    opacity: 1;
  }

  .slider-value-input {
    width: 100px;
    box-sizing: border-box;
    padding: 0.25rem 0.5rem;
    text-align: right;
    font: inherit;
    font-weight: 600;
  }

  .slider-control-wrap {
    position: relative;
    width: 100%;
  }

  .slider-control {
    width: 100%;
    accent-color: var(--slider-accent-color, #228be6);
  }

  .slider-tooltip {
    position: absolute;
    bottom: calc(100% + 4px);
    z-index: 1;
    padding: 3px 6px;
    border-radius: 4px;
    color: white;
    background: var(--mantine-color-dark-5, #2c2e33);
    font-size: 0.75rem;
    line-height: 1.2;
    pointer-events: none;
    transform: translateX(-50%);
    white-space: nowrap;
  }

  .slider-min-max,
  .slider-help-text {
    color: var(--slider-muted-color, #868e96);
  }

  .slider-min-max {
    font-size: 0.75rem;
  }

  .slider-help-text {
    font-size: 0.875rem;
  }
</style>
