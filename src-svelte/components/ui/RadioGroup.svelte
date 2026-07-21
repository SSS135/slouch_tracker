<script lang="ts">
  export interface RadioOption {
    value: string;
    label: string;
    description?: string;
    badge?: string;
    badgeColor?: string;
    disabled?: boolean;
  }

  export interface RadioGroupProps {
    options: RadioOption[];
    value: string;
    onChange: (value: string) => void;
    name: string;
    disabled?: boolean;
    variant?: 'default' | 'compact';
  }

  let {
    options,
    value,
    onChange,
    name,
    disabled = false,
    variant = 'default',
  }: RadioGroupProps = $props();

  let isCompact = $derived(variant === 'compact');
</script>

<div class="radio-group" role="radiogroup" aria-label={name}>
  <div class:compact={isCompact} class="radio-options">
    {#each options as option (option.value)}
      {@const isOptionDisabled = disabled || option.disabled === true}
      {@const isSelected = value === option.value}
      {@const inputId = `${name}-${option.value}`}
      {@const descriptionId = `${inputId}-description`}
      <div
        class:selected={isSelected}
        class:disabled={isOptionDisabled}
        class="radio-option"
        style:opacity={isOptionDisabled ? 0.5 : 1}
      >
        <label class="radio-label" for={inputId}>
          <input
            id={inputId}
            class="radio-input"
            type="radio"
            name={name}
            value={option.value}
            checked={isSelected}
            disabled={isOptionDisabled}
            aria-describedby={option.description && !isCompact ? descriptionId : undefined}
            onchange={(event: Event) => {
              const input = event.currentTarget as HTMLInputElement;
              if (input.checked && !isOptionDisabled) {
                onChange(input.value);
              }
            }}
          />
          <span class="radio-content">
            <span class="radio-heading">
              <span class:compact-text={isCompact} class="radio-option-label">{option.label}</span>
              {#if option.badge}
                <span
                  class="badge"
                  style={`--badge-color: ${option.badgeColor || 'green'}`}
                >
                  {option.badge}
                </span>
              {/if}
            </span>
            {#if option.description && !isCompact}
              <span id={descriptionId} class="radio-description">{option.description}</span>
            {/if}
          </span>
        </label>
      </div>
    {/each}
  </div>
</div>

<style>
  .radio-group {
    width: 100%;
  }

  .radio-options {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-sm, 12px);
  }

  .radio-options.compact {
    gap: var(--mantine-spacing-xs, 10px);
  }

  .radio-option {
    width: 100%;
    box-sizing: border-box;
    cursor: pointer;
    border: 2px solid var(--mantine-color-dark-4, #495057);
    border-radius: 6px;
    background: var(--mantine-color-dark-8, #212529);
    transition: border-color 120ms ease, background-color 120ms ease;
  }

  .radio-option.selected {
    border-color: var(--mantine-color-blue-6, #228be6);
    background: var(--mantine-color-dark-7, #343a40);
  }

  .radio-option.disabled {
    cursor: not-allowed;
  }

  .radio-label {
    display: flex;
    align-items: flex-start;
    width: 100%;
    box-sizing: border-box;
    padding: 12px;
    cursor: inherit;
  }

  .radio-options.compact .radio-label {
    padding: 8px;
  }

  .radio-input {
    flex: 0 0 auto;
    width: 20px;
    height: 20px;
    margin: 2px 0 0;
    accent-color: var(--mantine-color-blue-6, #228be6);
  }

  .radio-content {
    display: flex;
    flex: 1;
    min-width: 0;
    flex-direction: column;
    gap: 4px;
    padding-left: 8px;
  }

  .radio-options.compact .radio-content {
    gap: 2px;
  }

  .radio-heading {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }

  .radio-option-label {
    color: inherit;
    font-size: 1rem;
    font-weight: 500;
    line-height: 1.55;
  }

  .radio-option.selected .radio-option-label {
    font-weight: 600;
  }

  .radio-option-label.compact-text {
    font-size: 0.875rem;
    line-height: 1.45;
  }

  .radio-description {
    color: var(--mantine-color-dimmed, #909296);
    font-size: 0.75rem;
    line-height: 1.4;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    border-radius: 4px;
    background: color-mix(in srgb, var(--badge-color) 18%, transparent);
    color: var(--badge-color);
    font-size: 0.625rem;
    font-weight: 600;
    line-height: 1.4;
    padding: 2px 6px;
    white-space: nowrap;
  }
</style>
