<script lang="ts">
  import { NOTIFICATION_EVENT } from '@/hooks/useNotification';
  import type { ClassifierConfig, ClassifierId, ParameterDefinition_Serialize, ParameterValue } from '@generated/bindings';
  import { coerceParamValue, defaultParams, useTrainingConfig } from '@/contexts/TrainingConfigContext';
  import RadioGroup from '../ui/RadioGroup.svelte';
  import Slider from '../ui/Slider.svelte';

  export interface ClassifierSelectorProps {
    config: ClassifierConfig;
    onChange: (config: ClassifierConfig) => void;
    disabled?: boolean;
  }

  let {
    config,
    onChange,
    disabled = false,
  }: ClassifierSelectorProps = $props();

  const trainingConfig = useTrainingConfig();
  const currentDef = $derived(trainingConfig.classifiers.find((definition) => definition.id === config.classifierId));

  const visibleParams = $derived.by((): Array<[string, ParameterDefinition_Serialize]> => {
    if (!currentDef) {
      return [];
    }

    return Object.entries(currentDef.params).filter(([, paramDef]) => {
      if (!paramDef.showWhen) {
        return true;
      }

      return Object.entries(paramDef.showWhen).every(([key, allowedValues]) => {
        const paramValue = config.params[key];
        if (Array.isArray(allowedValues)) {
          return allowedValues.includes(paramValue);
        }
        return paramValue === allowedValues;
      });
    });
  });

  $effect(() => {
    if (trainingConfig.classifiers.length > 0 && !currentDef && !disabled) {
      const defaultId = 'mlp';
      showClassifierResetNotification();
      const definition = trainingConfig.classifiers.find((entry) => entry.id === defaultId);
      onChange({ classifierId: defaultId, params: defaultParams(definition) });
    }
  });

  function showClassifierResetNotification(): void {
    if (typeof window === 'undefined') {
      return;
    }

    window.dispatchEvent(
      new CustomEvent(NOTIFICATION_EVENT, {
        detail: {
          title: 'Classifier Reset',
          message: 'Previously saved classifier is no longer available. Reset to MLP.',
          color: 'yellow',
          withBorder: true,
          autoClose: 5000,
        },
      }),
    );
  }

  function handleClassifierChange(classifierId: string): void {
    if (disabled) {
      return;
    }

    const definition = trainingConfig.classifiers.find((entry) => entry.id === classifierId);
    if (!definition) return;
    onChange({ classifierId: classifierId as ClassifierId, params: defaultParams(definition) });
  }

  function handleParamChange(paramName: string, value: ParameterValue): void {
    if (disabled) {
      return;
    }

    // Coerce integer-typed params (e.g. maxIterations, k) so slider float drift never reaches the
    // native backend, whose settings/model schemas type these fields as unsigned integers.
    const coerced = coerceParamValue(currentDef?.params[paramName], value);
    onChange({
      ...config,
      params: { ...config.params, [paramName]: coerced },
    });
  }

  function getParamValue(name: string, definition: ParameterDefinition_Serialize): ParameterValue {
    return config.params[name] ?? definition.default;
  }

  function getSelectOptions(definition: ParameterDefinition_Serialize): Array<{ value: string; label: string }> {
    return (definition.options ?? []).map((option) => ({
      value: String(option.value),
      label: option.label,
    }));
  }

  function handleSelectChange(
    name: string,
    definition: ParameterDefinition_Serialize,
    selectedValue: string,
  ): void {
    const option = definition.options?.find(
      (candidate) => String(candidate.value) === selectedValue,
    );
    if (option) handleParamChange(name, option.value);
  }
</script>

{#if !currentDef}
  <p class="loading" aria-live="polite">Loading...</p>
{:else}
  <div class="selector-stack">
    <section class="control-stack">
      <h3>Algorithm:</h3>
      <RadioGroup
        name="classifier-algorithm"
        value={config.classifierId}
        onChange={handleClassifierChange}
        disabled={disabled}
        options={trainingConfig.classifiers.map((definition) => ({
          value: definition.id,
          label: definition.name,
          description: definition.description,
          badge: definition.id === 'mlp' ? 'Recommended' : undefined,
          badgeColor: definition.id === 'mlp' ? 'green' : undefined,
        }))}
      />
    </section>

    {#if Object.keys(currentDef.params).length > 0}
      <section class="control-stack">
        <h3>Parameters:</h3>
        <div class="parameter-stack">
          {#each visibleParams as [name, definition] (name)}
            {@const value = getParamValue(name, definition)}
            {#if definition.type === 'range'}
              <Slider
                label={definition.label}
                value={typeof value === 'number' ? value : Number(value) || 0}
                minimumValue={definition.min ?? 0}
                maximumValue={definition.max ?? 100}
                step={definition.step ?? 1}
                onValueChange={(nextValue) => handleParamChange(name, nextValue)}
                disabled={disabled}
                scale={definition.scale ?? undefined}
                helpText={definition.description ?? undefined}
                editable
                showMinMax
              />
            {:else if definition.type === 'select'}
              <div class="parameter-control">
                <h4>{definition.label}</h4>
                {#if definition.description}
                  <p class="description">{definition.description}</p>
                {/if}
                <RadioGroup
                  name={`classifier-param-${name}`}
                  value={String(value)}
                  options={getSelectOptions(definition)}
                  onChange={(selectedValue) => handleSelectChange(name, definition, selectedValue)}
                  disabled={disabled}
                  variant="compact"
                />
              </div>
            {:else if definition.type === 'number'}
              <div class="parameter-control">
                <h4>{definition.label}</h4>
                {#if definition.description}
                  <p class="description">{definition.description}</p>
                {/if}
                <input
                  class="number-input"
                  type="number"
                  value={value}
                  min={definition.min}
                  max={definition.max}
                  step={definition.step}
                  disabled={disabled}
                  aria-label={definition.label}
                  oninput={(event) => {
                    const rawValue = (event.currentTarget as HTMLInputElement).value;
                    if (rawValue.trim() === '') return;
                    const nextValue = Number(rawValue);
                    if (!Number.isFinite(nextValue)) return;
                    if (definition.min !== null && definition.min !== undefined && nextValue < definition.min) return;
                    if (definition.max !== null && definition.max !== undefined && nextValue > definition.max) return;
                    handleParamChange(name, nextValue);
                  }}
                />
              </div>
            {:else if definition.type === 'boolean'}
              <label class="boolean-control">
                <span class="boolean-copy">
                  <span class="parameter-label">{definition.label}</span>
                  {#if definition.description}
                    <span class="description">{definition.description}</span>
                  {/if}
                </span>
                <input
                  type="checkbox"
                  checked={Boolean(value)}
                  disabled={disabled}
                  aria-label={definition.label}
                  onchange={(event) =>
                    handleParamChange(name, (event.currentTarget as HTMLInputElement).checked)}
                />
              </label>
            {/if}
          {/each}
        </div>
      </section>
    {/if}

    <p class="help-text">Changing classifier or parameters requires retraining the model</p>
  </div>
{/if}

<style>
  .selector-stack,
  .control-stack,
  .parameter-stack,
  .parameter-control,
  .boolean-copy {
    display: flex;
    flex-direction: column;
  }

  .selector-stack {
    gap: 1rem;
  }

  .control-stack {
    gap: 0.5rem;
  }

  .parameter-stack {
    gap: 1rem;
  }

  h3,
  h4,
  p {
    margin: 0;
  }

  h3 {
    color: var(--mantine-color-text, #f8f9fa);
    font-size: 0.875rem;
    font-weight: 500;
  }

  h4,
  .parameter-label {
    color: var(--mantine-color-text, #f8f9fa);
    font-size: 0.875rem;
    font-weight: 500;
  }

  .parameter-control {
    gap: 0.5rem;
  }

  .description,
  .help-text,
  .loading {
    color: var(--mantine-color-dimmed, #909296);
    font-size: 0.75rem;
    line-height: 1.4;
  }

  .boolean-control {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .boolean-copy {
    flex: 1;
    gap: 0.25rem;
  }

  input[type='checkbox'] {
    width: 1rem;
    height: 1rem;
    accent-color: var(--mantine-color-blue-6, #228be6);
  }

  .number-input {
    box-sizing: border-box;
    width: 100%;
    min-height: 2.25rem;
    padding: 0.375rem 0.625rem;
    border: 1px solid var(--mantine-color-default-border, #373a40);
    border-radius: 0.25rem;
    color: var(--mantine-color-text, #f8f9fa);
    background: var(--mantine-color-dark-7, #343a40);
    font: inherit;
  }

  .number-input:disabled,
  input[type='checkbox']:disabled {
    cursor: not-allowed;
    opacity: 0.65;
  }

  .help-text {
    font-style: italic;
  }

  .loading {
    color: var(--mantine-color-dimmed, #909296);
  }
</style>
