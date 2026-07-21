<script lang="ts">
  import { logger } from '@/services/logging';

  interface LoggerOption {
    value: string;
    label: string;
    description: string;
    badge?: string;
    badgeColor?: string;
  }

  const LOGGER_OPTIONS: LoggerOption[] = [
    {
      value: 'none',
      label: 'Error',
      description: 'Only warnings and errors reach the console.',
      badge: 'Default',
      badgeColor: 'green',
    },
    {
      value: 'all',
      label: 'Info',
      description: 'Enable INFO level logging for all categories.',
    },
    {
      value: 'debug',
      label: 'Debug',
      description: 'Enable detailed DEBUG output across all categories.',
    },
  ];

  function readLogParam(): string {
    if (typeof window === 'undefined') {
      return 'none';
    }

    return new URLSearchParams(window.location.search).get('log') ?? 'none';
  }

  let currentValue = $state(readLogParam());

  $effect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handlePopState = (): void => {
      const nextValue = readLogParam();
      currentValue = nextValue;
      logger.setFromURLParam(nextValue);
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  });

  function handleChange(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    if (!input.checked) return;
    const value = input.value;
    currentValue = value;

    const url = new URL(window.location.href);
    if (value === 'none') {
      url.searchParams.delete('log');
    } else {
      url.searchParams.set('log', value);
    }
    window.history.replaceState({}, '', url.toString());
    logger.setFromURLParam(value);

    window.dispatchEvent(
      new CustomEvent('logLevelChanged', {
        detail: { logParam: value === 'none' ? '' : value },
      }),
    );
  }

  function badgeColor(color: string | undefined): string {
    if (color === 'green') return 'var(--mantine-color-green-6, #40c057)';
    return color ?? 'var(--mantine-color-gray-6, #868e96)';
  }
</script>

<div class="logger-options" role="radiogroup" aria-label="logger-level">
  {#each LOGGER_OPTIONS as option (option.value)}
    {@const inputId = `logger-level-${option.value}`}
    <label class="logger-option" for={inputId}>
      <input
        id={inputId}
        type="radio"
        name="logger-level"
        value={option.value}
        checked={currentValue === option.value}
        onchange={handleChange}
      />
      <span class="option-copy">
        <span class="option-heading">
          <span class="option-label">{option.label}</span>
          {#if option.badge}
            <span class="badge" style:--badge-color={badgeColor(option.badgeColor)}>{option.badge}</span>
          {/if}
        </span>
        <span class="description">{option.description}</span>
      </span>
    </label>
  {/each}
</div>

<style>
  .logger-options {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-sm, 12px);
  }

  .logger-option {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    cursor: pointer;
  }

  .logger-option input {
    width: 20px;
    height: 20px;
    margin: 2px 0 0;
    accent-color: var(--mantine-color-blue-6, #228be6);
  }

  .option-copy {
    display: flex;
    min-width: 0;
    flex: 1;
    flex-direction: column;
    gap: 4px;
  }

  .option-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--mantine-spacing-xs, 10px);
  }

  .option-label {
    font-size: var(--mantine-font-size-md, 1rem);
    font-weight: 600;
    line-height: 1.55;
  }

  .description {
    color: var(--mantine-color-dimmed, #909296);
    font-size: var(--mantine-font-size-sm, 0.875rem);
    line-height: 1.45;
  }

  .badge {
    padding: 3px 8px;
    border-radius: 4px;
    color: var(--badge-color);
    background: color-mix(in srgb, var(--badge-color) 15%, transparent);
    font-size: var(--mantine-font-size-sm, 0.875rem);
    font-weight: 600;
    line-height: 1.45;
  }
</style>
