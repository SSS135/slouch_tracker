<script lang="ts">
  import {
    TrainingConfigProvider,
    useTrainingConfig,
    type TrainingConfigContextValue,
  } from '../TrainingConfigContext.svelte';

  interface Props {
    provide?: boolean;
    onReady?: (value: TrainingConfigContextValue) => void;
    onError?: (error: Error) => void;
  }

  let { provide = true, onReady, onError }: Props = $props();

  function initialize(): void {
    try {
      const value = provide ? TrainingConfigProvider() : useTrainingConfig();
      onReady?.(value);
    } catch (cause) {
      onError?.(cause instanceof Error ? cause : new Error(String(cause)));
    }
  }

  initialize();
</script>
