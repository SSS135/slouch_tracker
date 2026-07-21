<script lang="ts">
  import type { Snippet } from 'svelte';
  import { onMount } from 'svelte';
  import { QueryClientProvider } from '@tanstack/svelte-query';
  import { TrainingConfigProvider } from '../contexts/TrainingConfigContext';
  import { TrainingProvider } from '../contexts/TrainingContext';
  import { createAppQueryClient } from '../lib/query/client';
  import { provideNativeAppState } from '../lib/state/nativeApp.svelte';

  interface AppProvidersProps {
    children: Snippet;
  }

  let { children }: AppProvidersProps = $props();

  const queryClient = createAppQueryClient();
  const nativeApp = provideNativeAppState();

  TrainingConfigProvider();
  TrainingProvider();

  onMount(() => {
    void nativeApp.initialize().catch(() => undefined);

    return () => {
      queryClient.clear();
    };
  });
</script>

<QueryClientProvider client={queryClient}>
  {@render children()}
</QueryClientProvider>
