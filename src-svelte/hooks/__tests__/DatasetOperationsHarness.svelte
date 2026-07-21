<script lang="ts">
  import type { QueryClient } from '@tanstack/svelte-query';
  import { QueryClientProvider } from '@tanstack/svelte-query';
  import DatasetOperationsConsumer from './DatasetOperationsConsumer.svelte';
  import type { UseDatasetOperationsResult } from '../useDatasetOperations.svelte';

  interface Props {
    client: QueryClient;
    onReady: (result: UseDatasetOperationsResult) => void;
  }

  let { client, onReady }: Props = $props();
</script>

<QueryClientProvider {client}>
  <DatasetOperationsConsumer {onReady} />
</QueryClientProvider>
