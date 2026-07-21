<script lang="ts">
  import { onMount } from 'svelte';
  import { useDatasetOperations, type UseDatasetOperationsResult } from '../useDatasetOperations.svelte';

  interface Props {
    onReady: (result: UseDatasetOperationsResult) => void;
  }

  let { onReady }: Props = $props();
  const result = useDatasetOperations();

  onMount(() => onReady(result));
</script>
