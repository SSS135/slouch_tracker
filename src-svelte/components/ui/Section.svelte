<script lang="ts">
  import type { Snippet } from 'svelte';

  interface SectionProps {
    title: string;
    subtitle?: string;
    children: Snippet;
    headerRight?: Snippet;
  }

  let { title, subtitle, children, headerRight }: SectionProps = $props();
</script>

<div class="section">
  <div class="stack">
    <div class="group">
      <div class="heading">
        <p class="title">{title}</p>
        {#if subtitle}
          <p class="subtitle">{subtitle}</p>
        {/if}
      </div>
      {#if headerRight}
        <div class="header-right">{@render headerRight()}</div>
      {/if}
    </div>
    {@render children()}
  </div>
</div>

<style>
  .section {
    box-sizing: border-box;
    border: 1px solid var(--mantine-color-default-border, #424242);
    border-radius: var(--mantine-radius-lg, 16px);
    padding: var(--mantine-spacing-lg, 20px);
    background: var(--mantine-color-dark-7, #242424);
  }

  .stack {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-sm, 12px);
  }

  .group {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--mantine-spacing-md, 16px);
  }

  .heading {
    min-width: 0;
  }

  .title,
  .subtitle {
    margin: 0;
    padding: 0;
  }

  .title {
    color: var(--mantine-color-text, #c1c2c5);
    font-size: var(--mantine-font-size-md, 1rem);
    line-height: var(--mantine-line-height-md, 1.55);
    font-weight: 700;
  }

  .subtitle {
    color: var(--mantine-color-dimmed, #828282);
    font-size: var(--mantine-font-size-sm, 0.875rem);
    line-height: var(--mantine-line-height-sm, 1.45);
  }

  .header-right {
    min-width: 0;
    flex-shrink: 1;
  }
</style>
