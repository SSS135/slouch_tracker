<script lang="ts">
  interface HelpTextProps {
    text: string;
  }

  let { text }: HelpTextProps = $props();
</script>

<p class="help-text">{text}</p>

<style>
  .help-text {
    margin: 0;
    padding: 0;
    color: var(--mantine-color-dimmed, #909296);
    font-size: var(--mantine-font-size-sm, 0.875rem);
    line-height: var(--mantine-line-height-sm, 1.45);
    font-weight: normal;
    text-decoration: none;
    -webkit-tap-highlight-color: transparent;
  }
</style>
