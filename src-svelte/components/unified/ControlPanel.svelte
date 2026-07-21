<script module lang="ts">
  let nextControlPanelId = 0;
</script>

<script lang="ts">
  import type { Snippet } from 'svelte';

  export type TabType = 'runtime' | 'training';

  export interface TabDescriptor {
    id: TabType;
    label: string;
    disabled?: boolean;
    content: Snippet;
  }

  export interface ControlPanelProps {
    activeTab: TabType;
    onTabChange: (tab: TabType) => void;
    tabs: TabDescriptor[];
    collapsed?: boolean;
  }

  let { activeTab, onTabChange, tabs, collapsed = false }: ControlPanelProps = $props();
  const instanceId = `control-panel-${++nextControlPanelId}`;

  function panelId(tab: TabDescriptor): string {
    return `${instanceId}-${tab.id}`;
  }

  function tabId(tab: TabDescriptor): string {
    return `${panelId(tab)}-tab`;
  }

  function handleTabKeydown(event: KeyboardEvent, current: TabDescriptor): void {
    const enabledTabs = tabs.filter((tab) => !tab.disabled);
    const currentIndex = enabledTabs.findIndex((tab) => tab.id === current.id);
    let nextIndex: number | null = null;
    if (event.key === 'ArrowRight') nextIndex = (currentIndex + 1) % enabledTabs.length;
    else if (event.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + enabledTabs.length) % enabledTabs.length;
    else if (event.key === 'Home') nextIndex = 0;
    else if (event.key === 'End') nextIndex = enabledTabs.length - 1;
    if (nextIndex === null || !enabledTabs[nextIndex]) return;
    event.preventDefault();
    const next = enabledTabs[nextIndex];
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      onTabChange(next.id);
    }
    document.getElementById(tabId(next))?.focus();
  }
</script>

<div class:collapsed class="control-panel">
  <div class="tabs" role="tablist" aria-label="Control panel tabs" aria-orientation="horizontal">
    {#each tabs as tab (tab.id)}
      <button
        id={tabId(tab)}
        class:active={activeTab === tab.id}
        type="button"
        role="tab"
        aria-selected={activeTab === tab.id}
        aria-controls={panelId(tab)}
        tabindex={activeTab === tab.id ? 0 : -1}
        disabled={tab.disabled}
        onclick={() => onTabChange(tab.id)}
        onkeydown={(event) => handleTabKeydown(event, tab)}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  {#each tabs as tab (tab.id)}
    <div
      id={panelId(tab)}
      class="tab-panel"
      role="tabpanel"
      aria-labelledby={tabId(tab)}
      hidden={activeTab !== tab.id}
    >
      {#if activeTab === tab.id}
        <div class="scroll-area">
          <div class="panel-content">
            {@render tab.content()}
          </div>
        </div>
      {/if}
    </div>
  {/each}
</div>

<style>
  .control-panel {
    height: 100%;
    box-sizing: border-box;
    transform: translateX(0);
    transition: transform 0.3s ease-in-out;
    background: rgb(10 10 10 / 50%);
    backdrop-filter: blur(12px);
    box-shadow: 0 0 30px rgb(0 0 0 / 50%);
    display: flex;
    flex-direction: column;
  }

  .control-panel.collapsed {
    transform: translateX(100%);
  }

  .tabs {
    display: flex;
    flex: 0 0 auto;
    justify-content: center;
    border-bottom: 1px solid rgb(255 255 255 / 12%);
  }

  .tabs button {
    flex: 0 0 auto;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 8px 16px;
    color: rgb(255 255 255 / 70%);
    background: transparent;
    font-family: inherit;
    font-size: 14px;
    line-height: 1;
    cursor: pointer;
  }

  .tabs button:hover:not(:disabled),
  .tabs button:focus-visible {
    color: rgb(255 255 255 / 92%);
    background: rgb(255 255 255 / 6%);
  }

  .tabs button.active {
    border-bottom-color: var(--mantine-color-blue-6, #228be6);
    color: rgb(255 255 255 / 100%);
  }

  .tabs button:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  .tab-panel {
    flex: 1;
    min-height: 0;
  }

  .tab-panel[hidden] {
    display: none;
  }

  .scroll-area {
    box-sizing: border-box;
    height: 100%;
    overflow: auto;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
  }

  .scroll-area::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }

  .scroll-area::-webkit-scrollbar-thumb {
    border-radius: 3px;
    background: rgb(255 255 255 / 30%);
  }

  .scroll-area::-webkit-scrollbar-track {
    background: transparent;
  }

  .panel-content {
    box-sizing: border-box;
    padding: 16px 6px 16px 12px;
  }

  :global(.panel-shell.collapsed) {
    transform: none !important;
  }
</style>
