<script lang="ts">
  import { onMount } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import AppProviders from './providers/AppProviders.svelte';
  import PostureTrackerApp from './pages/PostureTrackerApp.svelte';
  import {
    NOTIFICATION_EVENT,
    type NotificationEventDetail,
  } from './hooks/useNotification';

  interface AppNotification extends NotificationEventDetail {
    id: number;
  }

  let notifications = $state<AppNotification[]>([]);
  let nextNotificationId = 0;
  const timers = new SvelteMap<number, ReturnType<typeof setTimeout>>();

  function notificationColor(color: NotificationEventDetail['color']): string {
    switch (color) {
      case 'teal':
        return '#12b886';
      case 'red':
        return '#fa5252';
      case 'yellow':
        return '#fab005';
      case 'blue':
        return '#228be6';
    }
  }

  function scheduleVisibleNotifications(): void {
    for (const notification of notifications.slice(0, 5)) {
      if (!timers.has(notification.id)) {
        timers.set(
          notification.id,
          setTimeout(() => dismissNotification(notification.id), notification.autoClose),
        );
      }
    }
  }

  function dismissNotification(id: number): void {
    const notification = notifications.find((item) => item.id === id);
    if (!notification) return;

    const timer = timers.get(id);
    if (timer) clearTimeout(timer);
    timers.delete(id);
    notifications = notifications.filter((item) => item.id !== id);
    notification.onClose?.();
    scheduleVisibleNotifications();
  }

  function handleNotification(event: Event): void {
    const detail = (event as CustomEvent<NotificationEventDetail>).detail;
    if (!detail) {
      return;
    }

    const id = ++nextNotificationId;
    notifications = [...notifications, { ...detail, id }];
    scheduleVisibleNotifications();
  }

  onMount(() => {
    window.addEventListener(NOTIFICATION_EVENT, handleNotification);

    return () => {
      window.removeEventListener(NOTIFICATION_EVENT, handleNotification);
      for (const timer of timers.values()) {
        clearTimeout(timer);
      }
      timers.clear();
    };
  });
</script>

<div class="app-shell">
  <main class="app-main">
    <AppProviders>
      <PostureTrackerApp />
    </AppProviders>
  </main>
</div>

<div class="notifications" aria-live="polite" aria-label="Notifications">
  {#each notifications.slice(0, 5) as notification (notification.id)}
    <div
      class="notification"
      role={notification.color === 'red' ? 'alert' : 'status'}
      style={`--notification-color: ${notificationColor(notification.color)};`}
    >
      <div class="notification-content">
        <strong>{notification.title}</strong>
        <span>{notification.message}</span>
      </div>
      <button
        type="button"
        class="notification-dismiss"
        aria-label="Dismiss notification"
        onclick={() => dismissNotification(notification.id)}
      >
        ×
      </button>
    </div>
  {/each}
</div>

<style>
  :global(*) {
    box-sizing: border-box;
  }

  :global(html),
  :global(body),
  :global(#root) {
    height: 100%;
    margin: 0;
    overflow: hidden;
  }

  :global(body) {
    background: #0a0a0a;
    color: #f8fafc;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .app-shell {
    height: 100%;
    background: #0a0a0a;
  }

  .app-main {
    width: 100%;
    height: 100%;
  }

  .notifications {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 1000;
    display: flex;
    width: min(24rem, calc(100vw - 2rem));
    flex-direction: column;
    gap: 0.75rem;
    pointer-events: none;
  }

  .notification {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    border: 1px solid var(--notification-color);
    border-left-width: 0.25rem;
    border-radius: 0.5rem;
    padding: 0.75rem 0.875rem;
    color: #f8fafc;
    background: #1f1f1f;
    box-shadow: 0 0.5rem 1.5rem rgb(0 0 0 / 35%);
    pointer-events: auto;
  }

  .notification-content {
    display: flex;
    min-width: 0;
    flex: 1;
    flex-direction: column;
    gap: 0.25rem;
    line-height: 1.35;
  }

  .notification-content span {
    color: #ced4da;
    overflow-wrap: anywhere;
  }

  .notification-dismiss {
    border: 0;
    padding: 0;
    color: #adb5bd;
    background: transparent;
    font: inherit;
    font-size: 1.25rem;
    line-height: 1;
    cursor: pointer;
  }

  .notification-dismiss:hover,
  .notification-dismiss:focus-visible {
    color: #fff;
  }
</style>
