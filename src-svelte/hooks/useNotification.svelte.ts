/**
 * Notification Hook
 *
 * Standardized notification system for the app.
 * Notifications are emitted as semantic DOM events so Svelte UI components
 * can render them without coupling this hook to a component library.
 */

export interface NotificationOptions {
  onDismiss?: () => void;
}

export interface ConfirmOptions {
  onConfirm?: () => void;
  onCancel?: () => void;
  confirmText?: string;
  cancelText?: string;
}

export type NotificationColor = 'teal' | 'red' | 'yellow' | 'blue';

export interface NotificationEventDetail {
  title: string;
  message: string;
  color: NotificationColor;
  withBorder: true;
  autoClose: number;
  onClose?: () => void;
}

export interface NotificationApi {
  showSuccess: (message: string, options?: NotificationOptions) => void;
  showError: (message: string, options?: NotificationOptions) => void;
  showWarning: (message: string, options?: NotificationOptions) => void;
  showInfo: (message: string, options?: NotificationOptions) => void;
  showConfirm: (
    title: string,
    message: string,
    options?: ConfirmOptions,
  ) => void;
}

export const NOTIFICATION_EVENT = 'slouch-tracker:notification';

function emitNotification(detail: NotificationEventDetail): void {
  if (typeof window === 'undefined') {
    return;
  }

  // Parent onMount handlers are installed later in the same mount turn than
  // child startup callbacks. Defer delivery until every mount callback in the
  // current turn has had a chance to register its sink.
  queueMicrotask(() => {
    window.dispatchEvent(
      new CustomEvent<NotificationEventDetail>(NOTIFICATION_EVENT, { detail }),
    );
  });
}

export function useNotification(): NotificationApi {
  const showSuccess = (
    message: string,
    options?: NotificationOptions,
  ): void => {
    emitNotification({
      title: 'Success',
      message,
      color: 'teal',
      withBorder: true,
      autoClose: 4000,
      onClose: options?.onDismiss,
    });
  };

  const showError = (
    message: string,
    options?: NotificationOptions,
  ): void => {
    emitNotification({
      title: 'Error',
      message,
      color: 'red',
      withBorder: true,
      autoClose: 6000,
      onClose: options?.onDismiss,
    });
  };

  const showWarning = (
    message: string,
    options?: NotificationOptions,
  ): void => {
    emitNotification({
      title: 'Warning',
      message,
      color: 'yellow',
      withBorder: true,
      autoClose: 5000,
      onClose: options?.onDismiss,
    });
  };

  const showInfo = (
    message: string,
    options?: NotificationOptions,
  ): void => {
    emitNotification({
      title: 'Info',
      message,
      color: 'blue',
      withBorder: true,
      autoClose: 4000,
      onClose: options?.onDismiss,
    });
  };

  const showConfirm = (
    title: string,
    message: string,
    options?: ConfirmOptions,
  ): void => {
    const result = window.confirm(`${title}\n\n${message}`);
    if (result) {
      options?.onConfirm?.();
    } else {
      options?.onCancel?.();
    }
  };

  return {
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showConfirm,
  };
}
