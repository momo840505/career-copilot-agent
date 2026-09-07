import { IconAlertTriangle, IconCheckCircle } from "../icons.jsx";

export default function ToastStack({ toasts, onDismiss }) {
  if (!toasts || toasts.length === 0) return null;

  return (
    <div className="toast-stack" aria-live="polite">
      {toasts.map((toast) => (
        <button
          key={toast.id}
          type="button"
          className={`toast toast-${toast.tone}`}
          onClick={() => onDismiss(toast.id)}
          aria-label={`Dismiss notification: ${toast.message}`}
        >
          {toast.tone === "error" ? (
            <IconAlertTriangle size={16} />
          ) : (
            <IconCheckCircle size={16} />
          )}
          <span>{toast.message}</span>
        </button>
      ))}
    </div>
  );
}
