import { IconCheckCircle, IconAlertTriangle } from "../icons.jsx";

// Fixed-position stack of auto-dismissing notifications, driven by App.jsx's
// `toasts` state + `notify()` helper (see App.jsx). Pure presentational component --
// no timers live here, App.jsx owns removal so a toast's lifetime survives this
// component re-rendering.
export default function ToastStack({ toasts, onDismiss }) {
  if (!toasts || toasts.length === 0) return null;

  return (
    <div className="toast-stack">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast-${toast.tone}`} onClick={() => onDismiss(toast.id)}>
          {toast.tone === "error" ? <IconAlertTriangle size={16} /> : <IconCheckCircle size={16} />}
          <span>{toast.message}</span>
        </div>
      ))}
    </div>
  );
}
