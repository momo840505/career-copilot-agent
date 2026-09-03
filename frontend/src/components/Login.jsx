import { useState } from "react";
import { verifyAccessCode, setStoredAccessCode, ApiError } from "../api.js";
import { IconBriefcase, IconAlertTriangle } from "../icons.jsx";

// Shown when the backend has ACCESS_CODE configured and either no code or the wrong
// code is stored locally. If the backend has no ACCESS_CODE set at all (local dev),
// App.jsx's mount-time verifyAccessCode("") call already succeeds and this screen
// never renders.
export default function Login({ onSuccess }) {
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setChecking(true);
    try {
      await verifyAccessCode(code);
      setStoredAccessCode(code);
      onSuccess();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError("Incorrect access code.");
      } else {
        setError("Could not reach the server. Is the backend running?");
      }
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="login-screen">
      <div className="bg-decor" aria-hidden="true">
        <span className="blob blob-a" />
        <span className="blob blob-b" />
      </div>
      <form className="login-card" onSubmit={handleSubmit}>
        <span className="brand-mark brand-mark-lg">
          <IconBriefcase size={24} />
        </span>
        <h1>Career Copilot</h1>
        <p>Enter the access code to continue.</p>
        <input
          type="password"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          placeholder="Access code"
          autoFocus
        />
        {error && (
          <p className="login-error">
            <IconAlertTriangle size={14} /> {error}
          </p>
        )}
        <button type="submit" disabled={checking}>
          {checking ? <span className="spinner spinner-light" /> : "Enter"}
        </button>
      </form>
    </div>
  );
}
