import { useEffect, useRef, useState } from "react";
import { verifyAccessCode, getStoredAccessCode, clearStoredAccessCode } from "./api.js";
import Login from "./components/Login.jsx";
import JDForm from "./components/JDForm.jsx";
import GapAnalysisResult from "./components/GapAnalysisResult.jsx";
import DraftResult from "./components/DraftResult.jsx";
import HistoryList from "./components/HistoryList.jsx";
import ToastStack from "./components/ToastStack.jsx";
import { IconBriefcase, IconFileText, IconClock, IconLogOut, IconSun, IconMoon } from "./icons.jsx";

const THEME_KEY = "career_copilot_theme";

export default function App() {
  // "checking" | "authed" | "needs-login"
  const [authStatus, setAuthStatus] = useState("checking");
  const [tab, setTab] = useState("new"); // "new" | "history"
  const [result, setResult] = useState(null); // { kind, data } | null
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || "light");
  const [toasts, setToasts] = useState([]);
  const toastId = useRef(0);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    // Works whether or not the backend has ACCESS_CODE set at all: with no code
    // configured, require_access_code is a no-op and this succeeds even with an
    // empty stored code (see api/auth.py + api.js's verifyAccessCode).
    verifyAccessCode(getStoredAccessCode())
      .then(() => setAuthStatus("authed"))
      .catch(() => setAuthStatus("needs-login"));
  }, []);

  function notify(message, tone = "success") {
    const id = ++toastId.current;
    setToasts((prev) => [...prev, { id, message, tone }]);
    setTimeout(() => dismissToast(id), 3200);
  }

  function dismissToast(id) {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }

  if (authStatus === "checking") {
    return (
      <div className="app-loading">
        <div className="spinner spinner-lg" />
      </div>
    );
  }

  if (authStatus === "needs-login") {
    return <Login onSuccess={() => setAuthStatus("authed")} />;
  }

  function handleLogout() {
    clearStoredAccessCode();
    setAuthStatus("needs-login");
  }

  return (
    <>
      <div className="bg-decor" aria-hidden="true">
        <span className="blob blob-a" />
        <span className="blob blob-b" />
        <span className="blob blob-c" />
      </div>

      <div className="app-shell">
        <header className="app-header">
          <div className="brand">
            <span className="brand-mark">
              <IconBriefcase size={20} />
            </span>
            <div>
              <h1 className="gradient-text">Career Copilot</h1>
              <p className="brand-tagline">Evidence-grounded gap analysis and cover letters</p>
            </div>
          </div>
          <nav>
            <button className={tab === "new" ? "tab active" : "tab"} onClick={() => setTab("new")}>
              <IconFileText size={15} /> New Analysis
            </button>
            <button className={tab === "history" ? "tab active" : "tab"} onClick={() => setTab("history")}>
              <IconClock size={15} /> History
            </button>
            <button
              className="tab theme-toggle"
              onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
              title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
              aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            >
              {theme === "light" ? <IconMoon size={16} /> : <IconSun size={16} />}
            </button>
            <button className="tab logout" onClick={handleLogout} title="Log out" aria-label="Log out">
              <IconLogOut size={16} />
            </button>
          </nav>
        </header>

        <main>
          <div className="tab-content" key={tab}>
            {tab === "new" && (
              <>
                <JDForm onResult={setResult} notify={notify} />
                {result && result.kind === "gap_analysis" && <GapAnalysisResult result={result.data} />}
                {result && result.kind === "draft" && (
                  <DraftResult
                    result={result.data}
                    celebrate
                    onUpdate={(data) => setResult({ kind: "draft", data })}
                  />
                )}
              </>
            )}
            {tab === "history" && <HistoryList />}
          </div>
        </main>
      </div>

      <ToastStack toasts={toasts} onDismiss={dismissToast} />
    </>
  );
}
