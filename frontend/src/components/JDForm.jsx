import { useState } from "react";
import { runGapAnalysis, runDraft, ApiError } from "../api.js";
import { IconFileText, IconBarChart, IconSparkles, IconAlertTriangle, IconZap } from "../icons.jsx";

const EXAMPLE_JD = `1. Data organization & analysis support
Help clean, organize, and cross-check data
Build tables and reports using Excel / Google Sheets
Assist with analysis of operational, course, or project data
Turn raw data into clear summaries and charts

2. Software & systems development support
Help engineers with system testing, data setup, and functional verification
Help write simple technical docs, usage guides, and test logs
Participate in projects for websites, internal systems, automation tools, or ERP systems

3. AI tooling & automation support
Help apply AI tools to improve data organization, document output, and workflow efficiency
Help design AI use cases, prompts, and automation flows`;

// Two actions on the same JD text: a quick gap analysis, or the full pipeline
// (gap analysis + draft + critic loop) that produces a cover letter. onResult is
// called with { kind: "gap_analysis" | "draft", data } on success. notify() (from
// App.jsx) pops a toast so success feels immediate even though the result card
// itself renders right below the fold.
export default function JDForm({ onResult, notify }) {
  const [jdText, setJdText] = useState("");
  const [loading, setLoading] = useState(null); // null | "gap" | "draft"
  const [error, setError] = useState("");

  async function handleRun(kind) {
    if (!jdText.trim()) {
      setError("Paste a job description first.");
      return;
    }
    setError("");
    setLoading(kind);
    try {
      if (kind === "gap") {
        const data = await runGapAnalysis(jdText);
        onResult({ kind: "gap_analysis", data });
        notify?.("Gap analysis ready! 🎯");
      } else {
        const data = await runDraft(jdText);
        onResult({ kind: "draft", data });
        notify?.(data.critic_passed ? "Cover letter drafted! ✨" : "Draft ready -- the critic flagged a few things.");
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 502) {
        setError(
          "The AI model couldn't produce a valid result after several retries. This can happen with unusual job descriptions -- try again, or try a shorter JD."
        );
      } else if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError("Could not reach the server. Is the backend running?");
      }
    } finally {
      setLoading(null);
    }
  }

  function handleKeyDown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && loading === null) {
      event.preventDefault();
      handleRun("draft");
    }
  }

  return (
    <div className="jd-form">
      <div className="jd-form-header">
        <label htmlFor="jd-text">
          <IconFileText size={15} /> Job description
        </label>
        <div className="jd-form-header-actions">
          {jdText.length > 0 && <span className="char-count">{jdText.length.toLocaleString()} characters</span>}
          <button type="button" className="link-btn" onClick={() => setJdText(EXAMPLE_JD)}>
            Try an example
          </button>
        </div>
      </div>
      <textarea
        id="jd-text"
        rows={14}
        value={jdText}
        onChange={(e) => setJdText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Paste the full job description here..."
      />
      {error && (
        <p className="form-error">
          <IconAlertTriangle size={15} /> {error}
        </p>
      )}
      <div className="jd-form-actions">
        <button onClick={() => handleRun("gap")} disabled={loading !== null}>
          {loading === "gap" ? (
            <>
              <span className="spinner" /> Analyzing...
            </>
          ) : (
            <>
              <IconBarChart size={15} /> Analyze Skill Gaps
            </>
          )}
        </button>
        <button onClick={() => handleRun("draft")} disabled={loading !== null} className="primary">
          {loading === "draft" ? (
            <>
              <span className="spinner spinner-light" /> Generating...
            </>
          ) : (
            <>
              <IconSparkles size={15} /> Generate Cover Letter
            </>
          )}
        </button>
        <span className="kbd-hint">
          <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Enter</kbd> to generate
        </span>
      </div>
      {loading && (
        <div className="loading-hint">
          <IconZap size={13} /> The critic model reviews and can request revisions -- this may take a moment.
        </div>
      )}
    </div>
  );
}
