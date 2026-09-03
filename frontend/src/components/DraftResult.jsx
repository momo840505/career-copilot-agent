import { useEffect, useState } from "react";
import { IconCheckCircle, IconAlertTriangle, IconCopy, IconCheck, IconDownload } from "../icons.jsx";

const CONFETTI_EMOJI = ["🎉", "✨", "🎊", "⭐", "💫", "🌟"];

function ConfettiBurst() {
  const [pieces] = useState(() =>
    Array.from({ length: 14 }, (_, i) => ({
      id: i,
      emoji: CONFETTI_EMOJI[i % CONFETTI_EMOJI.length],
      left: Math.random() * 100,
      delay: Math.random() * 0.3,
      duration: 1 + Math.random() * 0.6,
      drift: (Math.random() - 0.5) * 80,
    }))
  );

  return (
    <div className="confetti-layer" aria-hidden="true">
      {pieces.map((p) => (
        <span
          key={p.id}
          className="confetti-piece"
          style={{
            left: `${p.left}%`,
            animationDelay: `${p.delay}s`,
            animationDuration: `${p.duration}s`,
            "--drift": `${p.drift}px`,
          }}
        >
          {p.emoji}
        </span>
      ))}
    </div>
  );
}

// Renders a DraftResponse (see api/app.py): the cover letter itself (greeting / body
// / closing), the critic's verdict, and the claims made in the letter with their
// supporting evidence chunk ids. Also used by HistoryList when a saved record's kind
// is "draft" -- there, `celebrate` is left false so re-opening an old letter doesn't
// replay the confetti every time.
export default function DraftResult({ result, celebrate = false }) {
  const [copied, setCopied] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const letterText = `${result.greeting}\n\n${result.body}\n\n${result.closing}`;

  useEffect(() => {
    if (celebrate && result.critic_passed) {
      setShowConfetti(true);
      const timer = setTimeout(() => setShowConfetti(false), 1800);
      return () => clearTimeout(timer);
    }
  }, [celebrate, result]);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(letterText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard API can be unavailable (e.g. non-HTTPS/non-localhost) -- fail
      // quietly, the letter text is still fully visible and selectable on the page.
    }
  }

  function handleDownload() {
    const blob = new Blob([letterText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeTitle = (result.job_title || "cover-letter").toLowerCase().replace(/[^a-z0-9]+/g, "-");
    a.download = `${safeTitle}-cover-letter.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="result-card animate-in draft-card">
      {showConfetti && <ConfettiBurst />}
      <div className="draft-header">
        <h2>{result.job_title || "Cover Letter"}</h2>
        <div className="draft-header-actions">
          <button className="copy-btn" onClick={handleDownload}>
            <IconDownload size={14} /> Download
          </button>
          <button className="copy-btn" onClick={handleCopy}>
            {copied ? (
              <>
                <IconCheck size={14} /> Copied
              </>
            ) : (
              <>
                <IconCopy size={14} /> Copy letter
              </>
            )}
          </button>
        </div>
      </div>

      <span className={`critic-badge ${result.critic_passed ? "passed" : "flagged"}`}>
        {result.critic_passed ? <IconCheckCircle size={13} /> : <IconAlertTriangle size={13} />}
        {result.critic_passed ? "Critic: passed" : "Critic: flagged"}
        {typeof result.revision_count === "number" && ` · ${result.revision_count} revision${result.revision_count === 1 ? "" : "s"}`}
      </span>

      {!result.critic_passed && result.critic_issues && result.critic_issues.length > 0 && (
        <ul className="critic-issues">
          {result.critic_issues.map((issue, i) => (
            <li key={i}>{issue}</li>
          ))}
        </ul>
      )}

      <div className="letter paper">
        <p>{result.greeting}</p>
        <p className="letter-body">{result.body}</p>
        <p>{result.closing}</p>
      </div>

      {result.claims && result.claims.length > 0 && (
        <>
          <h3>Claims &amp; Evidence</h3>
          <ul className="claims-list">
            {result.claims.map((claim, i) => (
              <li key={i}>
                <span className="claim-index">{i + 1}</span>
                <div>
                  <p>{claim.text}</p>
                  {claim.evidence_chunk_ids && claim.evidence_chunk_ids.length > 0 && (
                    <div className="gap-evidence">Evidence: {claim.evidence_chunk_ids.join(", ")}</div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
