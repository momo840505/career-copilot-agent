import { IconCheckCircle, IconAlertTriangle, IconXCircle, IconSparkles } from "../icons.jsx";

// Renders a GapAnalysisResponse (see api/app.py): matched / partial / missing
// requirements, each with evidence chunk ids and a note, plus suggested talking
// points and an overall fit summary. Also used by HistoryList when a saved record's
// kind is "gap_analysis".

function GapItemList({ items, tone }) {
  if (!items || items.length === 0) {
    return <p className="gap-empty">None.</p>;
  }
  return (
    <ul className={`gap-item-list gap-section-${tone}`}>
      {items.map((item, i) => (
        <li key={i}>
          <strong>{item.requirement}</strong>
          {item.note && <span className="gap-note"> — {item.note}</span>}
          {item.evidence_chunk_ids && item.evidence_chunk_ids.length > 0 && (
            <div className="gap-evidence">Evidence: {item.evidence_chunk_ids.join(", ")}</div>
          )}
        </li>
      ))}
    </ul>
  );
}

function StatCard({ icon, count, label, tone }) {
  return (
    <div className={`stat-card stat-${tone}`}>
      <span className="stat-icon">{icon}</span>
      <span className="stat-count">{count}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

// A round "fit score" ring: matched counts fully, partial counts half. Purely a
// friendly at-a-glance visual on top of the same matched/partial/missing counts
// already shown below -- not a new field from the backend.
function ScoreRing({ matched, partial, total }) {
  const score = total > 0 ? Math.round(((matched + partial * 0.5) / total) * 100) : 0;
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - score / 100);
  const tone = score >= 70 ? "matched" : score >= 40 ? "partial" : "missing";

  return (
    <div className="score-ring">
      <svg viewBox="0 0 100 100" width="96" height="96">
        <circle cx="50" cy="50" r={radius} className="score-ring-track" />
        <circle
          cx="50"
          cy="50"
          r={radius}
          className={`score-ring-fill score-ring-${tone}`}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="score-ring-label">
        <span className="score-ring-number">{score}%</span>
        <span className="score-ring-caption">fit</span>
      </div>
    </div>
  );
}

export default function GapAnalysisResult({ result }) {
  const matched = result.matched?.length || 0;
  const partial = result.partial?.length || 0;
  const missing = result.missing?.length || 0;
  const total = matched + partial + missing || 1;

  return (
    <div className="result-card animate-in">
      <div className="gap-headline">
        <div>
          <h2>{result.job_title || "Gap Analysis"}</h2>
          {result.overall_fit_summary && <p className="fit-summary">"{result.overall_fit_summary}"</p>}
        </div>
        <ScoreRing matched={matched} partial={partial} total={total} />
      </div>

      <div className="stats-strip">
        <StatCard icon={<IconCheckCircle size={16} />} count={matched} label="Matched" tone="matched" />
        <StatCard icon={<IconAlertTriangle size={16} />} count={partial} label="Partial" tone="partial" />
        <StatCard icon={<IconXCircle size={16} />} count={missing} label="Missing" tone="missing" />
      </div>
      <div className="stat-bar">
        <span className="stat-bar-segment stat-bar-matched" style={{ width: `${(matched / total) * 100}%` }} />
        <span className="stat-bar-segment stat-bar-partial" style={{ width: `${(partial / total) * 100}%` }} />
        <span className="stat-bar-segment stat-bar-missing" style={{ width: `${(missing / total) * 100}%` }} />
      </div>

      <h3>
        <IconCheckCircle size={14} /> Matched
      </h3>
      <GapItemList items={result.matched} tone="matched" />

      <h3>
        <IconAlertTriangle size={14} /> Partial
      </h3>
      <GapItemList items={result.partial} tone="partial" />

      <h3>
        <IconXCircle size={14} /> Missing
      </h3>
      <GapItemList items={result.missing} tone="missing" />

      {result.suggested_talking_points && result.suggested_talking_points.length > 0 && (
        <>
          <h3>
            <IconSparkles size={14} /> Suggested Talking Points
          </h3>
          <ul className="talking-points">
            {result.suggested_talking_points.map((point, i) => (
              <li key={i}>{point}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
