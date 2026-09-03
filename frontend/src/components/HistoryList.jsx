import { useEffect, useState } from "react";
import { fetchHistoryList, fetchHistoryDetail, ApiError } from "../api.js";
import GapAnalysisResult from "./GapAnalysisResult.jsx";
import DraftResult from "./DraftResult.jsx";
import { IconFileText, IconSparkles, IconInbox } from "../icons.jsx";

// List of this browser's past runs (scoped by X-Client-Id, see api.js) on the left,
// full detail of whichever one is selected on the right.
export default function HistoryList() {
  const [records, setRecords] = useState(null); // null = still loading
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    fetchHistoryList()
      .then(setRecords)
      .catch((err) => {
        setError(err instanceof ApiError ? err.detail : "Could not reach the server.");
        setRecords([]);
      });
  }, []);

  function handleSelect(id) {
    setSelectedId(id);
    setDetail(null);
    setDetailLoading(true);
    fetchHistoryDetail(id)
      .then(setDetail)
      .catch((err) => {
        setError(err instanceof ApiError ? err.detail : "Could not load that record.");
      })
      .finally(() => setDetailLoading(false));
  }

  return (
    <div className="history-layout">
      <div className="history-list">
        <h2>History</h2>
        {error && <p className="form-error">{error}</p>}
        {records === null && (
          <div className="skeleton-stack">
            <div className="skeleton skeleton-item" />
            <div className="skeleton skeleton-item" />
            <div className="skeleton skeleton-item" />
          </div>
        )}
        {records && records.length === 0 && (
          <div className="empty-state">
            <IconInbox size={22} />
            <p>No runs yet. Analyze a job description to see it here.</p>
          </div>
        )}
        <ul>
          {records &&
            records.map((r) => (
              <li key={r.id}>
                <button
                  className={r.id === selectedId ? "history-item selected" : "history-item"}
                  onClick={() => handleSelect(r.id)}
                >
                  <span className="history-kind">
                    {r.kind === "draft" ? <IconSparkles size={12} /> : <IconFileText size={12} />}
                    {r.kind === "draft" ? "Cover Letter" : "Gap Analysis"}
                  </span>
                  <span className="history-title">{r.job_title}</span>
                  <span className="history-date">{new Date(r.created_at).toLocaleString()}</span>
                </button>
              </li>
            ))}
        </ul>
      </div>
      <div className="history-detail">
        {detailLoading && (
          <div className="result-card">
            <div className="skeleton skeleton-title" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line" style={{ width: "80%" }} />
            <div className="skeleton skeleton-line" style={{ width: "60%" }} />
          </div>
        )}
        {!detailLoading && !detail && (
          <div className="empty-state empty-state-bordered">
            <IconInbox size={22} />
            <p>Select a run to view its details.</p>
          </div>
        )}
        {detail && detail.kind === "draft" && <DraftResult result={detail.result} celebrate={false} />}
        {detail && detail.kind === "gap_analysis" && <GapAnalysisResult result={detail.result} />}
      </div>
    </div>
  );
}
