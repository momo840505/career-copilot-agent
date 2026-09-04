// Phase 7b: thin fetch wrapper around the FastAPI backend (api/app.py).
//
// Two headers matter on almost every call:
//   - X-Client-Id: a random id generated once per browser and stored in localStorage.
//     It scopes "my history" in the UI. It is NOT a security boundary -- it's just a
//     self-reported label the backend uses to filter rows (see api/db.py's docstring
//     on get_history/list_history). Anyone could send a different value and see a
//     different "client's" history; the real protection (such as it is) is the
//     access code below.
//   - X-Access-Code: only required when the backend has ACCESS_CODE set (Phase 7's
//     shared-secret gate protecting a public deployment from burning your OpenAI
//     quota). Left unset in local dev, the backend's gate is a no-op and this header
//     is simply ignored.

const CLIENT_ID_KEY = "career_copilot_client_id";
const ACCESS_CODE_KEY = "career_copilot_access_code";

export function getClientId() {
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

export function getStoredAccessCode() {
  return localStorage.getItem(ACCESS_CODE_KEY) || "";
}

export function setStoredAccessCode(code) {
  localStorage.setItem(ACCESS_CODE_KEY, code);
}

export function clearStoredAccessCode() {
  localStorage.removeItem(ACCESS_CODE_KEY);
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Request failed with status ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function apiFetch(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    "X-Client-Id": getClientId(),
    ...options.headers,
  };
  // Only fall back to the stored code if the caller didn't already put one in
  // options.headers. verifyAccessCode(code) explicitly sets X-Access-Code to the
  // code the user just typed on the login screen -- unconditionally overwriting it
  // here with whatever's in localStorage (the old behavior) meant a freshly typed
  // *correct* code got silently replaced by a stale *stored* one (or vice versa:
  // typing a wrong code while a correct one was stored would validate against the
  // stored code, then overwrite it with the wrong one just typed), so the login
  // screen could reject a correct code, or accept a wrong one and then break every
  // later request. Every other caller here never sets X-Access-Code explicitly, so
  // this fallback still applies to them exactly as before.
  if (!("X-Access-Code" in headers)) {
    const accessCode = getStoredAccessCode();
    if (accessCode) {
      headers["X-Access-Code"] = accessCode;
    }
  }

  const response = await fetch(path, { ...options, headers });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch {
      // response body wasn't JSON -- keep the generic message
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) return null;
  return response.json();
}

export function verifyAccessCode(code) {
  return apiFetch("/auth/verify", {
    method: "POST",
    headers: code ? { "X-Access-Code": code } : {},
  });
}

export function runGapAnalysis(jdText) {
  return apiFetch("/gap-analysis", {
    method: "POST",
    body: JSON.stringify({ jd_text: jdText }),
  });
}

export function runDraft(jdText) {
  return apiFetch("/draft", {
    method: "POST",
    body: JSON.stringify({ jd_text: jdText }),
  });
}

// Resumes a paused draft (see api/app.py's /draft/{thread_id}/decision): action is
// "approve" (finalizes the letter -- this is the only path that shows up in History)
// or "revise" (sends it back through draft_writer with `feedback`, and comes back
// with another pending_review to approve/revise again).
export function submitDraftDecision(threadId, action, feedback) {
  return apiFetch(`/draft/${encodeURIComponent(threadId)}/decision`, {
    method: "POST",
    body: JSON.stringify(feedback ? { action, feedback } : { action }),
  });
}

export function fetchHistoryList() {
  return apiFetch("/history", { method: "GET" });
}

export function fetchHistoryDetail(id) {
  return apiFetch(`/history/${encodeURIComponent(id)}`, { method: "GET" });
}
