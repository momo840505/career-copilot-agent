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
  const accessCode = getStoredAccessCode();
  if (accessCode) {
    headers["X-Access-Code"] = accessCode;
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

export function fetchHistoryList() {
  return apiFetch("/history", { method: "GET" });
}

export function fetchHistoryDetail(id) {
  return apiFetch(`/history/${encodeURIComponent(id)}`, { method: "GET" });
}
