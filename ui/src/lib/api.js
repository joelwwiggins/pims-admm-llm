const BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8008';

async function jsonFetch(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export function getHealth() {
  return jsonFetch('/health');
}

export function getRouting() {
  return jsonFetch('/api/routing');
}

export function postGraph(body) {
  return jsonFetch('/api/graph', { method: 'POST', body: JSON.stringify(body) });
}

export function postConnect(body) {
  return jsonFetch('/api/connect', { method: 'POST', body: JSON.stringify(body) });
}
