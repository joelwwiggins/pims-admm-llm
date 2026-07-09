/**
 * Backend client (Vite proxies /api and /health → :8008).
 */

const BASE = import.meta.env.VITE_API_BASE || '';

export async function getHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function getRouting() {
  const r = await fetch(`${BASE}/api/routing`);
  if (!r.ok) throw new Error(`routing ${r.status}`);
  return r.json();
}

/**
 * POST edge attempt for connect validation.
 * @returns {{ allowed: boolean, score: number, reason: string }}
 */
export async function postConnect(body) {
  const r = await fetch(`${BASE}/api/connect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`connect ${r.status}`);
  return r.json();
}

/**
 * POST full graph → LP/ADMM solve.
 * @param {object} graph { nodes, edges, recovery_path?, inventory_mode?, run_admm?, stub_only? }
 */
export async function postGraph(graph) {
  const body =
    graph?.nodes != null
      ? graph
      : { nodes: Array.isArray(graph) ? graph : [], edges: [] };

  const r = await fetch(`${BASE}/api/graph`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const j = await r.json();
      detail = j.detail || j.message || detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(`graph ${detail}`);
  }
  return r.json();
}
