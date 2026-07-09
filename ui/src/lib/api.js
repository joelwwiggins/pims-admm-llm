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
 * POST full graph; returns cluster/ADMM stubs.
 * @param {{ nodes: any[], edges: any[] } | any[]} nodesOrGraph
 * @param {any[]} [edges]
 */
export async function postGraph(nodesOrGraph, edges) {
  const body =
    edges !== undefined
      ? { nodes: nodesOrGraph, edges }
      : nodesOrGraph?.nodes
        ? nodesOrGraph
        : { nodes: nodesOrGraph, edges: [] };

  const r = await fetch(`${BASE}/api/graph`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`graph ${r.status}`);
  return r.json();
}
