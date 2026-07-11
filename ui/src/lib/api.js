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
 * @returns {{ allowed: boolean, score: number, reason: string, guesses?: any[], best?: any }}
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
 * When units are added, invent feed + product edges.
 * @param {{ active_units: string[], existing_edges?: object[] }} body
 */
export async function postAutoWire(body) {
  const r = await fetch(`${BASE}/api/auto_wire`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`auto_wire ${r.status}`);
  return r.json();
}

/**
 * Base-delta cascade solve (CDU+FCC[+COKER]).
 * @param {{ active_units?: string[], enable_coker?: boolean, max_crude_kbd?: number, crude_api?: number, drawn_edges?: object[] }} body
 */
export async function postBaseDeltaSolve(body) {
  const r = await fetch(`${BASE}/api/base_delta/solve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`base_delta ${r.status}`);
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

/** Absolute/relative URL for Excel results download (basename under api_excel/). */
export function excelResultsUrl(pathOrBasename) {
  const base = String(pathOrBasename || '').split(/[/\\]/).pop();
  if (!base) return '';
  return `${BASE}/api/excel/results?path=${encodeURIComponent(base)}`;
}

/** Download PIMS-shaped template (.xlsx blob). */
export async function getExcelTemplateBlob() {
  const r = await fetch(`${BASE}/api/excel/template`);
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const j = await r.json();
      detail = j.error || j.detail || detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(`excel template ${detail}`);
  }
  return r.blob();
}

/**
 * Upload PIMS-shaped workbook → mono + ADMM JSON summary.
 * @param {File|Blob} file
 * @param {{ returnXlsx?: boolean }} [opts]
 * @returns {Promise<object|Blob>} JSON body, or Blob when returnXlsx
 */
export async function postExcelSolve(file, opts = {}) {
  const form = new FormData();
  const name = file?.name || 'model.xlsx';
  form.append('file', file, name);
  const q = opts.returnXlsx ? '?return_xlsx=true' : '';
  const r = await fetch(`${BASE}/api/excel/solve${q}`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const j = await r.json();
      detail = j.error || j.detail || detail;
    } catch (_) {
      try {
        detail = (await r.text()).slice(0, 200) || detail;
      } catch (__) {
        /* ignore */
      }
    }
    throw new Error(`excel solve ${detail}`);
  }
  if (opts.returnXlsx) return r.blob();
  return r.json();
}
