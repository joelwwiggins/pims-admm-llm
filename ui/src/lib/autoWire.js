/**
 * Map /api/auto_wire suggestions onto the SvelteFlow PFD graph.
 *
 * When COKER (or FCC) is dropped, the backend invents feed + product edges.
 * This module:
 *  - finds source/target nodes by unitType
 *  - creates lightweight product-terminal nodes when sinks are not on canvas
 *  - avoids duplicate stream edges
 */

import { MarkerType } from '@xyflow/svelte';
import { UNIT_COLORS, UNIT_PORTS, emptyStream, defaultProcessConditions } from './data/plantTemplate.js';
import { paletteByType } from './palette.js';

/** Sinks that are conversion units on the palette */
const UNIT_SINKS = new Set([
  'CDU',
  'FCC',
  'COKER',
  'REFORMER',
  'HDT_NAPH',
  'BLENDER',
  'TANK',
  'SELL',
  'POOL_FCC',
  'POOL_COKER',
  'POOL_REFORMER',
]);

/** Product terminals → canvas unit type + label */
const TERMINAL_MAP = {
  FUEL_GAS: { unitType: 'SELL', label: 'Fuel Gas' },
  LPG: { unitType: 'SELL', label: 'LPG' },
  GASOLINE: { unitType: 'BLENDER', label: 'Gasoline pool' },
  DIESEL: { unitType: 'SELL', label: 'Diesel' },
  FO: { unitType: 'SELL', label: 'Fuel Oil' },
  REGEN_HEAT: { unitType: 'SELL', label: 'Regen heat' },
  COKE_SALES: { unitType: 'SELL', label: 'Coke sales' },
  H2_GRID: { unitType: 'SELL', label: 'H2 grid' },
  SELL: { unitType: 'SELL', label: 'Sell' },
  BLENDER: { unitType: 'BLENDER', label: 'Blender' },
  HDT_NAPH: { unitType: 'HDT_NAPH', label: 'HDT Naphtha' },
  FCC: { unitType: 'FCC', label: 'FCC' },
  COKER: { unitType: 'COKER', label: 'Coker' },
  REFORMER: { unitType: 'REFORMER', label: 'Reformer' },
  POOL_FCC: { unitType: 'TANK', label: 'FCC pool' },
  POOL_COKER: { unitType: 'TANK', label: 'Coker pool' },
  POOL_REFORMER: { unitType: 'TANK', label: 'Reformer pool' },
};

/** Prefer these source handles for common streams */
const STREAM_SOURCE_HANDLE = {
  cdu_gasoil: 'cdu_go',
  cdu_resid: 'cdu_resid',
  cdu_naphtha_light: 'cdu_naph_l',
  cdu_naphtha_heavy: 'cdu_naph_h',
  cdu_distillate: 'cdu_dist',
  cdu_offgas: 'cdu_naph_l',
  fcc_dry_gas: 'fcc_dry',
  fcc_lpg: 'fcc_lpg',
  fcc_naphtha: 'fcc_naph',
  fcc_lco: 'fcc_lco',
  fcc_slurry: 'fcc_slurry',
  fcc_coke: 'fcc_coke',
  coker_dry_gas: 'coker_dry',
  coker_lpg: 'coker_lpg',
  coker_naphtha: 'coker_naph',
  coker_gasoil: 'coker_go',
  coker_coke: 'coke',
};

const STREAM_TARGET_HANDLE = {
  cdu_gasoil: 'fcc_feed',
  cdu_resid: 'coker_feed',
  fcc_naphtha: 'hdt_fcc_in',
  coker_naphtha: 'hdt_cok_in',
};

function markerEnd() {
  return {
    type: MarkerType.ArrowClosed,
    width: 18,
    height: 18,
    color: '#6a8fb0',
  };
}

export function activeUnitTypes(nodes) {
  const types = new Set();
  for (const n of nodes || []) {
    const t = n.data?.unitType;
    if (t) types.add(String(t).toUpperCase());
  }
  // Always include CDU tower if any conversion unit present for wire API
  if (types.size && !types.has('CDU')) {
    // keep as-is; caller may still have only COKER mid-drop
  }
  return [...types];
}

export function edgesForApi(edges) {
  return (edges || []).map((e) => ({
    id: e.id,
    stream: e.data?.streamName || e.data?.label || e.sourceHandle || '',
    from: e.data?.fromUnit || '',
    to: e.data?.toUnit || '',
    source: e.source,
    target: e.target,
  }));
}

function findNodeByUnitType(nodes, unitType) {
  const want = String(unitType).toUpperCase();
  // Prefer active process units; last match wins for recently added
  let found = null;
  for (const n of nodes) {
    if (String(n.data?.unitType || '').toUpperCase() === want && n.data?.active !== false) {
      found = n;
    }
  }
  return found;
}

function findTerminalNode(nodes, sinkKey) {
  const key = String(sinkKey).toUpperCase();
  const mapped = TERMINAL_MAP[key];
  if (!mapped) return findNodeByUnitType(nodes, key);
  // Prefer exact terminal id convention
  const byId = nodes.find((n) => n.id === `term-${key.toLowerCase()}`);
  if (byId) return byId;
  // Prefer unit type with matching sinkLabel
  const labeled = nodes.find(
    (n) =>
      String(n.data?.unitType || '').toUpperCase() === mapped.unitType &&
      (n.data?.sinkKey === key || n.data?.label === mapped.label),
  );
  if (labeled) return labeled;
  return findNodeByUnitType(nodes, mapped.unitType);
}

function makeTerminalNode(sinkKey, seq, near) {
  const key = String(sinkKey).toUpperCase();
  const mapped = TERMINAL_MAP[key] || { unitType: 'SELL', label: key };
  const unitType = mapped.unitType;
  const colors = UNIT_COLORS[unitType] || UNIT_COLORS.SELL;
  const p = paletteByType(unitType);
  const id = `term-${key.toLowerCase()}`;
  const x = (near?.position?.x ?? 200) + 220;
  const y = (near?.position?.y ?? 120) + (seq % 5) * 70;
  return {
    id,
    type: 'pfdUnit',
    position: { x, y },
    data: {
      tag: `T-${key.slice(0, 4)}`,
      label: mapped.label,
      unitType,
      sinkKey: key,
      autoTerminal: true,
      active: true,
      submodel: 'lp',
      status: 'idle',
      headerColor: colors.header,
      accentColor: colors.accent,
      description: `Auto terminal for ${key}`,
      charge_kbd: 0,
      yields: [],
      ports: UNIT_PORTS[unitType] || UNIT_PORTS.SELL,
      processConditions: defaultProcessConditions(unitType),
    },
  };
}

function edgeExists(edges, stream, sourceId, targetId) {
  return edges.some((e) => {
    const s = e.data?.streamName || e.data?.label || e.sourceHandle;
    return (
      e.source === sourceId &&
      e.target === targetId &&
      String(s || '').toLowerCase().includes(String(stream || '').toLowerCase().slice(0, 8))
    );
  });
}

/**
 * Apply API auto_wire edges onto nodes/edges state.
 * @returns {{ nodes, edges, added, skipped, createdTerminals }}
 */
export function applyAutoWireToGraph(nodesIn, edgesIn, apiEdges, opts = {}) {
  let nodes = [...(nodesIn || [])];
  let edges = [...(edgesIn || [])];
  const added = [];
  const skipped = [];
  const createdTerminals = [];
  let termSeq = opts.termSeq || 0;

  for (const ae of apiEdges || []) {
    const stream = ae.stream || '';
    const fromType = String(ae.from || '').toUpperCase();
    const toKey = String(ae.to || '').toUpperCase();

    let source = findNodeByUnitType(nodes, fromType);
    if (!source && fromType) {
      skipped.push({ ...ae, reason: `no source node ${fromType}` });
      continue;
    }

    let target = null;
    if (UNIT_SINKS.has(toKey) && !TERMINAL_MAP[toKey]) {
      target = findNodeByUnitType(nodes, toKey);
    } else {
      target = findTerminalNode(nodes, toKey);
      if (!target && TERMINAL_MAP[toKey]) {
        const term = makeTerminalNode(toKey, termSeq++, source);
        nodes = [...nodes, term];
        createdTerminals.push(term.id);
        target = term;
      } else if (!target) {
        target = findNodeByUnitType(nodes, toKey);
      }
    }

    if (!target) {
      skipped.push({ ...ae, reason: `no target for ${toKey}` });
      continue;
    }
    if (source.id === target.id) {
      skipped.push({ ...ae, reason: 'self-loop' });
      continue;
    }
    if (edgeExists(edges, stream, source.id, target.id)) {
      skipped.push({ ...ae, reason: 'duplicate' });
      continue;
    }

    const sourceHandle = STREAM_SOURCE_HANDLE[stream] || undefined;
    const targetHandle = STREAM_TARGET_HANDLE[stream] || undefined;
    const id = `aw-${stream || 's'}-${source.id}-${target.id}`;
    const edge = {
      id,
      type: 'streamEdge',
      source: source.id,
      target: target.id,
      sourceHandle,
      targetHandle,
      markerEnd: markerEnd(),
      animated: !!ae.auto,
      data: {
        label: stream || `${fromType}→${toKey}`,
        streamName: stream,
        fromUnit: fromType,
        toUnit: toKey,
        stream: emptyStream(stream || 'stream'),
        score: ae.score ?? 0.9,
        reason: ae.reason || 'auto_wire',
        auto: true,
      },
    };
    edges = [...edges, edge];
    added.push(edge);
  }

  return { nodes, edges, added, skipped, createdTerminals, termSeq };
}
