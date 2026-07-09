/**
 * HYSYS-style full plant template: nodes + stream edges with rich metadata.
 * Layout is left-to-right process flow (CDU → conversion → blender).
 */

export const UNIT_COLORS = {
  CDU: { header: '#1e5a8a', accent: '#4a9fe0' },
  FCC: { header: '#8a4520', accent: '#e07a3d' },
  COKER: { header: '#6b1f1f', accent: '#d94a4a' },
  REFORMER: { header: '#1a5c38', accent: '#3dba72' },
  HDT_NAPH: { header: '#145a52', accent: '#2ab5a0' },
  BLENDER: { header: '#4a2a6b', accent: '#a56fd4' },
  TANK: { header: '#3a4550', accent: '#8a9aab' },
  SELL: { header: '#6b5a10', accent: '#e0c040' },
  warehouse: { header: '#3a3060', accent: '#8a7ab8' },
  transport: { header: '#3a5030', accent: '#7aaa50' },
};

/** @typedef {{ id: string, name: string, side: 'left'|'right'|'bottom'|'top', type?: 'feed'|'product'|'utility' }} PortDef */

/** Port definitions per unit type (realistic multi-stream PFD). */
export const UNIT_PORTS = {
  CDU: {
    feeds: [{ id: 'crude_in', name: 'Crude', side: 'left', type: 'feed' }],
    products: [
      { id: 'cdu_naph_l', name: 'Lt Naph', side: 'right', type: 'product' },
      { id: 'cdu_naph_h', name: 'Hvy Naph', side: 'right', type: 'product' },
      { id: 'cdu_dist', name: 'Distillate', side: 'right', type: 'product' },
      { id: 'cdu_go', name: 'Gasoil', side: 'right', type: 'product' },
      { id: 'cdu_resid', name: 'Resid', side: 'bottom', type: 'product' },
    ],
  },
  FCC: {
    feeds: [{ id: 'fcc_feed', name: 'Gasoil', side: 'left', type: 'feed' }],
    products: [
      { id: 'fcc_naph', name: 'FCC Naph', side: 'right', type: 'product' },
      { id: 'fcc_lco', name: 'LCO', side: 'right', type: 'product' },
      { id: 'fcc_slurry', name: 'Slurry', side: 'bottom', type: 'product' },
    ],
  },
  COKER: {
    feeds: [{ id: 'coker_feed', name: 'Resid', side: 'left', type: 'feed' }],
    products: [
      { id: 'coker_naph', name: 'Coker Naph', side: 'right', type: 'product' },
      { id: 'coker_go', name: 'Coker GO', side: 'right', type: 'product' },
      { id: 'coke', name: 'Coke', side: 'bottom', type: 'product' },
    ],
  },
  REFORMER: {
    feeds: [{ id: 'ref_feed', name: 'Hvy Naph', side: 'left', type: 'feed' }],
    products: [{ id: 'reformate', name: 'Reformate', side: 'right', type: 'product' }],
  },
  HDT_NAPH: {
    feeds: [
      { id: 'hdt_fcc_in', name: 'FCC Naph', side: 'left', type: 'feed' },
      { id: 'hdt_cok_in', name: 'Coker Naph', side: 'left', type: 'feed' },
    ],
    products: [{ id: 'hdt_out', name: 'HDT Naph', side: 'right', type: 'product' }],
  },
  BLENDER: {
    feeds: [
      { id: 'bl_ref', name: 'Reformate', side: 'left', type: 'feed' },
      { id: 'bl_hdt', name: 'HDT Naph', side: 'left', type: 'feed' },
      { id: 'bl_lco', name: 'LCO/Slurry', side: 'left', type: 'feed' },
      { id: 'bl_cgo', name: 'Coker GO', side: 'left', type: 'feed' },
      { id: 'bl_dist', name: 'Distillate', side: 'left', type: 'feed' },
    ],
    products: [
      { id: 'gasoline', name: 'Gasoline', side: 'right', type: 'product' },
      { id: 'diesel', name: 'Diesel', side: 'right', type: 'product' },
      { id: 'fo', name: 'Fuel Oil', side: 'bottom', type: 'product' },
    ],
  },
  TANK: {
    feeds: [{ id: 'tank_in', name: 'In', side: 'left', type: 'feed' }],
    products: [{ id: 'tank_out', name: 'Out', side: 'right', type: 'product' }],
  },
  SELL: {
    feeds: [{ id: 'sell_in', name: 'In', side: 'left', type: 'feed' }],
    products: [],
  },
  warehouse: {
    feeds: [{ id: 'wh_in', name: 'In', side: 'left', type: 'feed' }],
    products: [{ id: 'wh_out', name: 'Out', side: 'right', type: 'product' }],
  },
  transport: {
    feeds: [{ id: 'tr_in', name: 'In', side: 'left', type: 'feed' }],
    products: [{ id: 'tr_out', name: 'Out', side: 'right', type: 'product' }],
  },
};

function yieldRows(rows) {
  return rows.map((r) => ({
    product: r[0],
    yield_pct: r[1],
    flow_kbd: r[2],
    sulfur_wt: r[3],
    ron: r[4],
  }));
}

export function createFullPlantPfd() {
  const colors = UNIT_COLORS;

  const nodes = [
    {
      id: 'cdu-1',
      type: 'pfdUnit',
      position: { x: 40, y: 220 },
      data: {
        tag: 'U-100',
        label: 'CDU',
        unitType: 'CDU',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.CDU.header,
        accentColor: colors.CDU.accent,
        description: 'Atmospheric crude distillation',
        charge_kbd: 140.0,
        yields: yieldRows([
          ['Lt Naphtha', 12.5, 17.5, 0.005, 68],
          ['Hvy Naphtha', 14.0, 19.6, 0.01, 55],
          ['Distillate', 23.7, 33.2, 0.12, 0],
          ['Gasoil', 28.9, 40.4, 0.45, 0],
          ['Resid', 20.9, 29.3, 2.5, 0],
        ]),
        ports: UNIT_PORTS.CDU,
      },
    },
    {
      id: 'fcc-1',
      type: 'pfdUnit',
      position: { x: 320, y: 40 },
      data: {
        tag: 'U-210',
        label: 'FCC',
        unitType: 'FCC',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.FCC.header,
        accentColor: colors.FCC.accent,
        description: 'Fluid catalytic cracker',
        charge_kbd: 40.4,
        yields: yieldRows([
          ['FCC Naphtha', 27.9, 11.3, 0.005, 92],
          ['LCO', 35.9, 14.5, 0.35, 0],
          ['Slurry', 24.7, 10.0, 0.8, 0],
          ['Gas/Coke', 11.5, 4.6, 0, 0],
        ]),
        ports: UNIT_PORTS.FCC,
      },
    },
    {
      id: 'cok-1',
      type: 'pfdUnit',
      position: { x: 320, y: 400 },
      data: {
        tag: 'U-220',
        label: 'Coker',
        unitType: 'COKER',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.COKER.header,
        accentColor: colors.COKER.accent,
        description: 'Delayed coker',
        charge_kbd: 40.0,
        yields: yieldRows([
          ['Coker Naphtha', 16.1, 6.4, 0.25, 72],
          ['Coker GO', 53.8, 21.5, 0.4, 0],
          ['Coke', 22.0, 8.8, 0, 0],
          ['Gas', 8.1, 3.2, 0, 0],
        ]),
        ports: UNIT_PORTS.COKER,
      },
    },
    {
      id: 'ref-1',
      type: 'pfdUnit',
      position: { x: 600, y: 220 },
      data: {
        tag: 'U-310',
        label: 'Reformer',
        unitType: 'REFORMER',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.REFORMER.header,
        accentColor: colors.REFORMER.accent,
        description: 'Catalytic reformer (heavy SR naphtha)',
        charge_kbd: 12.0,
        yields: yieldRows([
          ['Reformate', 86.0, 10.3, 0.0005, 100],
          ['H2 / lights', 14.0, 1.7, 0, 0],
        ]),
        ports: UNIT_PORTS.REFORMER,
      },
    },
    {
      id: 'hdt-1',
      type: 'pfdUnit',
      position: { x: 600, y: 40 },
      data: {
        tag: 'U-320',
        label: 'HDT Naphtha',
        unitType: 'HDT_NAPH',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.HDT_NAPH.header,
        accentColor: colors.HDT_NAPH.accent,
        description: 'Naphtha hydrotreater (FCC + coker naph)',
        charge_kbd: 17.7,
        yields: yieldRows([
          ['HDT Naphtha', 98.0, 17.3, 0.008, 88],
          ['Lights', 2.0, 0.4, 0, 0],
        ]),
        ports: UNIT_PORTS.HDT_NAPH,
      },
    },
    {
      id: 'bl-1',
      type: 'pfdUnit',
      position: { x: 880, y: 200 },
      data: {
        tag: 'U-400',
        label: 'Blender',
        unitType: 'BLENDER',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.BLENDER.header,
        accentColor: colors.BLENDER.accent,
        description: 'Product pool blender (RON / S specs)',
        charge_kbd: 0,
        yields: yieldRows([
          ['Gasoline', 0, 35.8, 0.01, 91],
          ['Diesel', 0, 62.8, 0.05, 0],
          ['Fuel Oil', 0, 15.6, 1.2, 0],
        ]),
        ports: UNIT_PORTS.BLENDER,
      },
    },
  ];

  /** Stream property factory */
  function stream(name, flow, extra = {}) {
    return {
      name,
      flow_kbd: flow,
      temperature_f: extra.T ?? 200,
      pressure_psig: extra.P ?? 50,
      vapor_fraction: extra.vf ?? 0,
      density_api: extra.api ?? 35,
      sulfur_wt: extra.S ?? 0.1,
      ron: extra.ron ?? null,
      composition: extra.comp || [
        { cut: 'C5-C6', vol_pct: 12 },
        { cut: 'C7-C8', vol_pct: 28 },
        { cut: 'C9-C10', vol_pct: 35 },
        { cut: 'C11+', vol_pct: 25 },
      ],
    };
  }

  const edges = [
    {
      id: 's-gasoil',
      type: 'streamEdge',
      source: 'cdu-1',
      target: 'fcc-1',
      sourceHandle: 'cdu_go',
      targetHandle: 'fcc_feed',
      data: {
        label: 'gasoil',
        stream: stream('cdu_gasoil', 40.4, { T: 650, P: 30, api: 25, S: 0.45 }),
      },
    },
    {
      id: 's-resid',
      type: 'streamEdge',
      source: 'cdu-1',
      target: 'cok-1',
      sourceHandle: 'cdu_resid',
      targetHandle: 'coker_feed',
      data: {
        label: 'resid',
        stream: stream('cdu_resid', 40.0, { T: 700, P: 25, api: 12, S: 2.5, comp: [
          { cut: 'VGO', vol_pct: 20 },
          { cut: 'Vac resid', vol_pct: 80 },
        ]}),
      },
    },
    {
      id: 's-heavy-naph',
      type: 'streamEdge',
      source: 'cdu-1',
      target: 'ref-1',
      sourceHandle: 'cdu_naph_h',
      targetHandle: 'ref_feed',
      data: {
        label: 'heavy naph',
        stream: stream('cdu_naphtha_heavy', 12.0, { T: 280, P: 40, api: 55, S: 0.01, ron: 55 }),
      },
    },
    {
      id: 's-fcc-naph',
      type: 'streamEdge',
      source: 'fcc-1',
      target: 'hdt-1',
      sourceHandle: 'fcc_naph',
      targetHandle: 'hdt_fcc_in',
      data: {
        label: 'fcc naph',
        stream: stream('fcc_naphtha', 11.3, { T: 250, P: 35, api: 50, S: 0.12, ron: 92 }),
      },
    },
    {
      id: 's-coker-naph',
      type: 'streamEdge',
      source: 'cok-1',
      target: 'hdt-1',
      sourceHandle: 'coker_naph',
      targetHandle: 'hdt_cok_in',
      data: {
        label: 'coker naph',
        stream: stream('coker_naphtha', 6.4, { T: 240, P: 30, api: 48, S: 0.8, ron: 70 }),
      },
    },
    {
      id: 's-hdt-naph',
      type: 'streamEdge',
      source: 'hdt-1',
      target: 'bl-1',
      sourceHandle: 'hdt_out',
      targetHandle: 'bl_hdt',
      data: {
        label: 'hdt naph',
        stream: stream('hdt_naphtha', 17.3, { T: 200, P: 45, api: 52, S: 0.008, ron: 88 }),
      },
    },
    {
      id: 's-reformate',
      type: 'streamEdge',
      source: 'ref-1',
      target: 'bl-1',
      sourceHandle: 'reformate',
      targetHandle: 'bl_ref',
      data: {
        label: 'reformate',
        stream: stream('reformate', 10.3, { T: 220, P: 50, api: 45, S: 0.0005, ron: 100 }),
      },
    },
    {
      id: 's-lco-slurry',
      type: 'streamEdge',
      source: 'fcc-1',
      target: 'bl-1',
      sourceHandle: 'fcc_lco',
      targetHandle: 'bl_lco',
      data: {
        label: 'LCO/slurry',
        stream: stream('fcc_lco', 14.5, { T: 450, P: 25, api: 22, S: 0.6 }),
      },
    },
    {
      id: 's-coker-go',
      type: 'streamEdge',
      source: 'cok-1',
      target: 'bl-1',
      sourceHandle: 'coker_go',
      targetHandle: 'bl_cgo',
      data: {
        label: 'coker GO',
        stream: stream('coker_gasoil', 21.5, { T: 500, P: 25, api: 20, S: 1.5 }),
      },
    },
  ];

  return { nodes, edges };
}

export function emptyStream(name = 'new_stream') {
  return {
    name,
    flow_kbd: 0,
    temperature_f: 100,
    pressure_psig: 14.7,
    vapor_fraction: 0,
    density_api: 30,
    sulfur_wt: 0.1,
    ron: null,
    composition: [
      { cut: 'C5-C6', vol_pct: 25 },
      { cut: 'C7-C8', vol_pct: 25 },
      { cut: 'C9-C10', vol_pct: 25 },
      { cut: 'C11+', vol_pct: 25 },
    ],
  };
}
