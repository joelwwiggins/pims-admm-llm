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
    feeds: [
      { id: 'fcc_feed', name: 'Gasoil (pool)', side: 'left', type: 'feed' },
      { id: 'fcc_feed_direct', name: 'Gasoil (direct)', side: 'left', type: 'feed' },
    ],
    products: [
      { id: 'fcc_dry', name: 'Dry Gas', side: 'top', type: 'product' },
      { id: 'fcc_lpg', name: 'LPG', side: 'top', type: 'product' },
      { id: 'fcc_naph', name: 'FCC Gaso', side: 'right', type: 'product' },
      { id: 'fcc_lco', name: 'LCO', side: 'right', type: 'product' },
      { id: 'fcc_slurry', name: 'Slurry', side: 'bottom', type: 'product' },
      { id: 'fcc_coke', name: 'Coke', side: 'bottom', type: 'product' },
    ],
  },
  COKER: {
    feeds: [
      { id: 'coker_feed', name: 'Resid (pool)', side: 'left', type: 'feed' },
      { id: 'coker_feed_direct', name: 'Resid (direct)', side: 'left', type: 'feed' },
    ],
    products: [
      { id: 'coker_dry', name: 'Dry Gas', side: 'top', type: 'product' },
      { id: 'coker_lpg', name: 'LPG', side: 'top', type: 'product' },
      { id: 'coker_naph', name: 'Coker Naph', side: 'right', type: 'product' },
      { id: 'coker_go', name: 'Coker GO', side: 'right', type: 'product' },
      { id: 'coke', name: 'Petcoke', side: 'bottom', type: 'product' },
    ],
  },
  REFORMER: {
    feeds: [
      { id: 'ref_feed', name: 'Hvy Naph (direct)', side: 'left', type: 'feed' },
      { id: 'ref_feed_pool', name: 'Hvy Naph (pool)', side: 'left', type: 'feed' },
    ],
    products: [
      { id: 'reformate', name: 'Reformate', side: 'right', type: 'product' },
      { id: 'ref_h2', name: 'H2', side: 'top', type: 'product' },
      { id: 'ref_lights', name: 'Lights', side: 'bottom', type: 'product' },
    ],
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

/** Planning-grade process conditions (HYSYS/PIMS inspector). */
export function defaultProcessConditions(unitType) {
  const table = {
    CDU: {
      flash_zone_temp_f: 680,
      overflash_frac: 0.02,
      atm_tower_pressure_psig: 5,
    },
    FCC: {
      riser_outlet_temp_f: 980,
      catalyst_to_oil: 6.5,
      catalyst_activity: 68,
      additive_rate_wt_pct_cat: 0,
      feed_preheat_temp_f: 550,
      recycle_ratio: 0,
      regenerator_temp_f: 1320,
    },
    COKER: {
      drum_outlet_temp_f: 920,
      drum_pressure_psig: 25,
      recycle_ratio: 0.15,
      cycle_time_hr: 16,
      furnace_coil_outlet_temp_f: 920,
    },
    REFORMER: {
      weighted_wait_avg_f: 940,
      pressure_psig: 150,
      h2_hc_ratio: 4.0,
      space_velocity_whsv: 1.5,
      reactor_severity_count: 3,
    },
    HDT_NAPH: {
      reactor_inlet_temp_f: 550,
      pressure_psig: 600,
      h2_partial_pressure_psia: 450,
      lhsv: 2.0,
      h2_oil_scf_bbl: 800,
    },
    BLENDER: {
      blend_mode: 'delta_base',
      ron_spec: 87,
      max_sulfur_wt: 0.01,
    },
  };
  return { ...(table[unitType] || {}) };
}

export function defaultFeedQuality(unitType) {
  const table = {
    FCC: {
      api: 25.0,
      sulfur_wt: 0.45,
      ccr_wt: 0.8,
      uop_k: 11.8,
      aniline_point_f: 160,
      metals_ni_v_ppm: 2.0,
      basic_nitrogen_ppm: 400,
    },
    COKER: {
      api: 12.0,
      sulfur_wt: 2.5,
      ccr_wt: 12.0,
      asphaltenes_wt: 8.0,
      viscosity_cst_210f: 450,
    },
    REFORMER: {
      api: 55.0,
      sulfur_wt: 0.01,
      nitrogen_ppm: 0.5,
      paraffins_vol: 0.45,
      naphthenes_vol: 0.35,
      aromatics_vol: 0.20,
      n_plus_a: 0.55,
    },
    CDU: {
      api: 32.0,
      sulfur_wt: 1.2,
      ccr_wt: 3.5,
      nitrogen_ppm: 1500,
    },
  };
  return { ...(table[unitType] || {}) };
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
        processConditions: defaultProcessConditions('CDU'),
        feedQuality: defaultFeedQuality('CDU'),
        ports: UNIT_PORTS.CDU,
      },
    },
    {
      id: 'pool-fcc',
      type: 'pfdUnit',
      position: { x: 220, y: 40 },
      data: {
        tag: 'T-210',
        label: 'Pool FCC Feed',
        unitType: 'TANK',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.TANK.header,
        accentColor: colors.TANK.accent,
        description: 'Gasoil/VGO feed pooler in front of FCC (or direct bypass)',
        charge_kbd: 40.4,
        yields: [],
        processConditions: {},
        feedQuality: {},
        ports: UNIT_PORTS.TANK,
      },
    },
    {
      id: 'pool-coker',
      type: 'pfdUnit',
      position: { x: 220, y: 400 },
      data: {
        tag: 'T-220',
        label: 'Pool Coker Feed',
        unitType: 'TANK',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.TANK.header,
        accentColor: colors.TANK.accent,
        description: 'Resid feed pooler in front of coker (or direct bypass)',
        charge_kbd: 40.0,
        yields: [],
        processConditions: {},
        feedQuality: {},
        ports: UNIT_PORTS.TANK,
      },
    },
    {
      id: 'pool-ref',
      type: 'pfdUnit',
      position: { x: 480, y: 220 },
      data: {
        tag: 'T-310',
        label: 'Pool Reformer Feed',
        unitType: 'TANK',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.TANK.header,
        accentColor: colors.TANK.accent,
        description: 'Heavy naphtha feed pooler in front of reformer (or direct)',
        charge_kbd: 12.0,
        yields: [],
        processConditions: {},
        feedQuality: {},
        ports: UNIT_PORTS.TANK,
      },
    },
    {
      id: 'fuel-gas',
      type: 'pfdUnit',
      position: { x: 600, y: -40 },
      data: {
        tag: 'U-FG',
        label: 'Fuel Gas',
        unitType: 'SELL',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.SELL.header,
        accentColor: colors.SELL.accent,
        description: 'Fuel gas sink (dry gas / offgas / lights)',
        charge_kbd: 0,
        yields: [],
        processConditions: {},
        feedQuality: {},
        ports: UNIT_PORTS.SELL,
      },
    },
    {
      id: 'lpg-1',
      type: 'pfdUnit',
      position: { x: 880, y: 40 },
      data: {
        tag: 'U-LPG',
        label: 'LPG',
        unitType: 'SELL',
        active: true,
        submodel: 'lp',
        status: 'running',
        headerColor: colors.SELL.header,
        accentColor: colors.SELL.accent,
        description: 'LPG product pool (FCC/coker/reformer lights)',
        charge_kbd: 0,
        yields: [],
        processConditions: {},
        feedQuality: {},
        ports: UNIT_PORTS.SELL,
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
          ['Dry Gas', 5.3, 2.1, 0, 0],
          ['LPG', 15.1, 6.1, 0, 0],
          ['FCC Gasoline', 41.9, 16.9, 0.005, 93],
          ['LCO', 18.8, 7.6, 0.35, 0],
          ['Slurry', 12.0, 4.8, 0.8, 0],
          ['Coke (wt%)', 5.5, 2.2, 0, 0],
        ]),
        processConditions: defaultProcessConditions('FCC'),
        feedQuality: defaultFeedQuality('FCC'),
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
          ['Dry Gas', 6.2, 2.5, 0, 0],
          ['LPG', 5.2, 2.1, 0, 0],
          ['Coker Naphtha', 12.9, 5.2, 0.25, 72],
          ['Coker GO', 43.3, 17.3, 0.4, 0],
          ['Petcoke (wt%)', 28.3, 11.3, 0, 0],
        ]),
        processConditions: defaultProcessConditions('COKER'),
        feedQuality: defaultFeedQuality('COKER'),
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
          ['H2', 3.6, 0.4, 0, 0],
          ['Lights C1-C4', 10.4, 1.2, 0, 0],
        ]),
        processConditions: defaultProcessConditions('REFORMER'),
        feedQuality: defaultFeedQuality('REFORMER'),
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
        processConditions: defaultProcessConditions('HDT_NAPH'),
        feedQuality: defaultFeedQuality('HDT_NAPH'),
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
        processConditions: defaultProcessConditions('BLENDER'),
        feedQuality: defaultFeedQuality('BLENDER'),
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
      id: 's-go-to-pool',
      type: 'streamEdge',
      source: 'cdu-1',
      target: 'pool-fcc',
      sourceHandle: 'cdu_go',
      targetHandle: 'tank_in',
      data: {
        label: 'gasoil→pool',
        stream: stream('cdu_gasoil', 40.4, { T: 650, P: 30, api: 25, S: 0.45 }),
      },
    },
    {
      id: 's-pool-to-fcc',
      type: 'streamEdge',
      source: 'pool-fcc',
      target: 'fcc-1',
      sourceHandle: 'tank_out',
      targetHandle: 'fcc_feed',
      data: {
        label: 'FCC feed (pool)',
        stream: stream('cdu_gasoil', 40.4, { T: 650, P: 30, api: 25, S: 0.45 }),
      },
    },
    {
      id: 's-resid-to-pool',
      type: 'streamEdge',
      source: 'cdu-1',
      target: 'pool-coker',
      sourceHandle: 'cdu_resid',
      targetHandle: 'tank_in',
      data: {
        label: 'resid→pool',
        stream: stream('cdu_resid', 40.0, { T: 700, P: 25, api: 12, S: 2.5 }),
      },
    },
    {
      id: 's-pool-to-coker',
      type: 'streamEdge',
      source: 'pool-coker',
      target: 'cok-1',
      sourceHandle: 'tank_out',
      targetHandle: 'coker_feed',
      data: {
        label: 'coker feed (pool)',
        stream: stream('cdu_resid', 40.0, { T: 700, P: 25, api: 12, S: 2.5 }),
      },
    },
    {
      id: 's-heavy-to-pool',
      type: 'streamEdge',
      source: 'cdu-1',
      target: 'pool-ref',
      sourceHandle: 'cdu_naph_h',
      targetHandle: 'tank_in',
      data: {
        label: 'hvy naph→pool',
        stream: stream('cdu_naphtha_heavy', 12.0, { T: 280, P: 40, api: 55, S: 0.01, ron: 55 }),
      },
    },
    {
      id: 's-pool-to-ref',
      type: 'streamEdge',
      source: 'pool-ref',
      target: 'ref-1',
      sourceHandle: 'tank_out',
      targetHandle: 'ref_feed_pool',
      data: {
        label: 'ref feed (pool)',
        stream: stream('cdu_naphtha_heavy', 12.0, { T: 280, P: 40, api: 55, S: 0.01, ron: 55 }),
      },
    },
    {
      id: 's-fcc-dry',
      type: 'streamEdge',
      source: 'fcc-1',
      target: 'fuel-gas',
      sourceHandle: 'fcc_dry',
      targetHandle: 'sell_in',
      data: {
        label: 'FCC dry gas',
        stream: stream('fcc_dry_gas', 2.1, { T: 100, P: 80, api: 0, S: 0, vf: 1 }),
      },
    },
    {
      id: 's-fcc-lpg',
      type: 'streamEdge',
      source: 'fcc-1',
      target: 'lpg-1',
      sourceHandle: 'fcc_lpg',
      targetHandle: 'sell_in',
      data: {
        label: 'FCC LPG',
        stream: stream('fcc_lpg', 6.1, { T: 80, P: 120, api: 0, S: 0, vf: 1 }),
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
        label: 'fcc gasoline',
        stream: stream('fcc_naphtha', 16.9, { T: 250, P: 35, api: 50, S: 0.12, ron: 93 }),
      },
    },
    {
      id: 's-fcc-lco',
      type: 'streamEdge',
      source: 'fcc-1',
      target: 'bl-1',
      sourceHandle: 'fcc_lco',
      targetHandle: 'bl_lco',
      data: {
        label: 'LCO',
        stream: stream('fcc_lco', 7.6, { T: 450, P: 25, api: 22, S: 0.35 }),
      },
    },
    {
      id: 's-fcc-slurry',
      type: 'streamEdge',
      source: 'fcc-1',
      target: 'bl-1',
      sourceHandle: 'fcc_slurry',
      targetHandle: 'bl_lco',
      data: {
        label: 'slurry',
        stream: stream('fcc_slurry', 4.8, { T: 500, P: 20, api: 8, S: 0.8 }),
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
        stream: stream('coker_naphtha', 5.2, { T: 240, P: 30, api: 48, S: 0.8, ron: 70 }),
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
        stream: stream('coker_gasoil', 17.3, { T: 500, P: 25, api: 20, S: 1.5 }),
      },
    },
    {
      id: 's-coker-dry',
      type: 'streamEdge',
      source: 'cok-1',
      target: 'fuel-gas',
      sourceHandle: 'coker_dry',
      targetHandle: 'sell_in',
      data: {
        label: 'coker dry gas',
        stream: stream('coker_dry_gas', 2.5, { T: 100, P: 70, api: 0, S: 0, vf: 1 }),
      },
    },
    {
      id: 's-coker-lpg',
      type: 'streamEdge',
      source: 'cok-1',
      target: 'lpg-1',
      sourceHandle: 'coker_lpg',
      targetHandle: 'sell_in',
      data: {
        label: 'coker LPG',
        stream: stream('coker_lpg', 2.1, { T: 80, P: 110, api: 0, S: 0, vf: 1 }),
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
      id: 's-ref-h2',
      type: 'streamEdge',
      source: 'ref-1',
      target: 'fuel-gas',
      sourceHandle: 'ref_h2',
      targetHandle: 'sell_in',
      data: {
        label: 'H2 make',
        stream: stream('reformer_h2', 0.4, { T: 100, P: 200, api: 0, S: 0, vf: 1 }),
      },
    },
    {
      id: 's-ref-lights',
      type: 'streamEdge',
      source: 'ref-1',
      target: 'fuel-gas',
      sourceHandle: 'ref_lights',
      targetHandle: 'sell_in',
      data: {
        label: 'ref lights',
        stream: stream('reformer_lights', 1.2, { T: 90, P: 90, api: 0, S: 0, vf: 1 }),
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
