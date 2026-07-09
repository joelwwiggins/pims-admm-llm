/** Dock palette — process + supply-chain stubs. Colors match industrial UI. */

export const PROCESS_UNITS = [
  { type: 'CDU', label: 'CDU', category: 'process', submodel: 'lp', color: '#4a90d9' },
  { type: 'FCC', label: 'FCC', category: 'process', submodel: 'lp', color: '#e07a3d' },
  { type: 'COKER', label: 'Coker', category: 'process', submodel: 'lp', color: '#c0392b' },
  { type: 'REFORMER', label: 'Reformer', category: 'process', submodel: 'lp', color: '#27ae60' },
  { type: 'HDT_NAPH', label: 'HDT Naphtha', category: 'process', submodel: 'lp', color: '#16a085' },
  { type: 'BLENDER', label: 'Blender', category: 'process', submodel: 'lp', color: '#9b59b6' },
  { type: 'TANK', label: 'Tank', category: 'process', submodel: 'lp', color: '#7f8c8d' },
  { type: 'SELL', label: 'Sell', category: 'process', submodel: 'lp', color: '#f1c40f' },
];

export const SUPPLY_UNITS = [
  { type: 'warehouse', label: 'Warehouse', category: 'supply_chain', submodel: 'lp', color: '#6b5b95' },
  { type: 'transport', label: 'Transport', category: 'supply_chain', submodel: 'lp', color: '#88b04b' },
];

export const ALL_UNITS = [...PROCESS_UNITS, ...SUPPLY_UNITS];

export function paletteByType(unitType) {
  return ALL_UNITS.find((u) => u.type === unitType) || null;
}

/** Full-plant snap template (chemical defaults, no FCC→reformer). */
export function fullPlantTemplate() {
  const mk = (id, type, x, y) => {
    const p = paletteByType(type);
    return {
      id,
      type: 'unit',
      position: { x, y },
      data: {
        label: p?.label || type,
        unitType: type,
        category: p?.category || 'process',
        color: p?.color || '#4a90d9',
        submodel: 'lp',
        active: true,
      },
    };
  };
  const nodes = [
    mk('cdu-1', 'CDU', 40, 200),
    mk('fcc-1', 'FCC', 320, 60),
    mk('cok-1', 'COKER', 320, 340),
    mk('ref-1', 'REFORMER', 560, 200),
    mk('hdt-1', 'HDT_NAPH', 560, 60),
    mk('bl-1', 'BLENDER', 800, 200),
  ];
  const edges = [
    { id: 'e-cdu-fcc', source: 'cdu-1', target: 'fcc-1', label: 'gasoil', type: 'smoothstep', animated: true },
    { id: 'e-cdu-cok', source: 'cdu-1', target: 'cok-1', label: 'resid', type: 'smoothstep', animated: true },
    { id: 'e-cdu-ref', source: 'cdu-1', target: 'ref-1', label: 'heavy naph', type: 'smoothstep', animated: true },
    { id: 'e-fcc-hdt', source: 'fcc-1', target: 'hdt-1', label: 'fcc naph', type: 'smoothstep', animated: true },
    { id: 'e-cok-hdt', source: 'cok-1', target: 'hdt-1', label: 'coker naph', type: 'smoothstep', animated: true },
    { id: 'e-hdt-bl', source: 'hdt-1', target: 'bl-1', label: 'hdt naph', type: 'smoothstep', animated: true },
    { id: 'e-ref-bl', source: 'ref-1', target: 'bl-1', label: 'reformate', type: 'smoothstep', animated: true },
    { id: 'e-fcc-bl', source: 'fcc-1', target: 'bl-1', label: 'LCO/slurry', type: 'smoothstep', animated: true },
    { id: 'e-cok-bl', source: 'cok-1', target: 'bl-1', label: 'coker GO', type: 'smoothstep', animated: true },
  ];
  return { nodes, edges };
}
