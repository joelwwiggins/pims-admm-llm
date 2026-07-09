/** Unit palette for wave3 snap-together flowsheet. */
export const PALETTE = [
  { type: 'CDU', label: 'CDU', category: 'process', color: '#4a90d9', submodel: 'lp' },
  { type: 'FCC', label: 'FCC', category: 'process', color: '#e07a3d', submodel: 'lp' },
  { type: 'COKER', label: 'Coker', category: 'process', color: '#c44d58', submodel: 'lp' },
  { type: 'REFORMER', label: 'Reformer', category: 'process', color: '#5cb85c', submodel: 'lp' },
  { type: 'HDT_NAPH', label: 'HDT Naphtha', category: 'process', color: '#5bc0de', submodel: 'lp' },
  { type: 'BLENDER', label: 'Blender', category: 'process', color: '#9b59b6', submodel: 'lp' },
  { type: 'TANK', label: 'Tank', category: 'process', color: '#7f8c8d', submodel: 'lp' },
  { type: 'SELL', label: 'Sell', category: 'process', color: '#f0ad4e', submodel: 'lp' },
  { type: 'warehouse', label: 'Warehouse', category: 'supply_chain', color: '#6b5b95', submodel: 'lp' },
  { type: 'transport', label: 'Transport', category: 'supply_chain', color: '#88b04b', submodel: 'lp' },
];

/** Alias used by some scaffold drafts. */
export const UNIT_PALETTE = PALETTE;

export const PROCESS_UNITS = PALETTE.filter((p) => p.category === 'process');
export const SUPPLY_UNITS = PALETTE.filter((p) => p.category === 'supply_chain');

export function paletteByType(type) {
  return PALETTE.find((p) => p.type === type);
}
