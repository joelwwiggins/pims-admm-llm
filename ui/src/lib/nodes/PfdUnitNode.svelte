<script>
  import { Handle, Position, NodeToolbar } from '@xyflow/svelte';

  /** @type {{ id: string, data: Record<string, any>, selected?: boolean }} */
  let { id, data, selected = false } = $props();

  const ports = $derived(data.ports || { feeds: [], products: [] });
  const feeds = $derived(ports.feeds || []);
  const products = $derived(ports.products || []);
  const yields = $derived(data.yields || []);
  const showYields = $derived(
    ['FCC', 'COKER', 'REFORMER', 'CDU', 'HDT_NAPH'].includes(data.unitType) && yields.length > 0,
  );
  const agentStatus = $derived(data.agentStatus || data.status);
  const statusColor = $derived(
    !data.active
      ? '#666'
      : agentStatus === 'alarm' || agentStatus === 'pushback' || agentStatus === 'critical'
        ? (data.badgeColor || '#e05a5a')
        : agentStatus === 'watch'
          ? (data.badgeColor || '#e0c040')
          : agentStatus === 'ok' || agentStatus === 'running'
            ? (data.badgeColor || '#3dba72')
            : '#e0c040',
  );
  const agentBadge = $derived(data.agentSeverity || data.wiggle_room || null);
  const tip = $derived(
    `${data.tag || id} ${data.label || data.unitType}` +
      (data.charge_kbd ? ` · ${Number(data.charge_kbd).toFixed(1)} kbd` : '') +
      (data.agentSummary ? ` · ${data.agentSummary}` : '') +
      (data.wiggle_room ? ` · wiggle=${data.wiggle_room}` : '') +
      (data.description ? ` · ${data.description}` : ''),
  );

  function pos(side) {
    if (side === 'left') return Position.Left;
    if (side === 'right') return Position.Right;
    if (side === 'bottom') return Position.Bottom;
    return Position.Top;
  }

  function stackStyle(list, i, side) {
    const n = list.length || 1;
    const pct = ((i + 1) / (n + 1)) * 100;
    if (side === 'left' || side === 'right') {
      return `top: ${pct}%; transform: translateY(-50%);`;
    }
    return `left: ${pct}%; transform: translateX(-50%);`;
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="pfd-node"
  class:selected
  class:inactive={!data.active}
  style:--header={data.headerColor || '#1e5a8a'}
  style:--accent={data.accentColor || '#4a9fe0'}
  title={tip}
>
  <NodeToolbar isVisible={selected} position={Position.Top} offset={8}>
    <div class="toolbar-tip">
      <strong>{data.tag} {data.label}</strong>
      <span>{data.description || data.unitType}</span>
      {#if data.charge_kbd}
        <span>Charge {Number(data.charge_kbd).toFixed(1)} kbd</span>
      {/if}
      <span class="hint">Inspector → right panel</span>
    </div>
  </NodeToolbar>

  {#each feeds as port, i}
    <Handle
      type="target"
      position={pos(port.side || 'left')}
      id={port.id}
      class="pfd-handle"
      style={stackStyle(feeds, i, port.side || 'left')}
      title={`Feed: ${port.name}`}
    />
    {#if (port.side || 'left') === 'left'}
      <span class="port-label in" style={stackStyle(feeds, i, 'left')}>{port.name}</span>
    {/if}
  {/each}

  <div class="header">
    <div class="tag">{data.tag || id}</div>
    <div class="title">{data.label || data.unitType}</div>
    <div class="status" title={data.active ? (data.agentSummary || data.status || 'active') : 'inactive'}>
      <span class="led" style:background={statusColor}></span>
      {data.active ? (agentStatus === 'alarm' ? 'ALERT' : agentStatus === 'watch' ? 'WATCH' : agentStatus === 'ok' ? 'OK' : (data.status || 'ON')) : 'OFF'}
    </div>
  </div>

  {#if agentBadge && data.active}
    <div
      class="agent-badge"
      class:alarm={agentStatus === 'alarm' || agentStatus === 'pushback'}
      class:watch={agentStatus === 'watch'}
      class:ok={agentStatus === 'ok'}
      title={data.agentSummary || agentBadge}
    >
      {#if data.n_pushbacks}
        {data.n_pushbacks} PB
      {:else if data.wiggle_room === 'none'}
        NO WIGGLE
      {:else}
        {String(agentBadge).toUpperCase().slice(0, 8)}
      {/if}
    </div>
  {/if}

  <div class="body">
    <div class="unit-type">{data.unitType}</div>
    {#if data.charge_kbd}
      <div class="charge">Charge <strong>{Number(data.charge_kbd).toFixed(1)}</strong> kbd</div>
    {/if}
    {#if data.wiggle_room}
      <div class="wiggle">wiggle: <strong>{data.wiggle_room}</strong></div>
    {/if}

    {#if showYields}
      <div class="yield-box">
        <div class="yield-head">Yields</div>
        <table>
          <tbody>
            {#each yields.slice(0, 4) as y}
              <tr>
                <td class="prod">{y.product}</td>
                <td class="num">{Number(y.yield_pct).toFixed(1)}%</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>

  {#each products as port, i}
    <Handle
      type="source"
      position={pos(port.side || 'right')}
      id={port.id}
      class="pfd-handle"
      style={stackStyle(products, i, port.side || 'right')}
      title={`Product: ${port.name}`}
    />
    {#if (port.side || 'right') === 'right'}
      <span class="port-label out" style={stackStyle(products, i, 'right')}>{port.name}</span>
    {/if}
  {/each}
</div>

<style>
  .pfd-node {
    position: relative;
    min-width: 168px;
    max-width: 200px;
    background: linear-gradient(180deg, #1c2634 0%, #141c28 100%);
    border: 1.5px solid #3a4d64;
    border-radius: 6px;
    box-shadow:
      0 4px 16px rgba(0, 0, 0, 0.45),
      inset 0 1px 0 rgba(255, 255, 255, 0.04);
    color: #e8eef5;
    font-size: 0.72rem;
  }
  .pfd-node.selected {
    border-color: var(--accent);
    box-shadow:
      0 0 0 1px var(--accent),
      0 6px 20px rgba(0, 0, 0, 0.5);
  }
  .pfd-node.inactive {
    opacity: 0.55;
    filter: grayscale(0.4);
  }
  .agent-badge {
    position: absolute;
    top: -8px;
    right: -6px;
    z-index: 5;
    font-size: 0.55rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid #0006;
    box-shadow: 0 2px 8px #0008;
  }
  .agent-badge.alarm {
    background: #a02222;
    color: #ffd0d0;
  }
  .agent-badge.watch {
    background: #8a7010;
    color: #fff3c0;
  }
  .agent-badge.ok {
    background: #1a6b3a;
    color: #c8ffd8;
  }
  .wiggle {
    margin-top: 2px;
    font-size: 0.6rem;
    color: #8a9bb0;
  }
  .wiggle strong {
    color: #e0c040;
  }

  .toolbar-tip {
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: #0d1520f2;
    border: 1px solid #3a5a78;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.68rem;
    color: #c5d2e0;
    box-shadow: 0 4px 16px #0008;
    max-width: 220px;
  }
  .toolbar-tip strong {
    color: #fff;
    font-size: 0.75rem;
  }
  .toolbar-tip .hint {
    color: #5eb0f0;
    font-size: 0.6rem;
    margin-top: 2px;
  }

  .header {
    background: linear-gradient(180deg, color-mix(in srgb, var(--header) 90%, #fff 5%), var(--header));
    border-radius: 4px 4px 0 0;
    padding: 6px 10px 5px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.35);
  }
  .tag {
    font-size: 0.62rem;
    color: rgba(255, 255, 255, 0.65);
    font-family: ui-monospace, monospace;
    letter-spacing: 0.04em;
  }
  .title {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    line-height: 1.2;
  }
  .status {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-top: 2px;
    font-size: 0.6rem;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.8);
  }
  .led {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    box-shadow: 0 0 6px currentColor;
  }

  .body {
    padding: 6px 10px 8px;
  }
  .unit-type {
    color: #7d90a8;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .charge {
    margin-top: 2px;
    color: #b8c8d8;
  }
  .charge strong {
    color: #fff;
    font-variant-numeric: tabular-nums;
  }

  .yield-box {
    margin-top: 6px;
    background: #0d131c;
    border: 1px solid #2a384c;
    border-radius: 4px;
    padding: 4px 6px;
  }
  .yield-head {
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6a7e96;
    margin-bottom: 2px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  td {
    padding: 1px 0;
    font-size: 0.62rem;
  }
  td.prod {
    color: #c5d2e0;
    max-width: 90px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: #9ec9ef;
    font-family: ui-monospace, monospace;
  }

  :global(.pfd-handle) {
    width: 10px !important;
    height: 10px !important;
    background: #1a2330 !important;
    border: 2px solid var(--accent, #4a9fe0) !important;
    border-radius: 2px !important;
  }
  :global(.pfd-handle:hover) {
    background: var(--accent, #4a9fe0) !important;
  }

  .port-label {
    position: absolute;
    font-size: 0.52rem;
    color: #6a7e96;
    pointer-events: none;
    white-space: nowrap;
    z-index: 2;
  }
  .port-label.in {
    left: 12px;
  }
  .port-label.out {
    right: 12px;
    text-align: right;
  }
</style>
