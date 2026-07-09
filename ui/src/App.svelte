<script>
  import {
    SvelteFlow,
    Background,
    Controls,
    MiniMap,
    addEdge,
    BackgroundVariant,
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';

  import UnitNode from './lib/UnitNode.svelte';
  import { PROCESS_UNITS, SUPPLY_UNITS, paletteByType } from './lib/palette.js';
  import { postConnect, postGraph, getRouting, getHealth } from './lib/api.js';

  const nodeTypes = { unit: UnitNode };

  let nodes = $state.raw([
    {
      id: 'cdu-1',
      type: 'unit',
      position: { x: 80, y: 160 },
      data: {
        label: 'CDU',
        unitType: 'CDU',
        category: 'process',
        color: '#4a90d9',
        submodel: 'lp',
        active: true,
      },
    },
    {
      id: 'fcc-1',
      type: 'unit',
      position: { x: 360, y: 80 },
      data: {
        label: 'FCC',
        unitType: 'FCC',
        category: 'process',
        color: '#e07a3d',
        submodel: 'lp',
        active: true,
      },
    },
    {
      id: 'blender-1',
      type: 'unit',
      position: { x: 640, y: 160 },
      data: {
        label: 'Blender',
        unitType: 'BLENDER',
        category: 'process',
        color: '#9b59b6',
        submodel: 'lp',
        active: true,
      },
    },
  ]);

  let edges = $state.raw([]);
  let status = $state('Drag units from palette · connect handles to validate via API');
  let lastGraph = $state(null);
  let nodeSeq = $state(1);

  function unitTypeOf(nodeId) {
    const n = nodes.find((x) => x.id === nodeId);
    return n?.data?.unitType || null;
  }

  /**
   * onConnect stub: POST /api/connect for validation, then add edge if allowed.
   * @param {import('@xyflow/svelte').Connection} connection
   */
  async function onconnect(connection) {
    const sourceType = unitTypeOf(connection.source);
    const targetType = unitTypeOf(connection.target);
    status = `Validating ${sourceType || connection.source} → ${targetType || connection.target}…`;

    try {
      const res = await postConnect({
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle,
        targetHandle: connection.targetHandle,
        sourceType,
        targetType,
        portAttrs: { sourceType, targetType },
      });

      if (!res.allowed) {
        status = `Rejected (score=${res.score}): ${res.reason}`;
        return;
      }

      edges = addEdge(
        {
          ...connection,
          id: `e-${connection.source}-${connection.target}-${Date.now()}`,
          label: res.score != null ? String(Number(res.score).toFixed(2)) : undefined,
          data: { score: res.score, reason: res.reason },
        },
        edges,
      );
      status = `Connected (score=${res.score}): ${res.reason}`;
    } catch (err) {
      edges = addEdge(
        {
          ...connection,
          id: `e-${connection.source}-${connection.target}-${Date.now()}`,
          label: 'offline',
          data: { offline: true },
        },
        edges,
      );
      status = `API offline — edge added locally (${err.message})`;
    }
  }

  function makeNode(unitType, position) {
    const p = paletteByType(unitType);
    const id = `${String(unitType).toLowerCase()}-${nodeSeq++}`;
    return {
      id,
      type: 'unit',
      position,
      data: {
        label: p?.label || unitType,
        unitType,
        category: p?.category || 'process',
        color: p?.color || '#4a90d9',
        submodel: p?.submodel || 'lp',
        active: true,
      },
    };
  }

  function addFromPalette(unitType) {
    const offset = nodes.length * 24;
    nodes = [...nodes, makeNode(unitType, { x: 120 + offset, y: 80 + offset })];
    status = `Added ${unitType}`;
  }

  function ondragstart(evt, unitType) {
    evt.dataTransfer.setData('application/pims-unit', unitType);
    evt.dataTransfer.effectAllowed = 'move';
  }

  function ondragover(evt) {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'move';
  }

  function ondrop(evt) {
    evt.preventDefault();
    const unitType = evt.dataTransfer.getData('application/pims-unit');
    if (!unitType) return;
    const bounds = evt.currentTarget.getBoundingClientRect();
    const position = {
      x: evt.clientX - bounds.left - 70,
      y: evt.clientY - bounds.top - 30,
    };
    nodes = [...nodes, makeNode(unitType, position)];
    status = `Dropped ${unitType}`;
  }

  async function submitGraph() {
    status = 'POST /api/graph…';
    try {
      const payloadNodes = nodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: n.data,
      }));
      const payloadEdges = edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
        data: e.data || {},
      }));
      const res = await postGraph(payloadNodes, payloadEdges);
      lastGraph = res;
      const clusterSummary = (res.clusters || [])
        .map((c) => `${c.id}[${(c.node_ids || []).length}]`)
        .join(', ');
      status = `${res.message} | clusters: ${clusterSummary || 'none'} | admm=${res.admm_status}`;
    } catch (err) {
      status = `graph failed: ${err.message}`;
    }
  }

  async function loadRouting() {
    try {
      const [health, routing] = await Promise.all([getHealth(), getRouting()]);
      const nArcs = (routing.arcs || []).length;
      status = `API ok (${health.wave || health.ok}) · routing ${routing.version || '?'} · ${nArcs} arcs · ${
        (routing.units || []).length
      } units`;
    } catch (err) {
      status = `routing load failed: ${err.message} (start uvicorn on :8008)`;
    }
  }

  $effect(() => {
    loadRouting();
  });
</script>

<div class="layout">
  <aside class="sidebar">
    <h1>Wave3 Flowsheet</h1>
    <div class="sub">Snap-together · SvelteFlow + ADMM stubs</div>

    <div class="palette-section">
      <h2>Process units</h2>
      {#each PROCESS_UNITS as u}
        <button
          type="button"
          class="palette-item"
          draggable="true"
          ondragstart={(e) => ondragstart(e, u.type)}
          onclick={() => addFromPalette(u.type)}
        >
          <span class="dot" style:background={u.color}></span>
          {u.label}
        </button>
      {/each}
    </div>

    <div class="palette-section">
      <h2>Supply chain</h2>
      {#each SUPPLY_UNITS as u}
        <button
          type="button"
          class="palette-item"
          draggable="true"
          ondragstart={(e) => ondragstart(e, u.type)}
          onclick={() => addFromPalette(u.type)}
        >
          <span class="dot" style:background={u.color}></span>
          {u.label}
        </button>
      {/each}
    </div>

    {#if lastGraph}
      <pre class="graph-out">{JSON.stringify(lastGraph, null, 2)}</pre>
    {/if}
  </aside>

  <div class="canvas-wrap" role="presentation" {ondragover} {ondrop}>
    <div class="toolbar">
      <button type="button" onclick={submitGraph}>Submit graph</button>
      <button type="button" onclick={loadRouting}>Load routing</button>
      <div class="status">{status}</div>
    </div>

    <SvelteFlow
      bind:nodes
      bind:edges
      {nodeTypes}
      {onconnect}
      fitView
      colorMode="dark"
      defaultEdgeOptions={{ type: 'smoothstep', animated: true }}
      style="width:100%;height:100%;"
    >
      <Background variant={BackgroundVariant.Dots} gap={18} size={1} />
      <Controls />
      <MiniMap pannable zoomable />
    </SvelteFlow>
  </div>
</div>
