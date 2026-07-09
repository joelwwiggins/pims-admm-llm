<script>
  import { onMount } from 'svelte';
  import {
    SvelteFlow,
    Background,
    Controls,
    MiniMap,
    addEdge,
    useSvelteFlow,
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  import UnitNode from './lib/UnitNode.svelte';
  import { UNIT_PALETTE } from './lib/palette.js';
  import { postGraph, postConnect, getRouting, getHealth } from './lib/api.js';

  let nodes = $state([]);
  let edges = $state([]);
  let status = $state('boot');
  let lastRebuild = $state(null);
  let idSeq = 1;

  const nodeTypes = { unit: UnitNode };

  onMount(async () => {
    try {
      const h = await getHealth();
      status = h?.status || 'ok';
    } catch (e) {
      status = 'api-offline';
    }
  });

  function onDragStart(event, unit) {
    event.dataTransfer.setData('application/pims-unit', JSON.stringify(unit));
    event.dataTransfer.effectAllowed = 'move';
  }

  function onDragOver(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }

  function onDrop(event) {
    event.preventDefault();
    const raw = event.dataTransfer.getData('application/pims-unit');
    if (!raw) return;
    const unit = JSON.parse(raw);
    const bounds = event.currentTarget.getBoundingClientRect();
    const position = {
      x: event.clientX - bounds.left - 80,
      y: event.clientY - bounds.top - 24,
    };
    const id = `${unit.type}_${idSeq++}`;
    nodes = [
      ...nodes,
      {
        id,
        type: 'unit',
        position,
        data: {
          label: unit.label,
          unitType: unit.type,
          submodel: unit.submodel || 'lp',
          active: true,
          category: unit.category || 'process',
        },
      },
    ];
    scheduleRebuild();
  }

  async function onConnect(connection) {
    const src = nodes.find((n) => n.id === connection.source);
    const tgt = nodes.find((n) => n.id === connection.target);
    try {
      const res = await postConnect({
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle,
        targetHandle: connection.targetHandle,
        sourceType: src?.data?.unitType,
        targetType: tgt?.data?.unitType,
      });
      if (!res.allowed) {
        status = `reject: ${res.reason || 'incompatible'}`;
        return;
      }
      edges = addEdge({ ...connection, id: `e_${connection.source}_${connection.target}_${edges.length}` }, edges);
      status = `connect ok score=${res.score ?? 1}`;
      scheduleRebuild();
    } catch (e) {
      // offline: still allow local edge
      edges = addEdge(connection, edges);
      status = 'connect local (api offline)';
      scheduleRebuild();
    }
  }

  let rebuildTimer;
  function scheduleRebuild() {
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(runRebuild, 250);
  }

  async function runRebuild() {
    try {
      const res = await postGraph({ nodes, edges });
      lastRebuild = res;
      status = res?.admm_status || 'rebuilt';
    } catch (e) {
      lastRebuild = { ok: false, error: String(e) };
      status = 'rebuild failed / offline';
    }
  }

  function onNodesChange(changes) {
    // apply simple position/remove
    for (const ch of changes) {
      if (ch.type === 'remove') {
        nodes = nodes.filter((n) => n.id !== ch.id);
        edges = edges.filter((e) => e.source !== ch.id && e.target !== ch.id);
      } else if (ch.type === 'position' && ch.position) {
        nodes = nodes.map((n) => (n.id === ch.id ? { ...n, position: ch.position } : n));
      }
    }
  }

  function onEdgesChange(changes) {
    for (const ch of changes) {
      if (ch.type === 'remove') edges = edges.filter((e) => e.id !== ch.id);
    }
  }
</script>

<div class="layout">
  <aside class="dock">
    <h1>PIMS dock</h1>
    <p class="muted">Drag units onto the canvas. Snap + ADMM rebuild via FastAPI.</p>
    <div class="palette">
      {#each UNIT_PALETTE as unit}
        <div
          class="chip"
          draggable="true"
          ondragstart={(e) => onDragStart(e, unit)}
          title={unit.type}
        >
          <span class="tag">{unit.category === 'supply_chain' ? 'SC' : 'P'}</span>
          {unit.label}
        </div>
      {/each}
    </div>
    <div class="status">
      <div><strong>status:</strong> {status}</div>
      {#if lastRebuild}
        <pre>{JSON.stringify(lastRebuild, null, 2)}</pre>
      {/if}
    </div>
  </aside>
  <main class="canvas" ondragover={onDragOver} ondrop={onDrop}>
    <SvelteFlow
      {nodes}
      {edges}
      {nodeTypes}
      onconnect={onConnect}
      onnodeschange={onNodesChange}
      onedgeschange={onEdgesChange}
      fitView
    >
      <Background />
      <Controls />
      <MiniMap />
    </SvelteFlow>
  </main>
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    height: 100vh;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .dock {
    background: #0f172a;
    color: #e2e8f0;
    padding: 1rem;
    overflow: auto;
  }
  .dock h1 {
    font-size: 1.1rem;
    margin: 0 0 0.5rem;
  }
  .muted {
    color: #94a3b8;
    font-size: 0.85rem;
  }
  .palette {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin: 1rem 0;
  }
  .chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.5rem 0.65rem;
    cursor: grab;
  }
  .chip:hover {
    border-color: #38bdf8;
  }
  .tag {
    display: inline-block;
    font-size: 0.65rem;
    background: #0369a1;
    border-radius: 4px;
    padding: 0 0.3rem;
    margin-right: 0.35rem;
  }
  .status {
    margin-top: 1rem;
    font-size: 0.75rem;
  }
  .status pre {
    background: #020617;
    padding: 0.5rem;
    border-radius: 6px;
    max-height: 240px;
    overflow: auto;
  }
  .canvas {
    height: 100%;
    background: #f8fafc;
  }
</style>
