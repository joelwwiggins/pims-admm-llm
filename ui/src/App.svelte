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
  import ResultsPanel from './lib/ResultsPanel.svelte';
  import { PROCESS_UNITS, SUPPLY_UNITS, paletteByType, fullPlantTemplate } from './lib/palette.js';
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
  let status = $state('Drag units · connect handles · Solve graph → real LP/ADMM');
  let lastGraph = $state(null);
  let nodeSeq = $state(10);
  let solving = $state(false);
  let apiOk = $state(false);
  let routingInfo = $state(null);

  // Solve options (issue #1 / wave3b)
  let recoveryPath = $state('mono-oracle'); // mono-oracle | pure-admm
  let inventoryMode = $state(false);
  let runAdmm = $state(true);
  let stubOnly = $state(false);

  function unitTypeOf(nodeId) {
    const n = nodes.find((x) => x.id === nodeId);
    return n?.data?.unitType || null;
  }

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
          type: 'smoothstep',
          animated: true,
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
          type: 'smoothstep',
          animated: true,
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

  function loadFullPlantTemplate() {
    const t = fullPlantTemplate();
    nodes = t.nodes;
    edges = t.edges;
    nodeSeq = 20;
    lastGraph = null;
    status = 'Loaded full-plant template (chem defaults: FCC/coker naph → HDT/gas, not reformer)';
  }

  function clearCanvas() {
    nodes = [];
    edges = [];
    lastGraph = null;
    status = 'Canvas cleared';
  }

  async function submitGraph() {
    solving = true;
    status = stubOnly ? 'POST /api/graph (stub)…' : `POST /api/graph (${recoveryPath})…`;
    try {
      const payloadNodes = nodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: { ...n.data },
      }));
      const payloadEdges = edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
        data: e.data || {},
      }));
      const res = await postGraph({
        nodes: payloadNodes,
        edges: payloadEdges,
        recovery_path: recoveryPath,
        inventory_mode: inventoryMode,
        run_admm: runAdmm,
        stub_only: stubOnly,
      });
      lastGraph = res;
      const obj = res.objective != null ? Number(res.objective).toFixed(2) : '—';
      const path = res.admm?.dual_recovery_path || res.admm_status || '—';
      status = `${res.feasible || res.ok ? 'OK' : 'FAIL'} obj=${obj} path=${path} · ${res.message || ''}`;
    } catch (err) {
      status = `graph failed: ${err.message}`;
    } finally {
      solving = false;
    }
  }

  async function loadRouting() {
    try {
      const [health, routing] = await Promise.all([getHealth(), getRouting()]);
      apiOk = !!health.ok;
      routingInfo = routing;
      const nArcs = (routing.arcs || []).length;
      status = `API ok (${health.wave || health.ok}) · routing ${routing.version || '?'} · ${nArcs} arcs · ${
        (routing.units || []).length
      } units`;
    } catch (err) {
      apiOk = false;
      status = `routing load failed: ${err.message} (start: uvicorn api.main:app --reload --port 8008)`;
    }
  }

  $effect(() => {
    loadRouting();
  });
</script>

<div class="layout">
  <aside class="sidebar">
    <h1>PIMS Flowsheet</h1>
    <div class="sub">
      Wave3 · SvelteFlow
      <span class="pill" class:on={apiOk} class:off={!apiOk}>{apiOk ? 'API' : 'API off'}</span>
    </div>

    <div class="controls">
      <label>
        recovery
        <select bind:value={recoveryPath}>
          <option value="mono-oracle">mono-oracle (L∞=0 duals)</option>
          <option value="pure-admm">pure-admm (free λ)</option>
        </select>
      </label>
      <label class="chk">
        <input type="checkbox" bind:checked={inventoryMode} />
        inventory mode
      </label>
      <label class="chk">
        <input type="checkbox" bind:checked={runAdmm} />
        run ADMM metrics
      </label>
      <label class="chk">
        <input type="checkbox" bind:checked={stubOnly} />
        stub only (no LP)
      </label>
    </div>

    <div class="btn-row">
      <button type="button" class="primary" disabled={solving} onclick={submitGraph}>
        {solving ? 'Solving…' : 'Solve graph'}
      </button>
      <button type="button" onclick={loadFullPlantTemplate}>Full plant</button>
      <button type="button" onclick={clearCanvas}>Clear</button>
      <button type="button" onclick={loadRouting}>Ping API</button>
    </div>

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

    <div class="palette-section">
      <h2>Results</h2>
      <ResultsPanel result={lastGraph} />
    </div>

    {#if routingInfo?.chemical_defaults}
      <details class="chem">
        <summary>Chemical defaults</summary>
        <pre>{JSON.stringify(routingInfo.chemical_defaults, null, 2)}</pre>
      </details>
    {/if}
  </aside>

  <div class="canvas-wrap" role="presentation" {ondragover} {ondrop}>
    <div class="toolbar">
      <div class="status">{status}</div>
      <div class="counts">{nodes.length} nodes · {edges.length} edges</div>
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
