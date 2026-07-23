<script>
  import {
    SvelteFlow,
    Background,
    Controls,
    MiniMap,
    BackgroundVariant,
    MarkerType,
    ConnectionMode,
  } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';

  import PfdUnitNode from './lib/nodes/PfdUnitNode.svelte';
  import StreamEdge from './lib/edges/StreamEdge.svelte';
  import InspectorPanel from './lib/InspectorPanel.svelte';
  import ResultsPanel from './lib/ResultsPanel.svelte';
  import ExcelPanel from './lib/ExcelPanel.svelte';
  import { createFullPlantPfd, UNIT_COLORS, UNIT_PORTS } from './lib/data/plantTemplate.js';
  import { PROCESS_UNITS, SUPPLY_UNITS, paletteByType } from './lib/palette.js';
  import { postConnect, postGraph, postAutoWire, postBaseDeltaSolve, getRouting, getHealth } from './lib/api.js';
  import { emptyStream } from './lib/data/plantTemplate.js';
  import {
    activeUnitTypes,
    edgesForApi,
    applyAutoWireToGraph,
  } from './lib/autoWire.js';

  const nodeTypes = { pfdUnit: PfdUnitNode };
  const edgeTypes = { streamEdge: StreamEdge };

  const plant = createFullPlantPfd();
  let nodes = $state.raw(plant.nodes);
  let edges = $state.raw(plant.edges);

  let selection = $state(null); // { kind, id, data }
  let status = $state('HYSYS-style PFD · click unit or stream to inspect');
  let solving = $state(false);
  let autoWiring = $state(false);
  let apiOk = $state(false);
  let lastGraph = $state(null);
  let showResults = $state(false);
  /** left dock: palette | excel */
  let leftTab = $state('palette');
  let nodeSeq = $state(50);

  let recoveryPath = $state('mono-oracle');
  let inventoryMode = $state(false);
  let runAdmm = $state(true);
  let closedLoop = $state(true);

  const defaultEdgeOptions = {
    type: 'streamEdge',
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 18,
      height: 18,
      color: '#6a8fb0',
    },
  };

  function findNode(id) {
    return nodes.find((n) => n.id === id);
  }
  function findEdge(id) {
    return edges.find((e) => e.id === id);
  }

  function onNodeClick({ node }) {
    selection = { kind: 'node', id: node.id, data: { ...node.data } };
    status = `Unit ${node.data?.tag || ''} ${node.data?.label || node.id}`;
  }

  function onEdgeClick({ edge }) {
    selection = {
      kind: 'edge',
      id: edge.id,
      data: { label: edge.data?.label, stream: { ...(edge.data?.stream || {}) } },
    };
    status = `Stream ${edge.data?.label || edge.id}`;
  }

  function onPaneClick() {
    selection = null;
  }

  function onUpdateNode(id, patch) {
    nodes = nodes.map((n) =>
      n.id === id ? { ...n, data: { ...n.data, ...patch } } : n,
    );
    if (selection?.kind === 'node' && selection.id === id) {
      selection = { ...selection, data: { ...selection.data, ...patch } };
    }
  }

  function onUpdateEdge(id, patch) {
    edges = edges.map((e) => {
      if (e.id !== id) return e;
      const data = { ...e.data, ...patch };
      if (patch.stream) data.stream = { ...e.data?.stream, ...patch.stream };
      return { ...e, data, label: data.label };
    });
    if (selection?.kind === 'edge' && selection.id === id) {
      selection = {
        ...selection,
        data: {
          ...selection.data,
          ...patch,
          stream: { ...selection.data.stream, ...(patch.stream || {}) },
        },
      };
    }
  }

  async function onconnect(connection) {
    const src = findNode(connection.source);
    const tgt = findNode(connection.target);
    const sourceType = src?.data?.unitType;
    const targetType = tgt?.data?.unitType;
    status = `Validating ${sourceType} → ${targetType}…`;

    let allowed = true;
    let reason = 'local';
    let score = 0.7;
    try {
      const streamGuess =
        connection.sourceHandle ||
        connection.targetHandle ||
        undefined;
      const res = await postConnect({
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle,
        targetHandle: connection.targetHandle,
        sourceType,
        targetType,
        stream: streamGuess,
      });
      allowed = res.allowed;
      reason = res.reason;
      score = res.score;
      if (!allowed) {
        status = `Rejected: ${reason}`;
        return;
      }
    } catch (err) {
      reason = `offline (${err.message})`;
    }

    const label = connection.sourceHandle || connection.targetHandle || 'stream';
    const id = `s-${connection.source}-${connection.target}-${Date.now()}`;
    edges = [
      ...edges,
      {
        id,
        type: 'streamEdge',
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle,
        targetHandle: connection.targetHandle,
        markerEnd: defaultEdgeOptions.markerEnd,
        data: {
          label: String(label).replace(/_/g, ' '),
          stream: emptyStream(String(label)),
          score,
          reason,
        },
      },
    ];
    status = `Connected: ${reason}`;
  }

  function makePfdNode(unitType, position) {
    const colors = UNIT_COLORS[unitType] || UNIT_COLORS.CDU;
    const p = paletteByType(unitType);
    const id = `${String(unitType).toLowerCase()}-${nodeSeq++}`;
    return {
      id,
      type: 'pfdUnit',
      position,
      data: {
        tag: `U-${nodeSeq}`,
        label: p?.label || unitType,
        unitType,
        active: true,
        submodel: 'lp',
        status: 'idle',
        headerColor: colors.header,
        accentColor: colors.accent,
        description: '',
        charge_kbd: 0,
        yields: [],
        ports: UNIT_PORTS[unitType] || UNIT_PORTS.TANK,
      },
    };
  }

  /** Call /api/auto_wire after a process unit is dropped/added; map edges onto canvas. */
  async function runAutoWire(hintType = '') {
    const units = activeUnitTypes(nodes);
    // Base-delta cascade only auto-wires conversion units we support
    const wireUnits = units.filter((u) =>
      ['CDU', 'FCC', 'COKER', 'REFORMER', 'HDT_NAPH', 'BLENDER'].includes(u),
    );
    if (!wireUnits.length) return;
    autoWiring = true;
    status = `Auto-wiring streams for ${hintType || wireUnits.join('+')}…`;
    try {
      const res = await postAutoWire({
        active_units: wireUnits,
        existing_edges: edgesForApi(edges),
      });
      if (!res.ok) {
        status = `Auto-wire failed: ${res.error || 'unknown'}`;
        return;
      }
      const applied = applyAutoWireToGraph(nodes, edges, res.edges || []);
      nodes = applied.nodes;
      edges = applied.edges;
      const nAdd = applied.added.length;
      const nTerm = applied.createdTerminals.length;
      status =
        nAdd > 0
          ? `Auto-wired ${nAdd} stream(s)` +
            (nTerm ? ` · created ${nTerm} terminal(s)` : '') +
            (hintType ? ` after adding ${hintType}` : '')
          : `No new wires (already connected)` + (hintType ? ` · ${hintType}` : '');
    } catch (err) {
      status = `Auto-wire offline: ${err.message}`;
    } finally {
      autoWiring = false;
    }
  }

  async function addFromPalette(unitType) {
    const offset = (nodes.length % 6) * 28;
    nodes = [...nodes, makePfdNode(unitType, { x: 100 + offset, y: 80 + offset })];
    status = `Added ${unitType}`;
    // Process conversion units trigger auto_wire (esp. COKER / FCC)
    if (['CDU', 'FCC', 'COKER', 'REFORMER', 'HDT_NAPH'].includes(unitType)) {
      await runAutoWire(unitType);
    }
  }

  function ondragstart(evt, unitType) {
    evt.dataTransfer.setData('application/pims-unit', unitType);
    evt.dataTransfer.effectAllowed = 'move';
  }
  function ondragover(evt) {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'move';
  }
  async function ondrop(evt) {
    evt.preventDefault();
    const unitType = evt.dataTransfer.getData('application/pims-unit');
    if (!unitType) return;
    const bounds = evt.currentTarget.getBoundingClientRect();
    nodes = [
      ...nodes,
      makePfdNode(unitType, {
        x: evt.clientX - bounds.left - 80,
        y: evt.clientY - bounds.top - 40,
      }),
    ];
    status = `Dropped ${unitType}`;
    if (['CDU', 'FCC', 'COKER', 'REFORMER', 'HDT_NAPH'].includes(unitType)) {
      await runAutoWire(unitType);
    }
  }

  async function baseDeltaSolve() {
    solving = true;
    status = 'Base-delta cascade solve…';
    try {
      const units = activeUnitTypes(nodes);
      const res = await postBaseDeltaSolve({
        active_units: units.length ? units : ['CDU', 'FCC'],
        enable_coker: units.includes('COKER'),
        max_crude_kbd: 100,
        drawn_edges: edgesForApi(edges),
      });
      lastGraph = {
        ...res,
        admm_status: 'base-delta',
        feasible: res.status === 'Optimal',
        ok: res.ok,
        message: `base-delta ${res.status} mb_ok=${res.mass_balance?.ok}`,
      };
      showResults = true;
      const obj = res.objective != null ? Number(res.objective).toFixed(2) : '—';
      status = `Base-delta ${res.status} · obj ${obj} · mb ${res.mass_balance?.ok ? 'ok' : 'FAIL'} · ${res.enabled_units?.join('+') || ''}`;
    } catch (err) {
      status = `Base-delta failed: ${err.message}`;
    } finally {
      solving = false;
    }
  }

  function loadFullPlant() {
    const t = createFullPlantPfd();
    nodes = t.nodes;
    edges = t.edges;
    selection = null;
    lastGraph = null;
    status = 'Loaded full-plant PFD template';
  }

  function clearCanvas() {
    nodes = [];
    edges = [];
    selection = null;
    lastGraph = null;
    status = 'Canvas cleared';
  }

  function applyNodeBadges(badges) {
    if (!badges || typeof badges !== 'object') return;
    nodes = nodes.map((n) => {
      const ut = n.data?.unitType;
      const b = badges[ut];
      if (!b) {
        return {
          ...n,
          data: {
            ...n.data,
            agentStatus: n.data?.active === false ? n.data.status : 'ok',
            agentSeverity: null,
            wiggle_room: null,
            agentSummary: null,
            n_pushbacks: 0,
            badgeColor: null,
          },
        };
      }
      return {
        ...n,
        data: {
          ...n.data,
          agentStatus: b.status || 'ok',
          agentSeverity: b.severity,
          wiggle_room: b.wiggle_room,
          agentSummary: b.summary,
          n_pushbacks: b.n_pushbacks || 0,
          badgeColor: b.badge_color,
          status: b.status === 'alarm' ? 'alarm' : b.status === 'watch' ? 'watch' : 'running',
        },
      };
    });
  }

  async function solve() {
    solving = true;
    status = `Solving (${recoveryPath}${closedLoop ? ' + agents' : ''})…`;
    try {
      const res = await postGraph({
        nodes: nodes.map((n) => ({
          id: n.id,
          type: n.type,
          position: n.position,
          data: n.data,
        })),
        edges: edges.map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle,
          targetHandle: e.targetHandle,
          data: e.data || {},
        })),
        recovery_path: recoveryPath,
        inventory_mode: inventoryMode,
        run_admm: runAdmm,
        stub_only: false,
        process_network: true,
        closed_loop: closedLoop,
        max_agent_rounds: 3,
      });
      lastGraph = res;
      showResults = true;
      applyNodeBadges(res.node_badges || res.process_network?.node_badges);
      const obj = res.objective != null ? Number(res.objective).toFixed(2) : '—';
      const pn = res.process_network;
      const pnNote = pn?.applied
        ? ` · agents r${pn.n_rounds || '?'}/${pn.max_rounds || 3} Δobj=${Number(pn.delta?.delta_obj || 0).toFixed(1)} → ${pn.recommended_plan}`
        : pn?.baseline
          ? ` · agents ${pn.baseline.severity || ''}`
          : '';
      status = `${res.feasible || res.ok ? 'Optimal' : 'Fail'} · obj ${obj} · ${res.admm_status || ''}${pnNote}`;
    } catch (err) {
      status = `Solve failed: ${err.message}`;
    } finally {
      solving = false;
    }
  }

  async function pingApi() {
    try {
      const h = await getHealth();
      apiOk = !!h.ok;
      const r = await getRouting();
      status = `API ok · routing ${r.version || '?'} · ${(r.arcs || []).length} arcs`;
    } catch (e) {
      apiOk = false;
      status = `API offline: ${e.message}`;
    }
  }

  $effect(() => {
    pingApi();
  });
</script>

<div class="pfd-app">
  <!-- TOP TOOLBAR -->
  <header class="toolbar">
    <div class="brand">
      <span class="logo">PFD</span>
      <div>
        <div class="brand-title">PIMS-ADMM Flowsheet</div>
        <div class="brand-sub">HYSYS-style · Wave3 superstructure</div>
      </div>
      <span class="pill" class:on={apiOk} class:off={!apiOk}>{apiOk ? 'API' : 'API off'}</span>
    </div>

    <div class="actions">
      <select bind:value={recoveryPath} title="ADMM dual recovery path">
        <option value="mono-oracle">mono-oracle</option>
        <option value="pure-admm">pure-admm</option>
      </select>
      <label class="chk"><input type="checkbox" bind:checked={inventoryMode} /> inventory</label>
      <label class="chk"><input type="checkbox" bind:checked={runAdmm} /> ADMM</label>
      <label class="chk" title="Process-network agents + closed-loop replan from pushbacks">
        <input type="checkbox" bind:checked={closedLoop} /> agents
      </label>
      <button type="button" class="primary" disabled={solving} onclick={solve}>
        {solving ? 'Running…' : 'Run'}
      </button>
      <button type="button" disabled={solving} onclick={baseDeltaSolve} title="CDU→FCC[+COKER] base-delta LP">
        Base-δ
      </button>
      <button type="button" disabled={autoWiring} onclick={() => runAutoWire('manual')} title="POST /api/auto_wire">
        {autoWiring ? 'Wiring…' : 'Auto-wire'}
      </button>
      <button type="button" onclick={pingApi}>Validate</button>
      <button type="button" onclick={loadFullPlant}>Reset PFD</button>
      <button type="button" onclick={clearCanvas}>Clear</button>
      <button type="button" class:on={showResults} onclick={() => (showResults = !showResults)}>
        Results
      </button>
      <button
        type="button"
        class:on={leftTab === 'excel'}
        title="Excel PIMS → mono+ADMM"
        onclick={() => (leftTab = leftTab === 'excel' ? 'palette' : 'excel')}
      >
        Excel
      </button>
    </div>
  </header>

  <div class="main">
    <!-- LEFT DOCK -->
    <aside class="dock">
      <div class="dock-tabs">
        <button
          type="button"
          class="dock-tab"
          class:on={leftTab === 'palette'}
          onclick={() => (leftTab = 'palette')}
        >Palette</button>
        <button
          type="button"
          class="dock-tab"
          class:on={leftTab === 'excel'}
          onclick={() => (leftTab = 'excel')}
        >Excel</button>
      </div>

      {#if leftTab === 'excel'}
        <h2>Excel PIMS MVP</h2>
        <ExcelPanel onStatus={(msg) => (status = msg)} />
      {:else}
        <h2>Unit palette</h2>
        <div class="palette-group">Process</div>
        {#each PROCESS_UNITS as u}
          <button
            type="button"
            class="pal"
            draggable="true"
            ondragstart={(e) => ondragstart(e, u.type)}
            onclick={() => addFromPalette(u.type)}
          >
            <span class="swatch" style:background={UNIT_COLORS[u.type]?.accent || u.color}></span>
            {u.label}
          </button>
        {/each}
        <div class="palette-group">Supply chain</div>
        {#each SUPPLY_UNITS as u}
          <button
            type="button"
            class="pal"
            draggable="true"
            ondragstart={(e) => ondragstart(e, u.type)}
            onclick={() => addFromPalette(u.type)}
          >
            <span class="swatch" style:background={UNIT_COLORS[u.type]?.accent || u.color}></span>
            {u.label}
          </button>
        {/each}

        {#if showResults && lastGraph}
          <div class="res-wrap">
            <h2>Solve results</h2>
            <ResultsPanel result={lastGraph} />
          </div>
        {/if}
      {/if}
    </aside>

    <!-- CANVAS -->
    <div class="canvas" role="presentation" {ondragover} {ondrop}>
      <div class="status-bar">{status}</div>
      <SvelteFlow
        bind:nodes
        bind:edges
        {nodeTypes}
        {edgeTypes}
        {defaultEdgeOptions}
        {onconnect}
        onnodeclick={onNodeClick}
        onedgeclick={onEdgeClick}
        onpaneclick={onPaneClick}
        fitView
        colorMode="dark"
        connectionMode={ConnectionMode.Loose}
        connectionRadius={24}
        style="width:100%;height:100%;"
      >
        <Background variant={BackgroundVariant.Lines} gap={24} color="#1a2430" />
        <Controls position="bottom-left" />
        <MiniMap
          position="bottom-right"
          pannable
          zoomable
          nodeColor={(n) => n.data?.accentColor || '#4a90d9'}
          maskColor="rgba(8,12,18,0.75)"
        />
      </SvelteFlow>
    </div>

    <!-- RIGHT INSPECTOR -->
    <InspectorPanel
      {selection}
      {onUpdateNode}
      {onUpdateEdge}
      onClose={() => (selection = null)}
    />
  </div>
</div>
