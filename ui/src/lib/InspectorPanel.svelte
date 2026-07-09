<script>
  /**
   * HYSYS-style right inspector: node unit properties / stream properties.
   * @type {{
   *   selection: null | { kind: 'node'|'edge', id: string, data: any, label?: string },
   *   onUpdateNode?: (id: string, patch: object) => void,
   *   onUpdateEdge?: (id: string, patch: object) => void,
   *   onClose?: () => void
   * }}
   */
  let { selection = null, onUpdateNode, onUpdateEdge, onClose } = $props();

  let tab = $state('general');

  $effect(() => {
    // reset tab when selection changes
    if (selection) tab = 'general';
  });

  const isNode = $derived(selection?.kind === 'node');
  const isEdge = $derived(selection?.kind === 'edge');
  const data = $derived(selection?.data || {});
  const stream = $derived(data.stream || {});
  const yields = $derived(data.yields || []);
  const processConditions = $derived(data.processConditions || {});
  const feedQuality = $derived(data.feedQuality || {});
  const pcKeys = $derived(Object.keys(processConditions));
  const fqKeys = $derived(Object.keys(feedQuality));

  function setNodeField(key, value) {
    if (!selection || selection.kind !== 'node') return;
    onUpdateNode?.(selection.id, { [key]: value });
  }

  function setYield(i, key, value) {
    if (!selection || selection.kind !== 'node') return;
    const next = (data.yields || []).map((row, idx) =>
      idx === i ? { ...row, [key]: value } : row,
    );
    onUpdateNode?.(selection.id, { yields: next });
  }

  function setStreamField(key, value) {
    if (!selection || selection.kind !== 'edge') return;
    const s = { ...(data.stream || {}), [key]: value };
    onUpdateEdge?.(selection.id, { stream: s, label: data.label });
  }

  function setComp(i, key, value) {
    if (!selection || selection.kind !== 'edge') return;
    const comp = [...(stream.composition || [])];
    comp[i] = { ...comp[i], [key]: value };
    setStreamField('composition', comp);
  }
</script>

<aside class="inspector">
  <div class="insp-head">
    <div>
      <div class="eyebrow">{isNode ? 'UNIT' : isEdge ? 'STREAM' : 'INSPECTOR'}</div>
      <div class="title">
        {#if isNode}
          {data.tag || ''} {data.label || selection?.id}
        {:else if isEdge}
          {data.label || stream.name || selection?.id}
        {:else}
          No selection
        {/if}
      </div>
    </div>
    {#if selection}
      <button type="button" class="x" onclick={() => onClose?.()} title="Close">×</button>
    {/if}
  </div>

  {#if !selection}
    <p class="hint">Click a unit or stream on the PFD to inspect properties (HYSYS-style).</p>
  {:else if isNode}
    <div class="tabs">
      <button type="button" class:on={tab === 'general'} onclick={() => (tab = 'general')}>General</button>
      <button type="button" class:on={tab === 'streams'} onclick={() => (tab = 'streams')}>Streams</button>
      {#if yields.length}
        <button type="button" class:on={tab === 'yields'} onclick={() => (tab = 'yields')}>Yields</button>
      {/if}
      {#if pcKeys.length || fqKeys.length}
        <button type="button" class:on={tab === 'process'} onclick={() => (tab = 'process')}>Process</button>
      {/if}
    </div>

    {#if tab === 'general'}
      <div class="section">
        <label>Tag
          <input value={data.tag || ''} oninput={(e) => setNodeField('tag', e.currentTarget.value)} />
        </label>
        <label>Name
          <input value={data.label || ''} oninput={(e) => setNodeField('label', e.currentTarget.value)} />
        </label>
        <label>Unit type
          <input value={data.unitType || ''} readonly />
        </label>
        <label class="row">
          <input
            type="checkbox"
            checked={!!data.active}
            onchange={(e) => setNodeField('active', e.currentTarget.checked)}
          />
          Active
        </label>
        <label>Submodel
          <select
            value={data.submodel || 'lp'}
            onchange={(e) => setNodeField('submodel', e.currentTarget.value)}
          >
            <option value="lp">LP (PuLP/CBC)</option>
            <option value="tensorflow">TensorFlow surrogate</option>
          </select>
        </label>
        <label>Status
          <select
            value={data.status || 'running'}
            onchange={(e) => setNodeField('status', e.currentTarget.value)}
          >
            <option value="running">running</option>
            <option value="idle">idle</option>
            <option value="alarm">alarm</option>
          </select>
        </label>
        {#if data.charge_kbd != null}
          <label>Charge (kbd)
            <input
              type="number"
              step="0.1"
              value={data.charge_kbd}
              oninput={(e) => setNodeField('charge_kbd', Number(e.currentTarget.value))}
            />
          </label>
        {/if}
        {#if data.description}
          <p class="desc">{data.description}</p>
        {/if}
      </div>
    {:else if tab === 'streams'}
      <div class="section">
        <h4>Feeds</h4>
        <ul>
          {#each data.ports?.feeds || [] as p}
            <li><code>{p.id}</code> {p.name}</li>
          {/each}
        </ul>
        <h4>Products</h4>
        <ul>
          {#each data.ports?.products || [] as p}
            <li><code>{p.id}</code> {p.name}</li>
          {/each}
        </ul>
      </div>
    {:else if tab === 'yields'}
      <div class="section">
        <table class="edit">
          <thead>
            <tr>
              <th>Product</th>
              <th>Yield %</th>
              <th>kbd</th>
              <th>S wt%</th>
              <th>RON</th>
            </tr>
          </thead>
          <tbody>
            {#each yields as y, i}
              <tr>
                <td>
                  <input value={y.product} oninput={(e) => setYield(i, 'product', e.currentTarget.value)} />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.1"
                    value={y.yield_pct}
                    oninput={(e) => setYield(i, 'yield_pct', Number(e.currentTarget.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.1"
                    value={y.flow_kbd}
                    oninput={(e) => setYield(i, 'flow_kbd', Number(e.currentTarget.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.001"
                    value={y.sulfur_wt}
                    oninput={(e) => setYield(i, 'sulfur_wt', Number(e.currentTarget.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.1"
                    value={y.ron ?? ''}
                    oninput={(e) =>
                      setYield(i, 'ron', e.currentTarget.value === '' ? null : Number(e.currentTarget.value))}
                  />
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
        <p class="note">Yields feed the LP block context (hard constraints still enforced by solver). Every product row maps to a routed stream.</p>
      </div>
    {:else if tab === 'process'}
      <div class="section">
        <h4>Operating variables</h4>
        {#if pcKeys.length}
          {#each pcKeys as key}
            <label>{key}
              <input
                type="number"
                step="any"
                value={processConditions[key]}
                oninput={(e) => {
                  const next = { ...processConditions, [key]: Number(e.currentTarget.value) };
                  setNodeField('processConditions', next);
                }}
              />
            </label>
          {/each}
        {:else}
          <p class="hint">No process conditions on this unit.</p>
        {/if}
        <h4>Feed quality vector</h4>
        {#if fqKeys.length}
          {#each fqKeys as key}
            <label>{key}
              <input
                type="number"
                step="any"
                value={feedQuality[key]}
                oninput={(e) => {
                  const next = { ...feedQuality, [key]: Number(e.currentTarget.value) };
                  setNodeField('feedQuality', next);
                }}
              />
            </label>
          {/each}
        {:else}
          <p class="hint">No feed quality vector on this unit.</p>
        {/if}
        <p class="note">Process conditions drive planning yield vectors (ROT, C/O, severity). Solver keeps hard LP constraints.</p>
      </div>
    {/if}
  {:else if isEdge}
    <div class="tabs">
      <button type="button" class:on={tab === 'general'} onclick={() => (tab = 'general')}>Properties</button>
      <button type="button" class:on={tab === 'comp'} onclick={() => (tab = 'comp')}>Composition</button>
    </div>

    {#if tab === 'general'}
      <div class="section">
        <label>Stream name
          <input
            value={stream.name || data.label || ''}
            oninput={(e) => {
              const v = e.currentTarget.value;
              onUpdateEdge?.(selection.id, {
                label: v,
                stream: { ...stream, name: v },
              });
            }}
          />
        </label>
        <label>Flow (kbd)
          <input
            type="number"
            step="0.1"
            value={stream.flow_kbd ?? 0}
            oninput={(e) => setStreamField('flow_kbd', Number(e.currentTarget.value))}
          />
        </label>
        <label>Temperature (°F)
          <input
            type="number"
            step="1"
            value={stream.temperature_f ?? 100}
            oninput={(e) => setStreamField('temperature_f', Number(e.currentTarget.value))}
          />
        </label>
        <label>Pressure (psig)
          <input
            type="number"
            step="1"
            value={stream.pressure_psig ?? 0}
            oninput={(e) => setStreamField('pressure_psig', Number(e.currentTarget.value))}
          />
        </label>
        <label>Vapor fraction
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={stream.vapor_fraction ?? 0}
            oninput={(e) => setStreamField('vapor_fraction', Number(e.currentTarget.value))}
          />
        </label>
        <label>API gravity
          <input
            type="number"
            step="0.1"
            value={stream.density_api ?? 30}
            oninput={(e) => setStreamField('density_api', Number(e.currentTarget.value))}
          />
        </label>
        <label>Sulfur (wt%)
          <input
            type="number"
            step="0.001"
            value={stream.sulfur_wt ?? 0}
            oninput={(e) => setStreamField('sulfur_wt', Number(e.currentTarget.value))}
          />
        </label>
        <label>RON
          <input
            type="number"
            step="0.1"
            value={stream.ron ?? ''}
            oninput={(e) =>
              setStreamField('ron', e.currentTarget.value === '' ? null : Number(e.currentTarget.value))}
          />
        </label>
      </div>
    {:else}
      <div class="section">
        <table class="edit">
          <thead>
            <tr><th>Cut</th><th>vol %</th></tr>
          </thead>
          <tbody>
            {#each stream.composition || [] as c, i}
              <tr>
                <td>
                  <input value={c.cut} oninput={(e) => setComp(i, 'cut', e.currentTarget.value)} />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.1"
                    value={c.vol_pct}
                    oninput={(e) => setComp(i, 'vol_pct', Number(e.currentTarget.value))}
                  />
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  {/if}
</aside>

<style>
  .inspector {
    width: 320px;
    background: #101820;
    border-left: 1px solid #2a3a4c;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }
  .insp-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.75rem 0.85rem;
    border-bottom: 1px solid #243040;
    background: linear-gradient(180deg, #162030, #101820);
  }
  .eyebrow {
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    color: #6a8098;
    font-weight: 700;
  }
  .title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8eef5;
  }
  .x {
    background: transparent;
    border: none;
    color: #8a9bb0;
    font-size: 1.3rem;
    cursor: pointer;
    line-height: 1;
  }
  .hint {
    color: #7a8a9a;
    font-size: 0.78rem;
    padding: 1rem;
    margin: 0;
  }
  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid #243040;
  }
  .tabs button {
    flex: 1;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #8a9bb0;
    padding: 0.5rem 0.25rem;
    font-size: 0.72rem;
    cursor: pointer;
  }
  .tabs button.on {
    color: #5eb0f0;
    border-bottom-color: #5eb0f0;
    font-weight: 600;
  }
  .section {
    padding: 0.75rem 0.85rem;
    overflow: auto;
    flex: 1;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.7rem;
    color: #8a9bb0;
    margin-bottom: 0.55rem;
  }
  label.row {
    flex-direction: row;
    align-items: center;
    gap: 0.45rem;
  }
  input,
  select {
    background: #0d131c;
    border: 1px solid #2e3f54;
    border-radius: 4px;
    color: #e8eef5;
    padding: 0.35rem 0.45rem;
    font-size: 0.78rem;
  }
  input[readonly] {
    opacity: 0.7;
  }
  .desc {
    font-size: 0.72rem;
    color: #9eb0c4;
    margin: 0.4rem 0 0;
  }
  h4 {
    margin: 0.4rem 0 0.25rem;
    font-size: 0.68rem;
    text-transform: uppercase;
    color: #6a8098;
  }
  ul {
    margin: 0 0 0.5rem;
    padding-left: 0;
    list-style: none;
  }
  li {
    font-size: 0.75rem;
    color: #c5d2e0;
    padding: 0.15rem 0;
  }
  code {
    color: #7eb8e8;
    font-size: 0.68rem;
    margin-right: 0.35rem;
  }
  table.edit {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.68rem;
  }
  table.edit th {
    text-align: left;
    color: #6a8098;
    font-weight: 600;
    padding: 0.2rem;
  }
  table.edit td {
    padding: 0.15rem;
  }
  table.edit input {
    width: 100%;
    min-width: 0;
    padding: 0.2rem;
    font-size: 0.68rem;
  }
  .note {
    font-size: 0.65rem;
    color: #6a8098;
    margin-top: 0.5rem;
  }
</style>
