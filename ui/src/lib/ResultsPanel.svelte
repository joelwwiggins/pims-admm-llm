<script>
  /** @type {{ result: Record<string, any> | null }} */
  let { result = null } = $props();

  const admm = $derived(result?.admm || null);
  const feeds = $derived(result?.unit_feeds || {});
  const products = $derived(result?.products || {});
  const splits = $derived(result?.routing_splits || {});
  const arcs = $derived(result?.arc_flows || {});
  const duals = $derived(result?.duals || {});
  const qualityDuals = $derived(result?.quality_duals || {});
  const quality = $derived(result?.quality || null);
  const processConditions = $derived(result?.process_conditions || {});
  const pn = $derived(result?.process_network || null);
  const pnBase = $derived(pn?.baseline || null);
  const pnReplan = $derived(pn?.replan || null);
  const pnDelta = $derived(pn?.delta || null);
  const pnActions = $derived(pn?.actions || []);

  function fmt(x) {
    if (x == null || Number.isNaN(Number(x))) return '—';
    const n = Number(x);
    if (Math.abs(n) >= 100) return n.toFixed(2);
    if (Math.abs(n) >= 1) return n.toFixed(3);
    return n.toExponential(3);
  }
</script>

{#if !result}
  <div class="empty">Solve a graph to see objective, feeds, products, ADMM metrics.</div>
{:else}
  <div class="results">
    <div class="row head">
      <span class:ok={result.ok || result.feasible} class:bad={!(result.ok || result.feasible)}>
        {(result.ok || result.feasible) ? 'FEASIBLE' : 'FAIL'}
      </span>
      <span class="path">{result.admm_status || admm?.dual_recovery_path || '—'}</span>
    </div>

    <div class="metric">
      <span class="k">objective</span>
      <strong>{result.objective != null ? Number(result.objective).toFixed(3) : '—'}</strong>
    </div>
    <div class="metric">
      <span class="k">solve_s</span>
      <strong>{result.solve_time_s != null ? Number(result.solve_time_s).toFixed(4) : '—'}</strong>
    </div>

    {#if pn}
      <h3>Process-network agents</h3>
      <div class="row head" style="margin-bottom:0.35rem">
        <span class:ok={!pn.applied || pn.recommended_plan === 'replan'} class:bad={pn.baseline?.severity === 'critical'}>
          {(pn.recommended_plan || 'baseline').toUpperCase()}
        </span>
        <span class="path">
          {pn.applied ? 'closed-loop' : 'single-round'}
          · sev {pnBase?.severity || '—'}
        </span>
      </div>
      {#if pnDelta?.applied}
        <table>
          <tbody>
            <tr><td>Δ objective</td><td>{fmt(pnDelta.delta_obj)} ({fmt(pnDelta.delta_obj_pct)}%)</td></tr>
            <tr><td>obj</td><td>{fmt(pnDelta.objective_0)} → {fmt(pnDelta.objective_1)}</td></tr>
            <tr><td>coker feed</td><td>{fmt(pnDelta.coker_feed_0)} → {fmt(pnDelta.coker_feed_1)}</td></tr>
            <tr><td>fuel oil</td><td>{fmt(pnDelta.fuel_oil_0)} → {fmt(pnDelta.fuel_oil_1)}</td></tr>
            <tr><td>hard pushbacks</td><td>{pnDelta.hard_pushbacks_0} → {pnDelta.hard_pushbacks_1}</td></tr>
          </tbody>
        </table>
      {/if}
      {#if pnActions.length}
        <h3>Replan actions</h3>
        <ul class="fb">
          {#each pnActions as a}
            <li><strong>{a.code}</strong> — {a.reason}</li>
          {/each}
        </ul>
      {/if}
      {#if pnBase?.areas?.length}
        <h3>Area status (baseline)</h3>
        <table>
          <tbody>
            {#each pnBase.areas as a}
              <tr>
                <td>
                  <span class="sev" class:alarm={a.status === 'pushback' || a.status === 'critical'} class:watch={a.status === 'watch'}>{a.area}</span>
                </td>
                <td>{a.wiggle_room} · {a.status}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
      {#if (pnBase?.pushbacks || []).length}
        <h3>Pushbacks</h3>
        <ul class="fb">
          {#each (pnReplan?.pushbacks || pnBase.pushbacks).slice(0, 8) as pb}
            <li>
              <strong>{pb.area}</strong>
              <span class="sev" class:alarm={pb.severity === 'pushback' || pb.severity === 'critical'} class:watch={pb.severity === 'watch'}>{pb.severity}</span>
              {pb.message}
            </li>
          {/each}
        </ul>
      {/if}
      {#if pnBase?.master_summary}
        <h3>Master</h3>
        <p class="master">{pnBase.master_summary}</p>
        {#if pnReplan?.master_summary && pn.applied}
          <p class="master replan"><em>Replan:</em> {pnReplan.master_summary}</p>
        {/if}
      {/if}
    {/if}

    {#if admm}
      <h3>ADMM</h3>
      <table>
        <tbody>
          <tr><td>path</td><td>{admm.dual_recovery_path}</td></tr>
          <tr><td>ρ</td><td>{admm.rho}</td></tr>
          <tr><td>iters</td><td>{admm.iterations}</td></tr>
          <tr><td>||r||</td><td>{fmt(admm.primal_residual_norm)}</td></tr>
          <tr><td>||s||</td><td>{fmt(admm.dual_residual_norm)}</td></tr>
          <tr><td>λ_vs_mono_L∞</td><td>{fmt(admm.lambda_vs_mono_Linf)}</td></tr>
        </tbody>
      </table>
    {/if}

    <h3>Unit feeds</h3>
    <table>
      <tbody>
        {#each Object.entries(feeds) as [k, v]}
          <tr><td>{k}</td><td>{Number(v).toFixed(3)}</td></tr>
        {/each}
      </tbody>
    </table>

    <h3>Products</h3>
    <table>
      <tbody>
        {#each Object.entries(products) as [k, v]}
          <tr><td>{k}</td><td>{Number(v).toFixed(3)}</td></tr>
        {/each}
      </tbody>
    </table>

    {#if Object.keys(splits).length}
      <h3>Routing splits</h3>
      <table>
        <tbody>
          {#each Object.entries(splits) as [k, v]}
            <tr><td>{k}</td><td>{Number(v).toFixed(3)}</td></tr>
          {/each}
        </tbody>
      </table>
    {/if}

    {#if Object.keys(arcs).length}
      <h3>Arc flows (nonzero)</h3>
      <table>
        <tbody>
          {#each Object.entries(arcs) as [k, v]}
            {#if Math.abs(Number(v)) > 1e-6}
              <tr><td>{k}</td><td>{Number(v).toFixed(3)}</td></tr>
            {/if}
          {/each}
        </tbody>
      </table>
    {/if}

    {#if Object.keys(duals).length}
      <h3>Shadow duals (nonzero)</h3>
      <table>
        <tbody>
          {#each Object.entries(duals) as [k, v]}
            <tr><td>{k}</td><td>{fmt(v)}</td></tr>
          {/each}
        </tbody>
      </table>
    {/if}

    {#if Object.keys(qualityDuals).length}
      <h3>Quality duals</h3>
      <table>
        <tbody>
          {#each Object.entries(qualityDuals) as [k, v]}
            <tr><td>{k}</td><td>{fmt(v)}</td></tr>
          {/each}
        </tbody>
      </table>
    {/if}

    {#if quality}
      <h3>Quality meta</h3>
      <table>
        <tbody>
          <tr><td>model</td><td>{quality.model || '—'}</td></tr>
          <tr><td>base</td><td>{quality.base_stream || quality.base || '—'}</td></tr>
          {#if quality.base_ron != null}<tr><td>base_ron</td><td>{fmt(quality.base_ron)}</td></tr>{/if}
          {#if quality.base_s != null}<tr><td>base_s</td><td>{fmt(quality.base_s)}</td></tr>{/if}
        </tbody>
      </table>
    {/if}

    {#if Object.keys(processConditions).length}
      <h3>Process conditions (solve)</h3>
      <table>
        <tbody>
          {#each Object.entries(processConditions) as [unit, pc]}
            <tr><td colspan="2" style="font-weight:600;color:#9eb0c4">{unit}</td></tr>
            {#each Object.entries(pc || {}) as [k, v]}
              {#if typeof v !== 'object'}
                <tr><td style="padding-left:0.5rem">{k}</td><td>{fmt(v)}</td></tr>
              {/if}
            {/each}
          {/each}
        </tbody>
      </table>
    {/if}

    <details>
      <summary>raw JSON</summary>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </details>
  </div>
{/if}

<style>
  .empty {
    color: #7a8a9a;
    font-size: 0.78rem;
    padding: 0.5rem 0;
  }
  .results {
    font-size: 0.75rem;
    color: #d0dbe8;
  }
  .row.head {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.5rem;
  }
  .ok {
    background: #1e6b3a;
    color: #d4ffe4;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.7rem;
  }
  .bad {
    background: #7a1f1f;
    color: #ffd4d4;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.7rem;
  }
  .path {
    color: #9eb0c4;
    font-family: ui-monospace, monospace;
  }
  .metric {
    display: flex;
    justify-content: space-between;
    padding: 0.2rem 0;
    border-bottom: 1px solid #243040;
  }
  .metric .k {
    color: #8a9bb0;
  }
  h3 {
    margin: 0.7rem 0 0.25rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #7d90a8;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  td {
    padding: 0.15rem 0;
    border-bottom: 1px solid #1e2836;
  }
  td:last-child {
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, monospace;
  }
  details {
    margin-top: 0.6rem;
  }
  summary {
    cursor: pointer;
    color: #8a9bb0;
  }
  pre {
    max-height: 220px;
    overflow: auto;
    background: #0d131c;
    padding: 0.4rem;
    border-radius: 6px;
    font-size: 0.65rem;
  }
  ul.fb {
    margin: 0.2rem 0 0.4rem;
    padding-left: 1rem;
    color: #c5d2e0;
    font-size: 0.7rem;
  }
  ul.fb li {
    margin-bottom: 0.25rem;
  }
  .sev {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0 4px;
    border-radius: 3px;
    margin-right: 4px;
    background: #2a3848;
    color: #9eb0c4;
  }
  .sev.alarm {
    background: #7a1f1f;
    color: #ffd4d4;
  }
  .sev.watch {
    background: #6a5810;
    color: #fff0b0;
  }
  p.master {
    font-size: 0.7rem;
    color: #b8c8d8;
    line-height: 1.35;
    margin: 0.2rem 0 0.4rem;
  }
  p.master.replan {
    color: #9ec9ef;
  }
</style>
