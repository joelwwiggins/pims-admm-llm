<script>
  /** @type {{ result: Record<string, any> | null }} */
  let { result = null } = $props();

  const admm = $derived(result?.admm || null);
  const feeds = $derived(result?.unit_feeds || {});
  const products = $derived(result?.products || {});
  const splits = $derived(result?.routing_splits || {});
  const arcs = $derived(result?.arc_flows || {});

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
  .metric label,
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
</style>
