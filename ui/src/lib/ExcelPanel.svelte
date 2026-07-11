<script>
  import {
    getExcelTemplateBlob,
    postExcelSolve,
    excelResultsUrl,
  } from './api.js';

  /** @type {{ onStatus?: (msg: string) => void }} */
  let { onStatus = () => {} } = $props();

  let fileInput = $state(null);
  let selectedFile = $state(/** @type {File | null} */ (null));
  let busy = $state(false);
  let error = $state('');
  let result = $state(/** @type {Record<string, any> | null} */ (null));

  const mono = $derived(result?.mono || null);
  const admm = $derived(result?.admm || null);
  const cmp = $derived(result?.comparison || null);
  const meta = $derived(result?.meta || null);

  function fmt(x, digits = 3) {
    if (x == null || Number.isNaN(Number(x))) return '—';
    const n = Number(x);
    if (Math.abs(n) >= 100) return n.toFixed(2);
    if (Math.abs(n) >= 1) return n.toFixed(digits);
    return n.toExponential(3);
  }

  function fmtPct(x) {
    if (x == null || Number.isNaN(Number(x))) return '—';
    return `${(Number(x) * 100).toFixed(3)}%`;
  }

  function onFileChange(evt) {
    const f = evt.target?.files?.[0] || null;
    selectedFile = f;
    error = '';
    if (f) onStatus(`Excel selected: ${f.name}`);
  }

  function clearFile() {
    selectedFile = null;
    if (fileInput) fileInput.value = '';
  }

  function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function downloadTemplate() {
    busy = true;
    error = '';
    onStatus('Downloading Excel template…');
    try {
      const blob = await getExcelTemplateBlob();
      triggerDownload(blob, 'pims_admm_template.xlsx');
      onStatus('Template downloaded · pims_admm_template.xlsx');
    } catch (e) {
      error = e.message || String(e);
      onStatus(`Template failed: ${error}`);
    } finally {
      busy = false;
    }
  }

  async function solve() {
    if (!selectedFile) {
      error = 'Choose a .xlsx file first (or download the template)';
      return;
    }
    busy = true;
    error = '';
    result = null;
    onStatus(`Excel solve: ${selectedFile.name}…`);
    try {
      const body = await postExcelSolve(selectedFile);
      result = body;
      const gap = body?.comparison?.objective_gap_rel;
      const v = body?.verdict || (body?.ok ? 'PASS' : 'FAIL');
      onStatus(
        `Excel ${v} · mono ${fmt(body?.mono?.objective, 2)} · ADMM ${fmt(body?.admm?.objective, 2)} · gap ${fmtPct(gap)}`,
      );
    } catch (e) {
      error = e.message || String(e);
      onStatus(`Excel solve failed: ${error}`);
    } finally {
      busy = false;
    }
  }

  function downloadResults() {
    const href =
      meta?.download_results_xlsx ||
      (meta?.results_xlsx ? excelResultsUrl(meta.results_xlsx) : '');
    if (!href) return;
    // use same-origin / proxy path
    window.open(href, '_blank');
  }
</script>

<div class="excel-panel">
  <p class="hint">
    PIMS-shaped workbook → mono + classic ADMM. Primary shadows = free online λ
    (not recovered blender duals).
  </p>

  <div class="row">
    <button type="button" class="btn" disabled={busy} onclick={downloadTemplate}>
      Template
    </button>
    <label class="btn file-btn">
      Upload
      <input
        bind:this={fileInput}
        type="file"
        accept=".xlsx,.xlsm,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        onchange={onFileChange}
        disabled={busy}
      />
    </label>
  </div>

  {#if selectedFile}
    <div class="file-line">
      <span class="fname" title={selectedFile.name}>{selectedFile.name}</span>
      <button type="button" class="linkish" onclick={clearFile} disabled={busy}>×</button>
    </div>
  {:else}
    <div class="file-line muted">No file selected</div>
  {/if}

  <button type="button" class="btn primary wide" disabled={busy || !selectedFile} onclick={solve}>
    {busy ? 'Solving…' : 'Solve Excel'}
  </button>

  {#if error}
    <div class="err">{error}</div>
  {/if}

  {#if result}
    <div class="results">
      <div class="row head">
        <span class:ok={result.ok} class:bad={!result.ok}>
          {result.ok ? 'PASS' : 'FAIL'}
        </span>
        <span class="path" title={result.verdict}>{result.verdict || '—'}</span>
      </div>

      <div class="metric">
        <span class="k">mono obj</span>
        <strong>{fmt(mono?.objective, 2)}</strong>
      </div>
      <div class="metric">
        <span class="k">ADMM obj</span>
        <strong>{fmt(admm?.objective, 2)}</strong>
      </div>
      <div class="metric">
        <span class="k">gap</span>
        <strong>{fmtPct(cmp?.objective_gap_rel)}</strong>
      </div>
      <div class="metric">
        <span class="k">dual L∞ (online λ)</span>
        <strong>{fmt(cmp?.dual_linf_online ?? cmp?.dual_Linf_online ?? cmp?.shadow_linf)}</strong>
      </div>
      <div class="metric">
        <span class="k">ρ / iters</span>
        <strong>{fmt(admm?.rho, 1)} / {admm?.iteration_count ?? '—'}</strong>
      </div>
      <div class="metric">
        <span class="k">||r||</span>
        <strong>{fmt(admm?.primal_residual)}</strong>
      </div>
      <div class="metric">
        <span class="k">wall (pipeline)</span>
        <strong>{fmt(meta?.pipeline_wall_s, 2)} s</strong>
      </div>

      <h3>Path</h3>
      <div class="mono-path">{admm?.dual_recovery_path || '—'}</div>

      {#if mono?.crude_rates && Object.keys(mono.crude_rates).length}
        <h3>Crude rates (mono)</h3>
        <table>
          <tbody>
            {#each Object.entries(mono.crude_rates) as [k, v]}
              {#if Math.abs(Number(v)) > 1e-6}
                <tr><td>{k}</td><td>{Number(v).toFixed(3)}</td></tr>
              {/if}
            {/each}
          </tbody>
        </table>
      {/if}

      {#if mono?.product_rates && Object.keys(mono.product_rates).length}
        <h3>Products (mono)</h3>
        <table>
          <tbody>
            {#each Object.entries(mono.product_rates) as [k, v]}
              {#if Math.abs(Number(v)) > 1e-6}
                <tr><td>{k}</td><td>{Number(v).toFixed(3)}</td></tr>
              {/if}
            {/each}
          </tbody>
        </table>
      {/if}

      {#if admm?.shadow_prices && Object.keys(admm.shadow_prices).length}
        <h3>ADMM shadows (online λ)</h3>
        <table>
          <tbody>
            {#each Object.entries(admm.shadow_prices) as [k, v]}
              <tr><td>{k}</td><td>{fmt(v)}</td></tr>
            {/each}
          </tbody>
        </table>
      {/if}

      {#if mono?.shadow_prices && Object.keys(mono.shadow_prices).length}
        <h3>Mono shadows</h3>
        <table>
          <tbody>
            {#each Object.entries(mono.shadow_prices) as [k, v]}
              <tr><td>{k}</td><td>{fmt(v)}</td></tr>
            {/each}
          </tbody>
        </table>
      {/if}

      {#if meta?.download_results_xlsx || meta?.results_xlsx}
        <button type="button" class="btn wide" onclick={downloadResults}>
          Download results .xlsx
        </button>
      {/if}

      <details>
        <summary>raw JSON</summary>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </details>
    </div>
  {/if}
</div>

<style>
  .excel-panel {
    font-size: 0.75rem;
    color: #d0dbe8;
  }
  .hint {
    margin: 0 0 0.5rem;
    color: #7a8a9a;
    font-size: 0.68rem;
    line-height: 1.35;
  }
  .row {
    display: flex;
    gap: 0.35rem;
    margin-bottom: 0.35rem;
  }
  .row.head {
    align-items: center;
    margin-bottom: 0.5rem;
  }
  .btn {
    background: #1a2838;
    color: #e8eef5;
    border: 1px solid #3a4a5e;
    border-radius: 5px;
    padding: 0.35rem 0.5rem;
    font-size: 0.72rem;
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    border-color: #5a9fd4;
  }
  .btn:disabled {
    opacity: 0.55;
    cursor: wait;
  }
  .btn.primary {
    background: linear-gradient(180deg, #2a6a9e, #1a4d7a);
    border-color: #3d8ec8;
    font-weight: 700;
  }
  .btn.wide {
    width: 100%;
    margin-top: 0.35rem;
  }
  .file-btn {
    position: relative;
    overflow: hidden;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 1;
  }
  .file-btn input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .file-line {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 0.35rem;
    min-height: 1.2rem;
  }
  .file-line.muted {
    color: #5a6e84;
  }
  .fname {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: ui-monospace, monospace;
    font-size: 0.68rem;
    color: #9eb0c4;
  }
  .linkish {
    background: transparent;
    border: none;
    color: #8a9bb0;
    cursor: pointer;
    font-size: 0.9rem;
    line-height: 1;
    padding: 0 0.2rem;
  }
  .err {
    margin-top: 0.4rem;
    color: #ffb0b0;
    background: #3a1a1a;
    border: 1px solid #6a3030;
    border-radius: 4px;
    padding: 0.35rem 0.45rem;
    font-size: 0.68rem;
  }
  .results {
    margin-top: 0.55rem;
    border-top: 1px solid #243040;
    padding-top: 0.45rem;
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
    font-size: 0.65rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 120px;
  }
  .metric {
    display: flex;
    justify-content: space-between;
    padding: 0.15rem 0;
    border-bottom: 1px solid #243040;
  }
  .metric .k {
    color: #8a9bb0;
  }
  h3 {
    margin: 0.65rem 0 0.2rem;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #7d90a8;
  }
  .mono-path {
    font-family: ui-monospace, monospace;
    font-size: 0.62rem;
    color: #9eb0c4;
    word-break: break-all;
    line-height: 1.3;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  td {
    padding: 0.12rem 0;
    border-bottom: 1px solid #1e2836;
  }
  td:last-child {
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, monospace;
  }
  details {
    margin-top: 0.55rem;
  }
  summary {
    cursor: pointer;
    color: #8a9bb0;
  }
  pre {
    max-height: 200px;
    overflow: auto;
    background: #0d131c;
    padding: 0.35rem;
    border-radius: 6px;
    font-size: 0.62rem;
  }
</style>
