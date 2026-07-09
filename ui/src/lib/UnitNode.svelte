<script>
  import { Handle, Position } from '@xyflow/svelte';

  let { data } = $props();

  function toggleActive() {
    data.active = !data.active;
  }

  function setSubmodel(e) {
    data.submodel = e.target.value;
  }
</script>

<div class="unit" class:inactive={!data.active}>
  <Handle type="target" position={Position.Left} id="in" />
  <div class="title">{data.label || data.unitType}</div>
  <div class="meta">{data.unitType}</div>
  <label class="row">
    <input type="checkbox" checked={data.active} onchange={toggleActive} />
    active
  </label>
  <label class="row">
    submodel
    <select value={data.submodel || 'lp'} onchange={setSubmodel}>
      <option value="lp">lp</option>
      <option value="tensorflow">tensorflow</option>
    </select>
  </label>
  <Handle type="source" position={Position.Right} id="out" />
</div>

<style>
  .unit {
    min-width: 140px;
    background: #fff;
    border: 2px solid #0ea5e9;
    border-radius: 10px;
    padding: 0.5rem 0.65rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
    font-size: 0.8rem;
  }
  .unit.inactive {
    opacity: 0.45;
    border-color: #94a3b8;
  }
  .title {
    font-weight: 700;
    color: #0f172a;
  }
  .meta {
    color: #64748b;
    font-size: 0.7rem;
    margin-bottom: 0.35rem;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.25rem;
  }
  select {
    font-size: 0.75rem;
  }
</style>
