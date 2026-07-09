<script>
  import { Handle, Position, useSvelteFlow } from '@xyflow/svelte';

  /** @type {{ id: string, data: Record<string, any> }} */
  let { id, data } = $props();

  const { updateNodeData } = useSvelteFlow();

  const isSupply = $derived(
    data.category === 'supply_chain' ||
      data.unitType === 'warehouse' ||
      data.unitType === 'transport',
  );

  function toggleActive(evt) {
    updateNodeData(id, { active: evt.currentTarget.checked });
  }

  function onSubmodelChange(evt) {
    updateNodeData(id, { submodel: evt.currentTarget.value });
  }
</script>

<div
  class="unit-node"
  class:inactive={!data.active}
  class:supply={isSupply}
  style:border-left="4px solid {data.color || '#4a90d9'}"
>
  <Handle type="target" position={Position.Left} id="in" />
  <div class="title">{data.label || data.unitType || id}</div>
  <div class="meta">{data.unitType}</div>
  <label class="row nodrag">
    <input type="checkbox" checked={!!data.active} onchange={toggleActive} />
    active
  </label>
  <label class="row nodrag">
    submodel
    <select value={data.submodel || 'lp'} onchange={onSubmodelChange}>
      <option value="lp">lp</option>
      <option value="tensorflow">tensorflow</option>
    </select>
  </label>
  <Handle type="source" position={Position.Right} id="out" />
</div>

<style>
  .unit-node {
    min-width: 150px;
    background: #1a2330;
    border: 1.5px solid #3d5168;
    border-radius: 10px;
    padding: 8px 10px 10px;
    box-shadow: 0 4px 14px #0006;
    color: #e8eef5;
    font-size: 0.8rem;
  }
  .unit-node.inactive {
    opacity: 0.55;
    border-style: dashed;
  }
  .unit-node.supply {
    border-color: #6b5b95;
  }
  .title {
    font-weight: 600;
    font-size: 0.9rem;
  }
  .meta {
    color: #9eb0c4;
    font-size: 0.7rem;
    margin-bottom: 4px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 4px;
    font-size: 0.72rem;
    color: #9eb0c4;
  }
  select {
    font-size: 0.72rem;
    background: #243447;
    color: #e8eef5;
    border: 1px solid #3a4a5e;
    border-radius: 4px;
  }
</style>
