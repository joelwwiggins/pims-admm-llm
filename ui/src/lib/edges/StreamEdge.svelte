<script>
  import { BaseEdge, EdgeLabel, getSmoothStepPath } from '@xyflow/svelte';

  /** @type {Record<string, any>} */
  let {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    data = {},
    selected = false,
    markerEnd,
    style = '',
  } = $props();

  const path = $derived.by(() => {
    const [edgePath, labelX, labelY] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 12,
      offset: 20,
    });
    return { edgePath, labelX, labelY };
  });

  const label = $derived(data?.label || data?.stream?.name || '');
  const flow = $derived(data?.stream?.flow_kbd);
</script>

<BaseEdge
  {id}
  path={path.edgePath}
  {markerEnd}
  interactionWidth={28}
  style={`
    stroke: ${selected ? '#5eb0f0' : '#6a8fb0'};
    stroke-width: ${selected ? 3.2 : 2.4};
    ${style}
  `}
/>

{#if label}
  <EdgeLabel x={path.labelX} y={path.labelY} selectEdgeOnClick>
    <div class="stream-label" class:selected>
      <span class="name">{label}</span>
      {#if flow != null}
        <span class="flow">{Number(flow).toFixed(1)} kbd</span>
      {/if}
    </div>
  </EdgeLabel>
{/if}

<style>
  .stream-label {
    pointer-events: all;
    cursor: pointer;
    background: #121a24ee;
    border: 1px solid #3a5068;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.68rem;
    color: #d0dce8;
    box-shadow: 0 2px 8px #0008;
    display: flex;
    flex-direction: column;
    align-items: center;
    line-height: 1.2;
    transform: translate(-50%, -50%);
  }
  .stream-label.selected {
    border-color: #5eb0f0;
    color: #fff;
  }
  .name {
    font-weight: 600;
  }
  .flow {
    font-size: 0.58rem;
    color: #8aa0b8;
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, monospace;
  }
</style>
