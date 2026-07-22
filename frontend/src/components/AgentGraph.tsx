import { Box, Paper, Typography } from "@mui/material";
import {
  Background,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import type { AgentEdgeModel, AgentNodeModel } from "../lib/graphModel";

const STATUS_COLOR: Record<AgentNodeModel["status"], string> = {
  pending: "#3b3f51",
  active: "#7aa2f7",
  done: "#9ece6a",
  failed: "#f7768e",
};

type AgentFlowNode = Node<{ model: AgentNodeModel }, "agent">;

function AgentNode({ data }: NodeProps<AgentFlowNode>) {
  const m = data.model;
  return (
    <Paper
      variant="outlined"
      sx={{
        p: 1,
        width: 200,
        borderColor: STATUS_COLOR[m.status],
        borderWidth: 2,
        boxShadow: m.status === "active" ? `0 0 12px ${STATUS_COLOR.active}66` : "none",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <Typography variant="subtitle2" fontWeight={700}>
        {m.label}
      </Typography>
      {m.sublabel && (
        <Typography variant="caption" color="text.secondary" noWrap display="block">
          {m.sublabel}
        </Typography>
      )}
      {(m.tokens > 0 || m.costUsd > 0) && (
        <Typography variant="caption" color="text.secondary">
          {m.tokens.toLocaleString()} tok · ${m.costUsd.toFixed(4)}
        </Typography>
      )}
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </Paper>
  );
}

const nodeTypes = { agent: AgentNode };
const COLUMN_X: Record<string, number> = {
  router: 0,
  planner: 220,
  analyst: 460,
  critic: 700,
  composer: 940,
};

export function AgentGraph({
  nodes,
  edges,
  onSelect,
}: {
  nodes: AgentNodeModel[];
  edges: AgentEdgeModel[];
  onSelect: (node: AgentNodeModel) => void;
}) {
  const analystCount = nodes.filter((n) => n.agent === "analyst").length;
  const midY = (Math.max(analystCount, 1) - 1) * 55;

  const flowNodes: AgentFlowNode[] = useMemo(() => {
    let analystIndex = 0;
    return nodes.map((m) => {
      const y = m.agent === "analyst" ? analystIndex++ * 110 : midY;
      return {
        id: m.id,
        type: "agent" as const,
        position: { x: COLUMN_X[m.agent] ?? 0, y },
        data: { model: m },
      };
    });
  }, [nodes, midY]);

  const flowEdges: Edge[] = useMemo(
    () =>
      edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        animated: e.retry,
        style: e.retry
          ? { stroke: STATUS_COLOR.failed, strokeDasharray: "6 4" }
          : { stroke: "#3b3f51" },
      })),
    [edges],
  );

  return (
    <Box sx={{ height: 340, border: 1, borderColor: "divider", borderRadius: 2 }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onNodeClick={(_evt, node) => onSelect(node.data.model)}
        fitView
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
      >
        <Background gap={24} color="#1c2030" />
      </ReactFlow>
    </Box>
  );
}
