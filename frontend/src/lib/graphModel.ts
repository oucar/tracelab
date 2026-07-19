import type { AgentEvent } from "./types";

export type NodeStatus = "pending" | "active" | "done" | "failed";

export interface AgentNodeModel {
  id: string; // "planner" | "analyst-<step>" | "critic" | "composer"
  agent: string;
  stepId: number | null;
  label: string;
  sublabel: string;
  status: NodeStatus;
  tokens: number;
  costUsd: number;
}

export interface AgentEdgeModel {
  id: string;
  source: string;
  target: string;
  retry: boolean;
}

interface PlanStepPayload {
  id: number;
  description: string;
  method: string;
}

/** Derive the agent graph purely from the event list — works for live SSE and stored spans. */
export function buildAgentGraph(events: AgentEvent[]): {
  nodes: AgentNodeModel[];
  edges: AgentEdgeModel[];
} {
  const planEvent = events.find(
    (e) => e.agent === "planner" && e.type === "llm_call" && Array.isArray(e.payload.plan),
  );
  const steps = (planEvent?.payload.plan ?? []) as PlanStepPayload[];
  const finished = events.some((e) => e.type === "run_finished");

  const rollup = (pred: (e: AgentEvent) => boolean) => {
    const sel = events.filter(pred);
    return {
      any: sel.length > 0,
      error: sel.some((e) => e.type === "error"),
      tokens: sel.reduce((a, e) => a + e.tokens_in + e.tokens_out, 0),
      costUsd: sel.reduce((a, e) => a + e.cost_usd, 0),
    };
  };
  const status = (r: { any: boolean; error: boolean }, done: boolean): NodeStatus =>
    r.error ? "failed" : done ? "done" : r.any ? "active" : "pending";

  const nodes: AgentNodeModel[] = [];

  const planner = rollup((e) => e.agent === "planner");
  nodes.push({
    id: "planner",
    agent: "planner",
    stepId: null,
    label: "planner",
    sublabel: steps.length ? `${steps.length}-step plan` : "",
    status: status(planner, planner.any),
    tokens: planner.tokens,
    costUsd: planner.costUsd,
  });

  for (const step of steps) {
    const r = rollup((e) => e.agent === "analyst" && e.payload.step_id === step.id);
    const done =
      finished ||
      events.some(
        (e) => e.agent === "analyst" && e.payload.step_id === step.id && e.payload.action === "finish",
      );
    nodes.push({
      id: `analyst-${step.id}`,
      agent: "analyst",
      stepId: step.id,
      label: `analyst ${step.id}`,
      sublabel: `${step.method} · ${step.description}`,
      status: status(r, done),
      tokens: r.tokens,
      costUsd: r.costUsd,
    });
  }

  const critic = rollup((e) => e.agent === "critic");
  const verdicts = events.filter((e) => e.type === "verdict").length;
  nodes.push({
    id: "critic",
    agent: "critic",
    stepId: null,
    label: "critic",
    sublabel: verdicts ? `${verdicts} verdicts` : "",
    status: status(critic, verdicts > 0 || finished),
    tokens: critic.tokens,
    costUsd: critic.costUsd,
  });

  const composer = rollup((e) => e.agent === "composer");
  nodes.push({
    id: "composer",
    agent: "composer",
    stepId: null,
    label: "composer",
    sublabel: "",
    status: status(composer, finished),
    tokens: composer.tokens,
    costUsd: composer.costUsd,
  });

  const edges: AgentEdgeModel[] = [];
  for (const step of steps) {
    edges.push({ id: `p-a${step.id}`, source: "planner", target: `analyst-${step.id}`, retry: false });
    edges.push({ id: `a${step.id}-c`, source: `analyst-${step.id}`, target: "critic", retry: false });
  }
  if (steps.length === 0) {
    // honest-failure path: planner straight to composer
    edges.push({ id: "p-comp", source: "planner", target: "composer", retry: false });
  }
  edges.push({ id: "c-comp", source: "critic", target: "composer", retry: false });
  for (const e of events) {
    if (e.agent === "critic" && e.type === "handoff" && Array.isArray(e.payload.retry_steps)) {
      for (const sid of e.payload.retry_steps as number[]) {
        edges.push({ id: `retry-${sid}`, source: "critic", target: `analyst-${sid}`, retry: true });
      }
    }
  }
  return { nodes, edges };
}
