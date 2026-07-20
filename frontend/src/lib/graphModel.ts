import type { AgentEvent } from "./types";

export type NodeStatus = "pending" | "active" | "done" | "failed";

export type NodeAgent = "router" | "planner" | "analyst" | "critic" | "composer";

export interface AgentNodeModel {
  id: string; // "router" | "planner" | "analyst-<step>" | "critic" | "composer"
  agent: NodeAgent;
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
  const routerEvents = events.filter((e) => e.agent === "router");
  const hasRouter = routerEvents.length > 0;
  const hasPlanner = events.some((e) => e.agent === "planner");
  // Old (pre-M4) recorded runs never emitted router events at all — in that
  // case the planner is always rendered, exactly as before this change.
  const plannerRendered = hasPlanner || !hasRouter;

  const planEvent = events.find(
    (e) => e.agent === "planner" && e.type === "llm_call" && Array.isArray(e.payload.plan),
  );
  let steps = (planEvent?.payload.plan ?? []) as PlanStepPayload[];

  // Simple-route runs skip the planner entirely: derive one analyst node per
  // distinct payload.step_id straight from the analyst events themselves.
  if (steps.length === 0 && !hasPlanner) {
    const stepIds = Array.from(
      new Set(
        events
          .filter((e) => e.agent === "analyst" && typeof e.payload.step_id === "number")
          .map((e) => e.payload.step_id as number),
      ),
    ).sort((a, b) => a - b);
    const question = events.find((e) => e.type === "run_started")?.payload.question as
      | string
      | undefined;
    steps = stepIds.map((id) => ({ id, description: question ?? "", method: "descriptive" }));
  }

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

  if (hasRouter) {
    const routerHandoff = routerEvents.find((e) => e.type === "handoff");
    const router = rollup((e) => e.agent === "router");
    nodes.push({
      id: "router",
      agent: "router",
      stepId: null,
      label: "router",
      sublabel: (routerHandoff?.payload.route as string) ?? "",
      status: status(router, router.any),
      tokens: router.tokens,
      costUsd: router.costUsd,
    });
  }

  if (plannerRendered) {
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
  }

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

  const folded = events.some(
    (e) => e.agent === "composer" && e.type === "handoff" && Boolean(e.payload.folded),
  );
  const composer = rollup((e) => e.agent === "composer");
  nodes.push({
    id: "composer",
    agent: "composer",
    stepId: null,
    label: "composer",
    sublabel: folded ? "folded" : "",
    status: status(composer, finished),
    tokens: composer.tokens,
    costUsd: composer.costUsd,
  });

  const edges: AgentEdgeModel[] = [];

  if (hasRouter) {
    if (plannerRendered) {
      edges.push({ id: "r-p", source: "router", target: "planner", retry: false });
    } else if (steps.length > 0) {
      edges.push({ id: "r-a", source: "router", target: `analyst-${steps[0].id}`, retry: false });
    } else {
      // crashed run: router fired but neither planner nor analyst got anywhere
      edges.push({ id: "r-comp", source: "router", target: "composer", retry: false });
    }
  }

  for (const step of steps) {
    if (plannerRendered) {
      edges.push({ id: `p-a${step.id}`, source: "planner", target: `analyst-${step.id}`, retry: false });
    }
    edges.push({ id: `a${step.id}-c`, source: `analyst-${step.id}`, target: "critic", retry: false });
  }
  if (plannerRendered && steps.length === 0) {
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
