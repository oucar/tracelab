/** Mirror of the backend AgentEvent — the one shape the whole UI consumes. */

export type EventType =
  | "run_started"
  | "llm_call"
  | "tool_call"
  | "handoff"
  | "verdict"
  | "answer_chunk"
  | "run_finished"
  | "error";

export interface AgentEvent {
  run_id: string;
  span_id: string;
  parent_span_id: string | null;
  agent: "planner" | "analyst" | "critic" | "composer" | "system";
  type: EventType;
  payload: Record<string, unknown>;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  started_at: number;
  duration_ms: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  nulls: number;
  unique: number;
  sample: string[];
}

export interface DatasetProfile {
  rows: number;
  columns: ColumnProfile[];
  preview: Record<string, string>[];
}

export interface Dataset {
  id: string;
  name: string;
  profile: DatasetProfile;
}

export interface Methodology {
  method: string;
  n: number;
  p_value: number | null;
  effect_size: number | null;
  effect_size_name: string;
  assumptions: string[];
}

export interface Claim {
  id: string;
  step_id: number;
  text: string;
  kind: "numeric" | "categorical" | "statistical";
  value: number | string | null;
  direction: "higher" | "lower" | "none" | null;
  significant: boolean | null;
  methodology: Methodology | null;
}

export interface VerifiedClaim {
  claim: Claim;
  status: "verified" | "unverified";
  detail: string;
}

export interface FinalAnswer {
  narrative: string;
  claims: VerifiedClaim[];
  charts: unknown[]; // validated at render time by the Zod chartSpecSchema
  failed: boolean;
}

export type RunStatusValue = "running" | "finished" | "error";

/** One row of GET /api/runs — dashboard aggregates, no heavy payloads. */
export interface RunSummary {
  id: string;
  dataset_id: string;
  question: string;
  status: RunStatusValue;
  replay_of: string;
  created_at: number;
  cost_usd: number;
  tokens_in: number;
  tokens_out: number;
  duration_ms: number;
  claims_total: number;
  claims_verified: number;
}

/** GET /api/runs/:id — the full run including its span tree. */
export interface RunDetail {
  id: string;
  dataset_id: string;
  question: string;
  status: RunStatusValue;
  answer: string;
  result: FinalAnswer | null;
  replay_of: string;
  created_at: number;
  spans: AgentEvent[];
}

export interface AppConfig {
  cheap_mode: boolean;
  daily_budget_usd: number;
  spent_today: number;
  models: Record<string, string>;
}
