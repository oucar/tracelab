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
