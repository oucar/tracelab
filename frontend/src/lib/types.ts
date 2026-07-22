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
  agent: "router" | "planner" | "analyst" | "critic" | "composer" | "system";
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

export interface JudgeScores {
  clarity: number;
  uncertainty_honesty: number;
  chart_appropriateness: number;
  methodological_soundness: number;
  rationale: string;
}

/** One row of GET /api/evals — eval run aggregates. */
export interface EvalRunSummary {
  id: string;
  created_at: number;
  label: string;
  git_sha: string;
  config_hash: string;
  config: Record<string, unknown>;
  questions_total: number;
  tier1_scorable: number;
  tier1_passed: number;
  tier1_pass_rate: number | null;
  judge_avg: number | null;
  cost_usd: number;
  duration_ms: number;
}

/** One row of GET /api/evals/:id results — per-question eval outcome. */
export interface EvalResultRow {
  eval_run_id: string;
  question_id: string;
  run_id: string;
  dataset: string;
  tags: string[];
  tier1_scorable: number;
  tier1_passed: number;
  tier1_detail: string;
  judge: JudgeScores | null;
  judge_rationale: string;
  cost_usd: number;
  duration_ms: number;
}

export interface CalibrationDimension {
  dimension: string;
  n: number;
  exact_pct: number;
  within1_pct: number;
  kappa: number;
  matrix: number[][];
}

/** GET /api/evals/calibration — human-vs-judge agreement report. */
export interface CalibrationReport {
  available: boolean;
  n: number;
  eval_run_id: string;
  dimensions: CalibrationDimension[];
  overall: { n: number; exact_pct: number; within1_pct: number; kappa: number } | null;
}
