import { create } from "zustand";
import type { AgentEvent } from "../lib/types";

export type RunStatus = "idle" | "running" | "finished" | "error";

interface RunStoreState {
  runId: string | null;
  status: RunStatus;
  events: AgentEvent[];
  answer: string;
  error: string | null;
  start: (runId: string) => void;
  ingest: (event: AgentEvent) => void;
  reset: () => void;
}

/**
 * The single live-run store. The SSE hook folds AgentEvents into it;
 * the event log, the (future) agent graph, and the cost ticker all derive
 * from this one place.
 */
export const useRunStore = create<RunStoreState>((set) => ({
  runId: null,
  status: "idle",
  events: [],
  answer: "",
  error: null,

  start: (runId) => set({ runId, status: "running", events: [], answer: "", error: null }),

  ingest: (event) =>
    set((s) => {
      const next: Partial<RunStoreState> = { events: [...s.events, event] };
      if (event.type === "run_finished") {
        next.status = "finished";
        next.answer = String(event.payload.answer ?? "");
      } else if (event.type === "error") {
        next.status = "error";
        next.error = String(event.payload.error ?? "unknown error");
      }
      return next;
    }),

  reset: () => set({ runId: null, status: "idle", events: [], answer: "", error: null }),
}));

export const selectTotalCost = (s: { events: AgentEvent[] }) =>
  s.events.reduce((acc, e) => acc + e.cost_usd, 0);
