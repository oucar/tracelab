import { useEffect } from "react";
import type { AgentEvent } from "../lib/types";
import { useRunStore } from "../store/runStore";

/** Subscribes to a run's SSE stream and folds every AgentEvent into the run store. */
export function useRunStream(runId: string | null) {
  const ingest = useRunStore((s) => s.ingest);

  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(`/api/runs/${runId}/events`);

    const onMessage = (e: MessageEvent) => {
      const event = JSON.parse(e.data) as AgentEvent;
      ingest(event);
      if (event.type === "run_finished" || event.type === "error") source.close();
    };

    // The backend names its SSE events by type; listen to all of them.
    const types = [
      "run_started",
      "llm_call",
      "tool_call",
      "handoff",
      "verdict",
      "answer_chunk",
      "run_finished",
      "error",
    ];
    types.forEach((t) => source.addEventListener(t, onMessage));
    source.onerror = () => source.close();

    return () => source.close();
  }, [runId, ingest]);
}
