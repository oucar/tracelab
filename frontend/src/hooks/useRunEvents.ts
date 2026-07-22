import { useCallback, useEffect, useState } from "react";
import { getRun } from "../lib/api";
import type { AgentEvent, RunDetail } from "../lib/types";

const EVENT_TYPES = [
  "run_started",
  "llm_call",
  "tool_call",
  "handoff",
  "verdict",
  "answer_chunk",
  "run_finished",
  "error",
];

/**
 * One event source for the run view: persisted spans first, then — while the
 * run is live — the SSE stream, deduped by span_id (the bus replays history).
 */
export function useRunEvents(runId: string | undefined) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);

  const refresh = useCallback(async () => {
    if (!runId) return;
    const detail = await getRun(runId);
    setRun(detail);
    setEvents(detail.spans);
  }, [runId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!runId || run?.status !== "running") return;
    const source = new EventSource(`/api/runs/${runId}/events`);
    const onMessage = (e: MessageEvent) => {
      const event = JSON.parse(e.data) as AgentEvent;
      setEvents((prev) =>
        prev.some((p) => p.span_id === event.span_id)
          ? prev.map((p) => (p.span_id === event.span_id ? event : p))
          : [...prev, event],
      );
      if (event.type === "run_finished" || event.type === "error") {
        source.close();
        void refresh(); // pick up final status/result from the store
      }
    };
    EVENT_TYPES.forEach((t) => source.addEventListener(t, onMessage));
    source.onerror = () => source.close();
    return () => source.close();
  }, [runId, run?.status, refresh]);

  return { run, events };
}
