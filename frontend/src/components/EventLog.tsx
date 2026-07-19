import { Box, Chip, Paper, Stack, Typography } from "@mui/material";
import type { AgentEvent } from "../lib/types";

const AGENT_COLORS: Record<string, "default" | "primary" | "secondary" | "warning" | "success"> = {
  system: "default",
  planner: "secondary",
  analyst: "primary",
  critic: "warning",
  composer: "success",
};

const step = (e: AgentEvent) =>
  e.payload.step_id !== undefined ? `[step ${e.payload.step_id}] ` : "";

function summary(e: AgentEvent): string {
  switch (e.type) {
    case "run_started":
      return `run started — "${String(e.payload.question ?? "")}"`;
    case "llm_call":
      if (e.payload.plan !== undefined)
        return `planned ${(e.payload.plan as unknown[]).length} step(s)`;
      if (e.payload.answer !== undefined) return "composed final answer";
      return e.payload.action === "run_code"
        ? `${step(e)}iteration ${e.payload.iteration}: decided to run code`
        : `${step(e)}iteration ${e.payload.iteration}: finished analysis`;
    case "tool_call":
      return `${step(e)}sandbox exec (exit=${e.payload.exit_code}${e.payload.timed_out ? ", TIMED OUT" : ""})`;
    case "verdict":
      return `${e.payload.status} — claim ${e.payload.claim_id}${
        e.payload.reason ? ` (${e.payload.reason})` : ""
      }`;
    case "handoff":
      return e.payload.retry_steps !== undefined
        ? `retry → step(s) ${(e.payload.retry_steps as number[]).join(", ")}`
        : `fan-out → ${(e.payload.steps as number[]).length} analyst(s)`;
    case "run_finished":
      return "run finished";
    case "error":
      return `error: ${String(e.payload.error ?? "")}`;
    default:
      return e.type;
  }
}

function detail(e: AgentEvent): string | null {
  if (e.type === "tool_call") {
    const code = String(e.payload.code ?? "");
    const stdout = String(e.payload.stdout ?? "");
    return `${code}\n── stdout ──\n${stdout}`.trim();
  }
  return null;
}

export function EventLog({ events }: { events: AgentEvent[] }) {
  if (events.length === 0) return null;
  return (
    <Stack spacing={1}>
      {events.map((e) => (
        <Paper key={e.span_id} variant="outlined" sx={{ p: 1.5 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip size="small" label={e.agent} color={AGENT_COLORS[e.agent] ?? "default"} />
            <Typography variant="body2" sx={{ flex: 1 }}>
              {summary(e)}
            </Typography>
            {e.duration_ms > 0 && (
              <Typography variant="caption" color="text.secondary">
                {(e.duration_ms / 1000).toFixed(1)}s
              </Typography>
            )}
          </Stack>
          {detail(e) && (
            <Box
              component="pre"
              sx={{
                mt: 1,
                p: 1,
                bgcolor: "grey.900",
                color: "grey.100",
                borderRadius: 1,
                fontSize: 12,
                overflow: "auto",
                maxHeight: 240,
                whiteSpace: "pre-wrap",
              }}
            >
              {detail(e)}
            </Box>
          )}
        </Paper>
      ))}
    </Stack>
  );
}
