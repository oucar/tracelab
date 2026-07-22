import { Box, Stack, Typography } from "@mui/material";
import type { AgentEvent } from "../lib/types";
import { AGENT, MONO } from "../theme";

const agentColor = (agent: string) => AGENT[agent] ?? AGENT.system;

const step = (e: AgentEvent) =>
  e.payload.step_id !== undefined ? `step ${e.payload.step_id} · ` : "";

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
      return `${step(e)}sandbox exec (exit ${e.payload.exit_code}${e.payload.timed_out ? " · TIMED OUT" : ""})`;
    case "verdict":
      return `${e.payload.status} — claim ${e.payload.claim_id}${
        e.payload.reason ? ` (${e.payload.reason})` : ""
      }`;
    case "handoff":
      if (e.payload.route !== undefined)
        return `route: ${String(e.payload.route)} → ${String(e.payload.to ?? "")}`;
      if (e.payload.retry_steps !== undefined)
        return `retry → step(s) ${(e.payload.retry_steps as number[]).join(", ")}`;
      if (e.payload.steps !== undefined)
        return `fan-out → ${(e.payload.steps as number[]).length} analyst(s)`;
      return "handoff";
    case "run_finished":
      return "run finished";
    case "error":
      return `error: ${String(e.payload.error ?? "")}`;
    default:
      return e.type;
  }
}

function terminal(e: AgentEvent): { code: string; stdout: string } | null {
  if (e.type !== "tool_call") return null;
  return { code: String(e.payload.code ?? ""), stdout: String(e.payload.stdout ?? "") };
}

function Marker({ color, first, last, active }: {
  color: string;
  first: boolean;
  last: boolean;
  active: boolean;
}) {
  return (
    <Box sx={{ position: "relative", width: 26, alignSelf: "stretch", flexShrink: 0 }}>
      {/* the continuous spine, capped at the first/last dot */}
      {!first && (
        <Box sx={{ position: "absolute", left: 12, top: 0, height: 15, width: "1.5px", bgcolor: "var(--border)" }} />
      )}
      {!last && (
        <Box sx={{ position: "absolute", left: 12, top: 15, bottom: 0, width: "1.5px", bgcolor: "var(--border)" }} />
      )}
      <Box
        className={active ? "pulse" : undefined}
        sx={{
          position: "absolute",
          left: 6,
          top: 9,
          width: 13,
          height: 13,
          borderRadius: "50%",
          bgcolor: color,
          border: "2.5px solid var(--bg)",
          boxShadow: active ? "none" : `0 0 0 1px ${color}55`,
        }}
      />
    </Box>
  );
}

export function EventLog({ events }: { events: AgentEvent[] }) {
  if (events.length === 0) return null;
  const streaming = !events.some((e) => e.type === "run_finished" || e.type === "error");
  const activeId = streaming ? events[events.length - 1].span_id : null;

  return (
    <Box>
      {events.map((e, i) => {
        const color = agentColor(e.agent);
        const term = terminal(e);
        return (
          <Box
            key={e.span_id}
            className="trace-in"
            sx={{ display: "flex", gap: 1.25, alignItems: "stretch" }}
          >
            <Marker
              color={color}
              first={i === 0}
              last={i === events.length - 1}
              active={e.span_id === activeId}
            />
            <Box sx={{ flex: 1, minWidth: 0, pb: 1.75 }}>
              <Stack direction="row" spacing={1} alignItems="baseline">
                <Typography
                  sx={{
                    fontFamily: MONO,
                    fontSize: "0.66rem",
                    fontWeight: 600,
                    letterSpacing: "0.04em",
                    textTransform: "uppercase",
                    color,
                    flexShrink: 0,
                  }}
                >
                  {e.agent}
                </Typography>
                <Typography variant="body2" sx={{ flex: 1, color: "text.primary", minWidth: 0 }}>
                  {summary(e)}
                </Typography>
                {e.duration_ms > 0 && (
                  <Typography
                    sx={{ fontFamily: MONO, fontSize: "0.68rem", color: "text.disabled", flexShrink: 0 }}
                  >
                    {(e.duration_ms / 1000).toFixed(1)}s
                  </Typography>
                )}
              </Stack>

              {term && (term.code || term.stdout) && (
                <Box
                  sx={{
                    mt: 1,
                    borderRadius: 2,
                    border: "1px solid var(--border)",
                    overflow: "hidden",
                    bgcolor: "#0c0e13",
                  }}
                >
                  <Box
                    sx={{
                      px: 1.25,
                      py: 0.5,
                      borderBottom: "1px solid var(--border)",
                      display: "flex",
                      gap: 0.75,
                      alignItems: "center",
                    }}
                  >
                    {["#f87171", "#fbbf24", "#4ade80"].map((c) => (
                      <Box key={c} sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: c, opacity: 0.7 }} />
                    ))}
                    <Typography sx={{ fontFamily: MONO, fontSize: "0.62rem", color: "text.disabled", ml: 0.5 }}>
                      sandbox.py
                    </Typography>
                  </Box>
                  <Box
                    component="pre"
                    sx={{
                      m: 0,
                      p: 1.25,
                      fontFamily: MONO,
                      fontSize: "0.72rem",
                      lineHeight: 1.55,
                      color: "var(--ink-2)",
                      overflow: "auto",
                      maxHeight: 260,
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    <Box component="span" sx={{ color: "var(--ink-1)" }}>{term.code}</Box>
                    {term.stdout && (
                      <>
                        {"\n"}
                        <Box component="span" sx={{ color: "var(--ink-3)" }}>{"── stdout ──\n"}</Box>
                        <Box component="span" sx={{ color: "var(--agent-analyst)" }}>{term.stdout}</Box>
                      </>
                    )}
                  </Box>
                </Box>
              )}
            </Box>
          </Box>
        );
      })}
    </Box>
  );
}
