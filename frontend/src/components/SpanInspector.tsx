import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import type { AgentNodeModel } from "../lib/graphModel";
import type { AgentEvent } from "../lib/types";

const PRE_KEYS = ["code", "stdout", "stderr", "answer"] as const;

function PayloadView({ payload }: { payload: Record<string, unknown> }) {
  const rest = Object.fromEntries(
    Object.entries(payload).filter(([k, v]) => !PRE_KEYS.includes(k as never) && v !== ""),
  );
  return (
    <Stack spacing={1}>
      {PRE_KEYS.map(
        (key) =>
          typeof payload[key] === "string" &&
          payload[key] !== "" && (
            <Box key={key}>
              <Typography variant="caption" color="text.secondary">
                {key}
              </Typography>
              <Box
                component="pre"
                sx={{
                  m: 0, p: 1, bgcolor: "background.default", borderRadius: 1,
                  fontSize: 12, overflow: "auto", maxHeight: 240,
                }}
              >
                {payload[key] as string}
              </Box>
            </Box>
          ),
      )}
      {Object.keys(rest).length > 0 && (
        <Box
          component="pre"
          sx={{
            m: 0, p: 1, bgcolor: "background.default", borderRadius: 1,
            fontSize: 12, overflow: "auto", maxHeight: 240,
          }}
        >
          {JSON.stringify(rest, null, 2)}
        </Box>
      )}
    </Stack>
  );
}

export function SpanInspector({
  events,
  selection,
}: {
  events: AgentEvent[];
  selection: AgentNodeModel | null;
}) {
  if (!selection) {
    return (
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography color="text.secondary">
          Click a node in the agent graph to inspect its spans.
        </Typography>
      </Paper>
    );
  }
  const spans = events.filter((e) =>
    selection.agent === "analyst"
      ? e.agent === "analyst" && e.payload.step_id === selection.stepId
      : e.agent === selection.agent,
  );
  return (
    <Paper variant="outlined" sx={{ p: 1 }}>
      <Typography variant="subtitle2" sx={{ px: 1, py: 0.5 }}>
        {selection.label} — {spans.length} span{spans.length === 1 ? "" : "s"}
      </Typography>
      {spans.map((e) => (
        <Accordion key={e.span_id} disableGutters variant="outlined" sx={{ bgcolor: "transparent" }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
              <Chip size="small" variant="outlined" label={e.type} />
              <Typography variant="caption" color="text.secondary" noWrap>
                {e.duration_ms}ms
                {e.tokens_in + e.tokens_out > 0 && ` · ${e.tokens_in + e.tokens_out} tok`}
                {e.cost_usd > 0 && ` · $${e.cost_usd.toFixed(4)}`}
              </Typography>
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <PayloadView payload={e.payload} />
          </AccordionDetails>
        </Accordion>
      ))}
    </Paper>
  );
}
