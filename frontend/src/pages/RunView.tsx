import ReplayIcon from "@mui/icons-material/Replay";
import { Alert, Box, Button, Chip, Container, Stack, Typography } from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useNavigate, useParams, Link as RouterLink } from "react-router-dom";
import { AgentGraph } from "../components/AgentGraph";
import { AnswerPanel } from "../components/AnswerPanel";
import { CostMeter } from "../components/CostMeter";
import { SpanInspector } from "../components/SpanInspector";
import { useRunEvents } from "../hooks/useRunEvents";
import { replayRun } from "../lib/api";
import { buildAgentGraph, type AgentNodeModel } from "../lib/graphModel";

const statusColor = { running: "info", finished: "success", error: "error" } as const;

export function RunView() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { run, events } = useRunEvents(runId);
  const [selection, setSelection] = useState<AgentNodeModel | null>(null);
  const { nodes, edges } = useMemo(() => buildAgentGraph(events), [events]);
  const replay = useMutation({
    mutationFn: replayRun,
    onSuccess: ({ run_id }) => navigate(`/runs/${run_id}`),
  });

  if (!run) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography color="text.secondary">loading run…</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={2}>
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
          <Typography variant="h6" fontWeight={700} sx={{ flexGrow: 1, minWidth: 200 }}>
            {run.question}
          </Typography>
          <Chip size="small" variant="outlined" color={statusColor[run.status]} label={run.status} />
          {run.replay_of && (
            <Chip
              size="small"
              variant="outlined"
              label={`replay of ${run.replay_of}`}
              component={RouterLink}
              to={`/runs/${run.replay_of}`}
              clickable
            />
          )}
          <Button
            size="small"
            startIcon={<ReplayIcon />}
            disabled={run.status === "running" || replay.isPending}
            onClick={() => runId && replay.mutate(runId)}
          >
            Replay offline
          </Button>
        </Stack>
        {replay.isError && <Alert severity="error">{String(replay.error)}</Alert>}

        <CostMeter events={events} />
        <AgentGraph nodes={nodes} edges={edges} onSelect={setSelection} />

        <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems="flex-start">
          <Box sx={{ flex: 1, minWidth: 0, width: "100%" }}>
            <SpanInspector events={events} selection={selection} />
          </Box>
          <Box sx={{ flex: 1, minWidth: 0, width: "100%" }}>
            {run.result && <AnswerPanel final={run.result} />}
          </Box>
        </Stack>
      </Stack>
    </Container>
  );
}
