import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  InputBase,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import { AnswerPanel } from "../components/AnswerPanel";
import { CostMeter } from "../components/CostMeter";
import { DatasetPanel } from "../components/DatasetPanel";
import { EventLog } from "../components/EventLog";
import { useRunStream } from "../hooks/useRunStream";
import { createRun, getSuggestions, uploadDataset } from "../lib/api";
import type { Dataset } from "../lib/types";
import { useRunStore } from "../store/runStore";

const READING = 920;

export function Workbench() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [question, setQuestion] = useState("");
  const [wide, setWide] = useState(false);

  const { runId, status, events, answer, final, error, start } = useRunStore();
  useRunStream(runId);

  const upload = useMutation({ mutationFn: uploadDataset, onSuccess: setDataset });
  const ask = useMutation({
    mutationFn: () => createRun(dataset!.id, question),
    onSuccess: ({ run_id }) => start(run_id),
  });
  const suggestions = useQuery({
    queryKey: ["suggestions", dataset?.id],
    queryFn: () => getSuggestions(dataset!.id),
    enabled: Boolean(dataset),
    staleTime: Infinity, // one mini-model call per dataset
  });

  const busy = status === "running" || ask.isPending;
  const canAsk = Boolean(dataset) && question.trim().length > 0 && !busy;

  return (
    <Box sx={{ py: 5, px: { xs: 2, sm: 3, md: 4 } }}>
      {/* Hero — reading width. */}
      <Box sx={{ maxWidth: READING, mx: "auto", mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 750 }}>
          Workbench
        </Typography>
        <Typography sx={{ color: "text.secondary", mt: 0.5 }}>
          Ask a question in plain English. Watch a team of agents plan, run Python, and verify every
          number before you see it.
        </Typography>
      </Box>

      {/* Dataset — expands to full width on toggle. */}
      <Box
        sx={{
          maxWidth: wide ? "100%" : READING,
          mx: "auto",
          mb: 3,
          transition: "max-width 320ms var(--ease-out)",
        }}
      >
        <DatasetPanel
          dataset={dataset}
          uploading={upload.isPending}
          onUpload={(f) => upload.mutate(f)}
          wide={wide}
          onToggleWide={() => setWide((w) => !w)}
        />
        {upload.isError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {String(upload.error)}
          </Alert>
        )}
      </Box>

      {/* Ask + run — reading width. */}
      <Stack spacing={3} sx={{ maxWidth: READING, mx: "auto" }}>
        {/* Command-bar-style ask input. */}
        <Paper
          variant="outlined"
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            px: 1.5,
            py: 0.5,
            bgcolor: "var(--surface-2)",
            transition: "border-color 150ms ease, box-shadow 150ms ease",
            "&:focus-within": {
              borderColor: "primary.main",
              boxShadow: "0 0 0 3px rgba(122,162,247,0.15)",
            },
            opacity: dataset ? 1 : 0.6,
          }}
        >
          <Box sx={{ color: "var(--ink-3)", fontFamily: "monospace", fontWeight: 700, pl: 0.5 }}>
            ›
          </Box>
          <InputBase
            fullWidth
            multiline
            maxRows={4}
            placeholder={dataset ? "Ask a question about this dataset…" : "Upload a dataset first"}
            disabled={!dataset || busy}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && canAsk) {
                e.preventDefault();
                ask.mutate();
              }
            }}
            sx={{ fontSize: "0.95rem", py: 1 }}
          />
          <Button
            variant="contained"
            disabled={!canAsk}
            onClick={() => ask.mutate()}
            endIcon={
              busy ? (
                <CircularProgress size={15} color="inherit" />
              ) : (
                <ArrowForwardRoundedIcon sx={{ fontSize: 18 }} />
              )
            }
            sx={{ flexShrink: 0 }}
          >
            Ask
          </Button>
        </Paper>
        {ask.isError && <Alert severity="error">{String(ask.error)}</Alert>}

        {dataset && status === "idle" && (suggestions.data?.suggestions.length ?? 0) > 0 && (
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
            <Typography variant="caption" sx={{ color: "text.disabled", mr: 0.25 }}>
              Try
            </Typography>
            {suggestions.data!.suggestions.map((q) => (
              <Box
                key={q}
                component="button"
                onClick={() => setQuestion(q)}
                sx={{
                  font: "inherit",
                  fontSize: "0.78rem",
                  color: "text.secondary",
                  bgcolor: "var(--surface-2)",
                  border: "1px solid var(--border)",
                  borderRadius: 999,
                  px: 1.25,
                  py: 0.5,
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "color 150ms ease, border-color 150ms ease, background-color 150ms ease",
                  "&:hover": {
                    color: "text.primary",
                    borderColor: "primary.main",
                    bgcolor: "rgba(122,162,247,0.06)",
                  },
                }}
              >
                {q}
              </Box>
            ))}
          </Stack>
        )}

        {status !== "idle" && (
          <Stack spacing={2}>
            <Stack direction="row" spacing={1.25} alignItems="center">
              {status === "running" && (
                <Box
                  className="pulse"
                  sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: "primary.main" }}
                />
              )}
              <Typography variant="subtitle2" sx={{ color: "text.secondary" }}>
                Trace · {status === "running" ? "live" : status}
              </Typography>
              <Box sx={{ flexGrow: 1 }} />
              {runId && (
                <Button size="small" component={RouterLink} to={`/runs/${runId}`}>
                  Open run view →
                </Button>
              )}
            </Stack>

            <CostMeter events={events} />

            <Paper variant="outlined" sx={{ p: 2 }}>
              <EventLog events={events} />
            </Paper>

            {status === "finished" &&
              (final ? (
                <AnswerPanel final={final} />
              ) : (
                <Alert severity="success">{answer}</Alert>
              ))}
            {status === "error" && <Alert severity="error">{error}</Alert>}
          </Stack>
        )}
      </Stack>
    </Box>
  );
}
