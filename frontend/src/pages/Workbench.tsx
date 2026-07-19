import SendIcon from "@mui/icons-material/Send";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Container,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { AnswerPanel } from "../components/AnswerPanel";
import { DatasetPanel } from "../components/DatasetPanel";
import { EventLog } from "../components/EventLog";
import { useRunStream } from "../hooks/useRunStream";
import { createRun, uploadDataset } from "../lib/api";
import type { Dataset } from "../lib/types";
import { useRunStore } from "../store/runStore";

export function Workbench() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [question, setQuestion] = useState("");

  const { runId, status, events, answer, final, error, start } = useRunStore();
  useRunStream(runId);

  const upload = useMutation({ mutationFn: uploadDataset, onSuccess: setDataset });
  const ask = useMutation({
    mutationFn: () => createRun(dataset!.id, question),
    onSuccess: ({ run_id }) => start(run_id),
  });

  const busy = status === "running" || ask.isPending;

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Stack spacing={3}>
        <Box>
          <Typography variant="h4" fontWeight={700}>
            tracelab
          </Typography>
          <Typography color="text.secondary">
            An agentic data analyst you can watch think.
          </Typography>
        </Box>

        <DatasetPanel
          dataset={dataset}
          uploading={upload.isPending}
          onUpload={(f) => upload.mutate(f)}
        />
        {upload.isError && <Alert severity="error">{String(upload.error)}</Alert>}

        <Paper variant="outlined" sx={{ p: 2 }}>
          <Stack direction="row" spacing={1}>
            <TextField
              fullWidth
              size="small"
              placeholder={
                dataset ? "Ask a question about this dataset…" : "Upload a dataset first"
              }
              disabled={!dataset || busy}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && dataset && question.trim() && !busy) ask.mutate();
              }}
            />
            <Button
              variant="contained"
              endIcon={busy ? <CircularProgress size={16} color="inherit" /> : <SendIcon />}
              disabled={!dataset || !question.trim() || busy}
              onClick={() => ask.mutate()}
            >
              Ask
            </Button>
          </Stack>
        </Paper>
        {ask.isError && <Alert severity="error">{String(ask.error)}</Alert>}

        {status !== "idle" && (
          <Stack spacing={2}>
            <Typography variant="h6">Run</Typography>
            <EventLog events={events} />
            {status === "running" && (
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={16} />
                <Typography variant="body2" color="text.secondary">
                  agents working…
                </Typography>
              </Stack>
            )}
            {status === "finished" &&
              (final ? <AnswerPanel final={final} /> : <Alert severity="success">{answer}</Alert>)}
            {status === "error" && <Alert severity="error">{error}</Alert>}
          </Stack>
        )}
      </Stack>
    </Container>
  );
}
