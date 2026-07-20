import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Alert, Chip, Container, Paper, Stack, Typography } from "@mui/material";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { LineChart } from "@mui/x-charts/LineChart";
import { getCalibration, getEvalRun, listEvalRuns } from "../lib/api";
import type { EvalResultRow, EvalRunSummary } from "../lib/types";
import { CalibrationGrid } from "../components/CalibrationGrid";

const runColumns: GridColDef<EvalRunSummary>[] = [
  {
    field: "created_at",
    headerName: "When",
    width: 170,
    valueFormatter: (value: number) => new Date(value * 1000).toLocaleString(),
  },
  { field: "label", headerName: "Label", width: 140 },
  { field: "git_sha", headerName: "SHA", width: 90 },
  { field: "config_hash", headerName: "Config", width: 110 },
  {
    field: "tier1_pass_rate",
    headerName: "Tier 1",
    width: 100,
    valueFormatter: (value: number | null) =>
      value == null ? "—" : `${(value * 100).toFixed(0)}%`,
  },
  {
    field: "judge_avg",
    headerName: "Judge",
    width: 90,
    valueFormatter: (value: number | null) => (value == null ? "—" : value.toFixed(2)),
  },
  {
    field: "cost_usd",
    headerName: "Cost",
    width: 90,
    valueFormatter: (value: number) => `$${value.toFixed(3)}`,
  },
  {
    field: "duration_ms",
    headerName: "Duration",
    width: 100,
    valueFormatter: (value: number) => `${(value / 1000).toFixed(0)}s`,
  },
];

const resultColumns: GridColDef<EvalResultRow>[] = [
  { field: "question_id", headerName: "Question", width: 130 },
  { field: "dataset", headerName: "Dataset", width: 100 },
  {
    field: "tier1_passed",
    headerName: "Tier 1",
    width: 110,
    renderCell: ({ row }) =>
      row.tier1_scorable ? (
        <Chip
          size="small"
          variant="outlined"
          color={row.tier1_passed ? "success" : "error"}
          label={row.tier1_passed ? "pass" : "fail"}
        />
      ) : (
        <Chip size="small" variant="outlined" label="judge-only" />
      ),
  },
  {
    field: "judge",
    headerName: "Judge avg",
    width: 100,
    valueGetter: (_value, row) =>
      row.judge
        ? (
            (row.judge.clarity +
              row.judge.uncertainty_honesty +
              row.judge.chart_appropriateness +
              row.judge.methodological_soundness) /
            4
          ).toFixed(2)
        : "—",
  },
  { field: "tier1_detail", headerName: "Detail", flex: 1, minWidth: 220 },
];

export function EvalsScreen() {
  const runs = useQuery({ queryKey: ["evalRuns"], queryFn: listEvalRuns });
  const calibration = useQuery({ queryKey: ["calibration"], queryFn: getCalibration });
  const [selected, setSelected] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: ["evalRun", selected],
    queryFn: () => getEvalRun(selected as string),
    enabled: selected != null,
  });

  const series = [...(runs.data ?? [])].sort((a, b) => a.created_at - b.created_at);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={2}>
        <Typography variant="h5" fontWeight={700}>
          Evals
        </Typography>
        {runs.error != null && <Alert severity="error">{String(runs.error)}</Alert>}

        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Score over time
          </Typography>
          {series.length === 0 ? (
            <Typography color="text.secondary">
              No eval runs yet — run `python -m app.evals run` in the backend.
            </Typography>
          ) : (
            <LineChart
              height={280}
              xAxis={[
                {
                  scaleType: "point",
                  data: series.map(
                    (r) => `${new Date(r.created_at * 1000).toLocaleDateString()} ${r.git_sha}`,
                  ),
                },
              ]}
              yAxis={[
                { id: "pct", min: 0, max: 100, label: "tier-1 pass %" },
                { id: "judge", min: 1, max: 5, label: "judge avg" },
              ]}
              leftAxis="pct"
              rightAxis="judge"
              series={[
                {
                  yAxisKey: "pct",
                  label: "tier-1 pass %",
                  data: series.map((r) => (r.tier1_pass_rate == null ? null : r.tier1_pass_rate * 100)),
                },
                {
                  yAxisKey: "judge",
                  label: "judge avg (1-5)",
                  data: series.map((r) => r.judge_avg),
                },
              ]}
            />
          )}
        </Paper>

        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Eval runs
          </Typography>
          <DataGrid
            rows={runs.data ?? []}
            columns={runColumns}
            loading={runs.isLoading}
            density="compact"
            autoHeight
            disableRowSelectionOnClick
            onRowClick={({ row }) => setSelected(row.id)}
            initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
            pageSizeOptions={[10, 25]}
            sx={{ "& .MuiDataGrid-row": { cursor: "pointer" } }}
          />
        </Paper>

        {selected != null && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Results — {selected}
            </Typography>
            <DataGrid
              rows={detail.data?.results ?? []}
              getRowId={(row) => row.question_id}
              columns={resultColumns}
              loading={detail.isLoading}
              density="compact"
              autoHeight
              disableRowSelectionOnClick
              initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
              pageSizeOptions={[25, 50]}
            />
          </Paper>
        )}

        <CalibrationGrid report={calibration.data} loading={calibration.isLoading} />
      </Stack>
    </Container>
  );
}
