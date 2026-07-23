import ReplayIcon from "@mui/icons-material/Replay";
import { Alert, Chip, Container, Stack, Typography } from "@mui/material";
import { DataGrid, GridActionsCellItem, type GridColDef } from "@mui/x-data-grid";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { listRuns, replayRun } from "../lib/api";
import type { RunSummary } from "../lib/types";

const statusColor = { running: "info", finished: "success", error: "error" } as const;

export function RunsDashboard() {
  const navigate = useNavigate();
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: listRuns,
    // Poll fast while anything is running, back off when everything is settled.
    refetchInterval: (query) =>
      (query.state.data ?? []).some((r) => r.status === "running") ? 2_000 : 10_000,
  });
  const replay = useMutation({
    mutationFn: replayRun,
    onSuccess: ({ run_id }) => navigate(`/runs/${run_id}`),
  });

  const columns: GridColDef<RunSummary>[] = [
    {
      field: "created_at",
      headerName: "When",
      width: 170,
      valueFormatter: (value: number) => new Date(value * 1000).toLocaleString(),
    },
    { field: "question", headerName: "Question", flex: 1, minWidth: 240 },
    {
      field: "status",
      headerName: "Status",
      width: 110,
      renderCell: ({ row }) => (
        <Chip size="small" variant="outlined" color={statusColor[row.status]} label={row.status} />
      ),
    },
    {
      field: "duration_ms",
      headerName: "Latency",
      width: 90,
      valueFormatter: (value: number) => (value ? `${(value / 1000).toFixed(1)}s` : "—"),
    },
    {
      field: "tokens",
      headerName: "Tokens",
      width: 90,
      valueGetter: (_value, row) => row.tokens_in + row.tokens_out,
    },
    {
      field: "cost_usd",
      headerName: "Cost",
      width: 100,
      valueFormatter: (value: number) => (value ? `$${value.toFixed(4)}` : "$0"),
    },
    {
      field: "claims_verified",
      headerName: "Verified",
      width: 90,
      valueGetter: (_value, row) =>
        row.claims_total ? `${row.claims_verified}/${row.claims_total}` : "—",
    },
    {
      field: "replay_of",
      headerName: "",
      width: 90,
      renderCell: ({ row }) =>
        row.replay_of ? <Chip size="small" variant="outlined" label="replay" /> : null,
    },
    {
      field: "actions",
      type: "actions",
      width: 60,
      getActions: ({ row }) => [
        <GridActionsCellItem
          key="replay"
          icon={<ReplayIcon fontSize="small" />}
          label="Replay offline"
          disabled={row.status === "running" || replay.isPending}
          onClick={() => replay.mutate(row.id)}
        />,
      ],
    },
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={2}>
        <Typography variant="h5" fontWeight={700}>
          Runs
        </Typography>
        {replay.isError && <Alert severity="error">{String(replay.error)}</Alert>}
        <DataGrid
          rows={runs.data ?? []}
          columns={columns}
          loading={runs.isLoading}
          density="compact"
          disableRowSelectionOnClick
          onRowClick={({ row }) => navigate(`/runs/${row.id}`)}
          initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
          pageSizeOptions={[25, 50]}
          sx={{ "& .MuiDataGrid-row": { cursor: "pointer" } }}
          autoHeight
        />
      </Stack>
    </Container>
  );
}
