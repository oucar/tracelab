import UploadFileIcon from "@mui/icons-material/UploadFile";
import { Box, Button, Chip, Paper, Stack, Typography } from "@mui/material";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { useRef } from "react";
import type { Dataset } from "../lib/types";

interface Props {
  dataset: Dataset | null;
  uploading: boolean;
  onUpload: (file: File) => void;
}

export function DatasetPanel({ dataset, uploading, onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const columns: GridColDef[] = (dataset?.profile.columns ?? []).map((c) => ({
    field: c.name,
    headerName: `${c.name} (${c.dtype})`,
    flex: 1,
    minWidth: 120,
    sortable: false,
  }));

  const rows = (dataset?.profile.preview ?? []).map((r, i) => ({ id: i, ...r }));

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
        <Typography variant="h6">Dataset</Typography>
        <Button
          startIcon={<UploadFileIcon />}
          variant="contained"
          size="small"
          disabled={uploading}
          onClick={() => inputRef.current?.click()}
        >
          {uploading ? "Uploading…" : "Upload CSV"}
        </Button>
        <input
          ref={inputRef}
          hidden
          type="file"
          accept=".csv"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = "";
          }}
        />
      </Stack>

      {dataset ? (
        <>
          <Stack direction="row" spacing={1} sx={{ mb: 1, flexWrap: "wrap" }}>
            <Chip size="small" label={dataset.name} color="primary" />
            <Chip size="small" label={`${dataset.profile.rows.toLocaleString()} rows`} />
            <Chip size="small" label={`${dataset.profile.columns.length} columns`} />
          </Stack>
          <Box sx={{ height: 300 }}>
            <DataGrid
              rows={rows}
              columns={columns}
              density="compact"
              hideFooter
              disableColumnMenu
            />
          </Box>
        </>
      ) : (
        <Typography color="text.secondary" variant="body2">
          Upload a CSV to get started. The profile (columns, types, nulls) is computed on upload
          and handed to the agents.
        </Typography>
      )}
    </Paper>
  );
}
