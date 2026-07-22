import CloseFullscreenRoundedIcon from "@mui/icons-material/CloseFullscreenRounded";
import OpenInFullRoundedIcon from "@mui/icons-material/OpenInFullRounded";
import TableChartOutlinedIcon from "@mui/icons-material/TableChartOutlined";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { Box, Button, IconButton, Paper, Stack, Tooltip, Typography } from "@mui/material";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { useRef, useState } from "react";
import type { Dataset } from "../lib/types";
import { MONO } from "../theme";

interface Props {
  dataset: Dataset | null;
  uploading: boolean;
  onUpload: (file: File) => void;
  wide?: boolean;
  onToggleWide?: () => void;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Stack direction="row" spacing={0.75} alignItems="baseline">
      <Typography sx={{ fontFamily: MONO, fontSize: "0.82rem", color: "text.primary", fontWeight: 600 }}>
        {value}
      </Typography>
      <Typography sx={{ fontSize: "0.72rem", color: "text.disabled" }}>{label}</Typography>
    </Stack>
  );
}

const gridSx = {
  border: "none",
  color: "var(--ink-2)",
  fontFamily: MONO,
  fontSize: "0.72rem",
  "--DataGrid-rowBorderColor": "var(--border)",
  "& .MuiDataGrid-columnHeaders": { borderColor: "var(--border)" },
  "& .MuiDataGrid-columnHeader": { bgcolor: "var(--surface-2)" },
  "& .MuiDataGrid-columnHeaderTitle": {
    fontFamily: "'Inter Variable', sans-serif",
    fontSize: "0.7rem",
    fontWeight: 600,
    color: "var(--ink-3)",
    letterSpacing: "0.02em",
  },
  "& .MuiDataGrid-cell": { borderColor: "var(--border)" },
  "& .MuiDataGrid-columnSeparator": { display: "none" },
  "& .MuiDataGrid-row:hover": { bgcolor: "var(--surface-2)" },
  "& .MuiDataGrid-filler, & .MuiDataGrid-scrollbarFiller": { bgcolor: "transparent" },
} as const;

export function DatasetPanel({ dataset, uploading, onUpload, wide, onToggleWide }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const columns: GridColDef[] = (dataset?.profile.columns ?? []).map((c) => ({
    field: c.name,
    headerName: `${c.name}  ·  ${c.dtype}`,
    flex: 1,
    minWidth: 130,
    sortable: false,
  }));
  const rows = (dataset?.profile.preview ?? []).map((r, i) => ({ id: i, ...r }));

  const pick = () => inputRef.current?.click();

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1.5 }}>
        <Typography variant="subtitle2">Dataset</Typography>
        {dataset && (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Button startIcon={<UploadFileIcon />} size="small" onClick={pick} disabled={uploading}>
              {uploading ? "Uploading…" : "Replace"}
            </Button>
            {onToggleWide && (
              <Tooltip title={wide ? "Collapse to reading width" : "Expand to full width"}>
                <IconButton size="small" onClick={onToggleWide} sx={{ color: "text.secondary" }}>
                  {wide ? (
                    <CloseFullscreenRoundedIcon sx={{ fontSize: 17 }} />
                  ) : (
                    <OpenInFullRoundedIcon sx={{ fontSize: 17 }} />
                  )}
                </IconButton>
              </Tooltip>
            )}
          </Stack>
        )}
      </Stack>
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

      {dataset ? (
        <>
          <Stack direction="row" spacing={2.5} sx={{ mb: 1.5, flexWrap: "wrap" }} useFlexGap>
            <Stat label="" value={dataset.name} />
            <Stat label="rows" value={dataset.profile.rows.toLocaleString()} />
            <Stat label="columns" value={String(dataset.profile.columns.length)} />
          </Stack>
          <Box sx={{ height: wide ? 460 : 280, transition: "height 300ms var(--ease-out)" }}>
            <DataGrid rows={rows} columns={columns} density="compact" hideFooter disableColumnMenu sx={gridSx} />
          </Box>
        </>
      ) : (
        <Box
          onClick={pick}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) onUpload(file);
          }}
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 1.25,
            py: 5,
            px: 2,
            textAlign: "center",
            borderRadius: 2.5,
            border: "1.5px dashed",
            borderColor: dragging ? "primary.main" : "var(--border-strong)",
            bgcolor: dragging ? "rgba(122,162,247,0.06)" : "var(--surface-2)",
            cursor: "pointer",
            transition: "border-color 150ms ease, background-color 150ms ease",
            "&:hover": { borderColor: "primary.main", bgcolor: "rgba(122,162,247,0.04)" },
          }}
        >
          <TableChartOutlinedIcon sx={{ fontSize: 30, color: "var(--ink-3)" }} />
          <Typography variant="body2" sx={{ color: "text.primary", fontWeight: 600 }}>
            Drop a CSV, or click to upload
          </Typography>
          <Typography variant="caption" sx={{ color: "text.disabled", maxWidth: "42ch" }}>
            The profile — columns, types, null counts — is computed on upload and handed to the
            agents. Try one of the bundled samples in <Box component="span" sx={{ fontFamily: MONO }}>data/samples/</Box>.
          </Typography>
          {uploading && (
            <Typography variant="caption" sx={{ color: "primary.main" }}>
              uploading…
            </Typography>
          )}
        </Box>
      )}
    </Paper>
  );
}
