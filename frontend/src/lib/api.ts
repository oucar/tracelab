import type {
  AppConfig,
  CalibrationReport,
  Dataset,
  EvalResultRow,
  EvalRunSummary,
  RunDetail,
  RunSummary,
} from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export async function uploadDataset(file: File): Promise<Dataset> {
  const form = new FormData();
  form.append("file", file);
  return json(await fetch("/api/datasets", { method: "POST", body: form }));
}

export async function listDatasets(): Promise<Dataset[]> {
  return json(await fetch("/api/datasets"));
}

export async function createRun(datasetId: string, question: string): Promise<{ run_id: string }> {
  return json(
    await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, question }),
    }),
  );
}

export async function listRuns(): Promise<RunSummary[]> {
  return json(await fetch("/api/runs"));
}

export async function getRun(runId: string): Promise<RunDetail> {
  return json(await fetch(`/api/runs/${runId}`));
}

export async function replayRun(runId: string): Promise<{ run_id: string }> {
  return json(await fetch(`/api/runs/${runId}/replay`, { method: "POST" }));
}

export async function getConfig(): Promise<AppConfig> {
  return json(await fetch("/api/config"));
}

export async function listEvalRuns(): Promise<EvalRunSummary[]> {
  return json(await fetch("/api/evals"));
}

export async function getEvalRun(
  id: string,
): Promise<{ run: EvalRunSummary; results: EvalResultRow[] }> {
  return json(await fetch(`/api/evals/${id}`));
}

export async function getCalibration(): Promise<CalibrationReport> {
  return json(await fetch("/api/evals/calibration"));
}

export async function getSuggestions(datasetId: string): Promise<{ suggestions: string[] }> {
  return json(await fetch(`/api/datasets/${datasetId}/suggestions`));
}
