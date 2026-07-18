import type { Dataset } from "./types";

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
