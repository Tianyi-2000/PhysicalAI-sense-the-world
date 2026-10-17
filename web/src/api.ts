export type Backend = "npu" | "cpu";

/** One classification (see linux/classify.py). */
export interface ClassifyResult {
  top: [label: string, prob: number][];
  backend: Backend;
  preprocess_ms: number;
  model_ms: number;
  total_ms: number;
}

/** One photo the server has classified (see linux/server.py). */
export interface Shot {
  id: number;
  time: number;
  image: string;
  result: ClassifyResult;
}

export interface Session {
  count: number;
  avg_model_ms: number;
  avg_total_ms: number;
  shots: Shot[];
}

export interface Info {
  backend: Backend;
  url: string;
}

async function json<T>(res: Response): Promise<T> {
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data;
}

export const getInfo = () => fetch("/info").then((res) => json<Info>(res));

export const getSession = () => fetch("/shots").then((res) => json<Session>(res));

export async function classify(photo: Blob): Promise<{ shot: Shot; roundTripMs: number }> {
  const start = performance.now();
  const shot = await fetch("/classify", { method: "POST", body: photo }).then((res) => json<Shot>(res));
  return { shot, roundTripMs: performance.now() - start };
}

/** "tabby, tabby cat" -> "tabby" */
export const shortLabel = (label: string) => label.split(",")[0];
