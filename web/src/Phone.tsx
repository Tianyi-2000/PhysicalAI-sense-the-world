import { useState } from "react";
import { classify, type ClassifyResult, type Info } from "./api";
import { Pipeline, Snapshot, Verdict } from "./components";

interface Latest {
  result: ClassifyResult;
  roundTripMs: number;
}

/** Phone view: take a photo, see what the board thinks it is. */
export default function Phone({ info }: { info: Info | null }) {
  const [photo, setPhoto] = useState<string | null>(null);
  const [latest, setLatest] = useState<Latest | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onPhoto(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (photo) URL.revokeObjectURL(photo);
    setPhoto(URL.createObjectURL(file)); // show the photo right away, scan while we wait
    setLatest(null);
    setError(null);
    setBusy(true);
    try {
      const { shot, roundTripMs } = await classify(file);
      setLatest({ result: shot.result, roundTripMs });
    } catch (err) {
      if (err instanceof TypeError) setError("Can't reach the board. Check the Wi-Fi.");
      else setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="phone">
      <section className="card">
        <label className={`shoot ${busy ? "busy" : ""}`}>
          {busy ? "Thinking..." : photo ? "Take another photo" : "Take a photo"}
          <input type="file" accept="image/*" capture="environment" onChange={onPhoto} disabled={busy} />
        </label>
        {error && <p className="error">{error}</p>}
        {photo && <Snapshot image={photo} scanning={busy} />}
        {latest && <Verdict result={latest.result} />}
      </section>
      <section className="card pipeline-card">
        <h2>Pipeline</h2>
        <Pipeline
          result={latest?.result}
          roundTripMs={latest?.roundTripMs}
          showUpload
          busy={busy}
          backend={info?.backend ?? null}
        />
      </section>
    </main>
  );
}
