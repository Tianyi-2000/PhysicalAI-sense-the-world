import { Fragment, useEffect, useState, type CSSProperties } from "react";
import { shortLabel, type Backend, type ClassifyResult, type Session, type Shot } from "./api";

const CONFIDENT = 0.3; // below this, show a hedged guess instead of a big answer

export function Snapshot({ image, scanning = false }: { image: string; scanning?: boolean }) {
  return (
    <div className={`snapshot ${scanning ? "scanning" : ""}`}>
      <img src={image} alt="" />
    </div>
  );
}

export function Verdict({ result }: { result: ClassifyResult }) {
  const [[label, prob], ...rest] = result.top;
  const sure = prob >= CONFIDENT;
  return (
    <div className={`verdict ${sure ? "" : "unsure"}`}>
      {!sure && <p className="hedge">Not sure. Best guess:</p>}
      <p className="label">{shortLabel(label)}</p>
      <Bar value={prob} />
      <ol>
        {rest.map(([name, p]) => (
          <li key={name}>
            <span>{shortLabel(name)}</span>
            <span>{(p * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Bar({ value }: { value: number }) {
  return (
    <div className="bar">
      <div style={{ width: `${value * 100}%` }} />
      <span>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

interface PipelineProps {
  result?: ClassifyResult;
  roundTripMs?: number;
  showUpload?: boolean; // only the phone knows the upload time
  busy?: boolean; // loop the animation while waiting
  replay?: boolean; // play the animation once (big screen, on each new photo)
  backend: Backend | null;
}

interface Stage {
  chip: string;
  role: string;
  ms?: number;
  hero?: boolean;
  offline?: boolean;
}

/** The board's chips as a flow diagram, with the time each one took. */
export function Pipeline({ result, roundTripMs, showUpload = false, busy = false, replay = false, backend }: PipelineProps) {
  const dev = backend === "cpu";
  const upload: Stage = {
    chip: "Phone",
    role: "Upload",
    ms: result && roundTripMs !== undefined ? roundTripMs - result.total_ms : undefined,
  };
  const stages: Stage[] = [
    ...(showUpload ? [upload] : []),
    { chip: dev ? "Laptop CPU" : "Linux CPU", role: "Preprocess", ms: result?.preprocess_ms },
    { chip: dev ? "Laptop CPU (dev)" : "Hexagon NPU", role: "DINOv2", ms: result?.model_ms, hero: true },
    {
      chip: "STM32 MCU",
      role: "LED matrix + RGB",
      ms: result?.action_ms ?? undefined,
      offline: result !== undefined && result.action_ms === null,
    },
  ];
  return (
    <div className={`flow ${busy ? "running" : replay ? "replay" : ""}`}>
      {stages.map((s, i) => (
        <Fragment key={s.role}>
          {i > 0 && <div className="wire" style={{ "--i": i } as CSSProperties} />}
          <div className={`node ${s.hero ? "hero" : ""} ${s.offline ? "offline" : ""}`} style={{ "--i": i } as CSSProperties}>
            <span className="chip">{s.chip}</span>
            <span className="role">{s.role}</span>
            <span className="ms">{s.offline ? "no MCU" : s.ms === undefined ? "–" : `${Math.round(s.ms)} ms`}</span>
          </div>
        </Fragment>
      ))}
    </div>
  );
}

export function Stats({ session }: { session: Session }) {
  return (
    <dl className="stats">
      <div><dt>Photos</dt><dd>{session.count}</dd></div>
      <div><dt>Avg model</dt><dd>{session.avg_model_ms.toFixed(0)} ms</dd></div>
      <div><dt>Avg total</dt><dd>{session.avg_total_ms.toFixed(0)} ms</dd></div>
    </dl>
  );
}

export function History({ shots }: { shots: Shot[] }) {
  if (shots.length === 0) return <p className="muted">No photos yet.</p>;
  return (
    <ul className="history">
      {shots.map((s) => (
        <li key={s.id}>
          <img src={s.image} alt="" />
          <span>{shortLabel(s.result.top[0][0])}</span>
        </li>
      ))}
    </ul>
  );
}

export function FullscreenButton() {
  const [full, setFull] = useState(Boolean(document.fullscreenElement));
  useEffect(() => {
    const onChange = () => setFull(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);
  if (full || !document.fullscreenEnabled) return null;
  return (
    <button className="badge" onClick={() => document.documentElement.requestFullscreen()}>
      Fullscreen
    </button>
  );
}
