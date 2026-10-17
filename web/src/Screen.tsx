import { useEffect, useState } from "react";
import { getSession, type Info, type Session } from "./api";
import { History, Pipeline, Snapshot, Stats, Verdict } from "./components";

const POLL_MS = 1000;

/** Big-screen view (/screen): shows every photo anyone takes, live. */
export default function Screen({ info }: { info: Info | null }) {
  const [session, setSession] = useState<Session | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () => getSession().then((s) => alive && setSession(s)).catch(() => {});
    tick();
    const timer = setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const latest = session?.shots[0];
  return (
    <main className="screen">
      {/* keyed on the shot id so each new photo pops in and replays the pipeline */}
      <section className="card latest pop" key={latest?.id ?? "empty"}>
        {latest ? (
          <>
            <Snapshot image={latest.image} />
            <Verdict result={latest.result} />
          </>
        ) : (
          <p className="muted">Waiting for the first photo...</p>
        )}
      </section>
      <div className="side">
        {info && (
          <section className="card join">
            <h2>Play</h2>
            <p>Open <strong>{info.url}</strong> on your phone and snap anything.</p>
          </section>
        )}
        <section className="card pipeline-card">
          <h2>Pipeline</h2>
          <Pipeline key={latest?.id} result={latest?.result} replay={Boolean(latest)} backend={info?.backend ?? null} />
        </section>
        {session && (
          <section className="card">
            <h2>Session</h2>
            <Stats session={session} />
          </section>
        )}
      </div>
      <section className="card wide">
        <h2>Recent</h2>
        <History shots={session?.shots ?? []} />
      </section>
    </main>
  );
}
