import { useEffect, useState } from "react";
import { getInfo, type Info } from "./api";
import { FullscreenButton } from "./components";
import Phone from "./Phone";
import Screen from "./Screen";

const HEALTH_POLL_MS = 3000;

export default function App() {
  const [info, setInfo] = useState<Info | null>(null);
  const [online, setOnline] = useState(true);
  const isScreen = location.pathname === "/screen";

  // Keep checking the server so we notice when the board drops off and comes back.
  useEffect(() => {
    let alive = true;
    const check = () =>
      getInfo()
        .then((next) => {
          if (!alive) return;
          setInfo(next);
          setOnline(true);
        })
        .catch(() => alive && setOnline(false));
    check();
    const timer = setInterval(check, HEALTH_POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className={`app ${isScreen ? "big" : ""}`}>
      <header>
        <div>
          <h1>Sense the World</h1>
          <p className="tagline">See → think → act, all on the edge.</p>
        </div>
        <div className="badges">
          <span className="badge">Arduino VENTUNO Q</span>
          {info && (
            <span className={`badge ${info.backend}`}>{info.backend === "npu" ? "Hexagon NPU" : "Laptop CPU (dev)"}</span>
          )}
          {isScreen && <FullscreenButton />}
        </div>
      </header>
      {!online && <p className="banner">Reconnecting to the board...</p>}
      {isScreen ? <Screen info={info} /> : <Phone info={info} />}
    </div>
  );
}
