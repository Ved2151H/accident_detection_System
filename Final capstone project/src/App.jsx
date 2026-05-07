import { useState, useEffect, useRef } from "react";
import "./App.css";

// ── Data ─────────────────────────────────────────────────────────────────────

const AREAS = [
  { name: "Main Square",   status: "Heavy",    color: "#ef4444", pct: 88 },
  { name: "Ring Road",     status: "Moderate", color: "#f59e0b", pct: 55 },
  { name: "Market Street", status: "Clear",    color: "#22c55e", pct: 22 },
  { name: "North Bridge",  status: "Heavy",    color: "#ef4444", pct: 78 },
  { name: "South Bypass",  status: "Clear",    color: "#22c55e", pct: 18 },
];

const FLOW_DATA = [
  { t: "08:00", v: 42  },
  { t: "08:10", v: 61  },
  { t: "08:20", v: 88  },
  { t: "08:30", v: 102 },
  { t: "08:40", v: 124 },
  { t: "now",   v: 118 },
];

const INTERSECTIONS = [
  { name: "Main Sq × Park Ave", vol: 241, sig: "red"   },
  { name: "Ring Rd × High St",  vol: 187, sig: "green" },
  { name: "Market × Broad",     vol: 95,  sig: "green" },
  { name: "North Bridge",       vol: 203, sig: "amber" },
  { name: "South Gate",         vol: 67,  sig: "green" },
  { name: "Central Station",    vol: 310, sig: "red"   },
  { name: "East Terminal",      vol: 134, sig: "green" },
  { name: "West Crossover",     vol: 88,  sig: "amber" },
];

const SIGNAL_CYCLE  = ["green", "yellow", "red"];
const SIGNAL_TIMERS = { green: 28, yellow: 8, red: 35 };
const SIGNAL_SUBS   = {
  green:  "Flow permitted · Auto cycle",
  yellow: "Caution · Clearing junction",
  red:    "Stop · Pedestrian crossing",
};
const SIG_COLORS = { green: "#22c55e", amber: "#f59e0b", red: "#ef4444" };
const MAX_FLOW   = Math.max(...FLOW_DATA.map((d) => d.v));

const CARS = [
  { color: "#3b5bdb", top: "30px",   speed: "4s",   delay: "0s"    },
  { color: "#c92a2a", bottom: "26px", speed: "3.2s", delay: "-1.2s" },
  { color: "#2f9e44", bottom: "26px", speed: "5s",   delay: "-2.5s" },
  { color: "#862e9c", top: "28px",   speed: "3.8s",  delay: "-3s"   },
];

const INCIDENTS = [
  { sev: "high", title: "Vehicle stall — Ring Road East",       meta: "Reported 08:42 · 3 units en route",  badge: "HIGH" },
  { sev: "med",  title: "Signal fault · Junction 4B",           meta: "Since 09:15 · Maintenance notified", badge: "MED"  },
  { sev: "low",  title: "Pedestrian density spike · Market St", meta: "Monitor mode · No action required",  badge: "LOW"  },
];

// ── Custom hook: live clock ───────────────────────────────────────────────────

function useClock() {
  const [time, setTime] = useState("");
  useEffect(() => {
    const tick = () => {
      const n = new Date();
      setTime(
        String(n.getHours()).padStart(2, "0") + ":" +
        String(n.getMinutes()).padStart(2, "0") + ":" +
        String(n.getSeconds()).padStart(2, "0")
      );
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

// ── Components ────────────────────────────────────────────────────────────────

function LogoMark() {
  return (
    <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="32" height="32" rx="6" fill="#0a0a0a" />
      <circle cx="16" cy="8"  r="4" fill="#ef4444" />
      <circle cx="16" cy="16" r="4" fill="#f59e0b" />
      <circle cx="16" cy="24" r="4" fill="#22c55e" />
      <rect x="12" y="8" width="8" height="16" rx="1" fill="none" stroke="#333" strokeWidth="1" />
    </svg>
  );
}

function Topbar({ time }) {
  return (
    <div className="topbar">
      <div className="logo">
        <div className="logo-mark"><LogoMark /></div>
        <div>
          <span className="logo-text">TrafficOS</span>
          <span className="logo-sub">Urban Management System · v2.4.1</span>
        </div>
      </div>
      <div className="topbar-right">
        <div className="status-label">
          <div className="status-dot" />
          All systems nominal
        </div>
        <div className="sys-time">{time}</div>
      </div>
    </div>
  );
}

function AreaStatus({ activeArea, onSelect }) {
  return (
    <div className="card">
      <div className="card-label">Area Status</div>
      <div className="areas-list">
        {AREAS.map((a, i) => (
          <div
            key={a.name}
            className={`area-row${activeArea === i ? " active" : ""}`}
            onClick={() => onSelect(i)}
          >
            <div
              className="area-dot"
              style={{ background: a.color, boxShadow: `0 0 6px ${a.color}50` }}
            />
            <div className="area-name">{a.name}</div>
            <div className="area-bar-wrap">
              <div className="area-bar" style={{ width: `${a.pct}%`, background: a.color }} />
            </div>
            <div className="area-status-text" style={{ color: a.color }}>
              {a.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SignalControl() {
  const [signal, setSignalState] = useState("green");
  const [timer, setTimer]        = useState(SIGNAL_TIMERS.green);
  const signalRef                = useRef("green");

  useEffect(() => {
    const id = setInterval(() => {
      setTimer((prev) => {
        if (prev <= 1) {
          const next =
            SIGNAL_CYCLE[(SIGNAL_CYCLE.indexOf(signalRef.current) + 1) % SIGNAL_CYCLE.length];
          signalRef.current = next;
          setSignalState(next);
          return SIGNAL_TIMERS[next];
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const handleSet = (s) => {
    signalRef.current = s;
    setSignalState(s);
    setTimer(SIGNAL_TIMERS[s]);
  };

  return (
    <div className="card">
      <div className="card-label">Signal Control · Main Square</div>
      <div className="signal-display">
        <div className="traffic-light">
          <div className={`bulb bulb-r${signal === "red"    ? " on" : ""}`} />
          <div className={`bulb bulb-y${signal === "yellow" ? " on" : ""}`} />
          <div className={`bulb bulb-g${signal === "green"  ? " on" : ""}`} />
        </div>
        <div className="signal-info">
          <div className={`signal-mode ${signal}`}>{signal.toUpperCase()}</div>
          <div className="signal-sub">{SIGNAL_SUBS[signal]}</div>
          <div className="signal-timer">{timer}</div>
          <div className="timer-label">seconds remaining</div>
        </div>
      </div>
      <div className="signal-btns">
        {SIGNAL_CYCLE.map((s) => (
          <button
            key={s}
            className={`sig-btn ${s}${signal === s ? " active" : ""}`}
            onClick={() => handleSet(s)}
          >
            ● {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>
    </div>
  );
}

function LiveStats() {
  const [stats, setStats] = useState({ veh: 124, spd: 32, flow: 87 });

  useEffect(() => {
    const id = setInterval(() => {
      setStats({
        veh:  124 + Math.round((Math.random() - 0.4) * 6),
        spd:  32  + Math.round((Math.random() - 0.5) * 4),
        flow: Math.min(99, Math.max(60, 87 + Math.round((Math.random() - 0.5) * 8))),
      });
    }, 2500);
    return () => clearInterval(id);
  }, []);

  const blocks = [
    { val: stats.veh,  color: "amber", label: "Vehicles active",   delta: "▲ +3 / min",     dir: "up"   },
    { val: stats.spd,  color: "green", label: "Avg speed km/h",    delta: "▼ –2 from peak", dir: "down" },
    { val: stats.flow, color: "cyan",  label: "Flow efficiency %", delta: "▲ +4 pts",        dir: "down" },
    { val: 2,          color: "red",   label: "Active incidents",  delta: "▲ 1 new",         dir: "up"   },
  ];

  return (
    <div className="card">
      <div className="card-label">Live Metrics</div>
      <div className="stats-grid">
        {blocks.map((b) => (
          <div key={b.label} className="stat-block">
            <div className={`stat-val ${b.color}`}>{b.val}</div>
            <div className="stat-lbl">{b.label}</div>
            <div className={`stat-delta ${b.dir}`}>{b.delta}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CameraFeed() {
  return (
    <div className="card span-2">
      <div className="card-label">Camera Feed · CAM-01 Main Square</div>
      <div className="cam-feed">
        <div className="cam-grid-overlay" />
        <div className="cam-road" />
        <div className="cam-lines" />
        <div className="cam-cross" />
        {CARS.map((c, i) => (
          <div
            key={i}
            className="cam-car"
            style={{
              background: c.color,
              top: c.top,
              bottom: c.bottom,
              "--speed": c.speed,
              animationDelay: c.delay,
            }}
          />
        ))}
        <div className="cam-label">CAM-01 · MAIN SQUARE · LIVE</div>
        <div className="cam-rec">
          <div className="rec-dot" />
          REC
        </div>
      </div>

      <div className="flow-wrap">
        <div className="flow-label">Vehicle throughput — last 6 intervals (vehicles/min)</div>
        {FLOW_DATA.map((d) => {
          const pct   = Math.round((d.v / MAX_FLOW) * 100);
          const color = pct > 80 ? "#ef4444" : pct > 55 ? "#f59e0b" : "#22c55e";
          return (
            <div key={d.t} className="flow-bar-row">
              <div className="flow-time">{d.t}</div>
              <div className="flow-bar-bg">
                <div className="flow-bar-fill" style={{ width: `${pct}%`, background: color }} />
              </div>
              <div className="flow-count">{d.v}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function IncidentIcon({ severity }) {
  if (severity === "high")
    return (
      <svg className="inc-icon" viewBox="0 0 16 16" fill="none">
        <path d="M8 2L14 13H2L8 2Z" stroke="#ef4444" strokeWidth="1.2" />
        <path d="M8 7v3M8 11.5v.5" stroke="#ef4444" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    );
  if (severity === "med")
    return (
      <svg className="inc-icon" viewBox="0 0 16 16" fill="none">
        <circle cx="8" cy="8" r="5.5" stroke="#f59e0b" strokeWidth="1.2" />
        <path d="M8 5v3.5M8 10v.5" stroke="#f59e0b" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    );
  return (
    <svg className="inc-icon" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="5.5" stroke="#06b6d4" strokeWidth="1.2" />
      <path d="M8 7.5h.5M8 9v2" stroke="#06b6d4" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}

function Incidents() {
  return (
    <div className="card">
      <div className="card-label">Incidents</div>
      <div className="incident-list">
        {INCIDENTS.map((inc) => (
          <div key={inc.title} className={`incident ${inc.sev}`}>
            <IncidentIcon severity={inc.sev} />
            <div className="inc-body">
              <div className="inc-title">{inc.title}</div>
              <div className="inc-meta">{inc.meta}</div>
            </div>
            <div className={`inc-badge ${inc.sev}`}>{inc.badge}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function IntersectionOverview() {
  return (
    <div className="card span-3">
      <div className="card-label">Intersection Overview</div>
      <div className="intersection-grid">
        {INTERSECTIONS.map((item) => {
          const c = SIG_COLORS[item.sig];
          return (
            <div key={item.name} className="int-cell">
              <div>
                <div className="int-name">{item.name}</div>
                <div className="int-vol">{item.vol} veh/hr</div>
              </div>
              <div
                className="int-sig"
                style={{ background: c, boxShadow: `0 0 7px ${c}60` }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const time                      = useClock();
  const [activeArea, setActiveArea] = useState(0);

  return (
    <div className="dashboard">
      <Topbar time={time} />
      <div className="main-grid">
        <AreaStatus activeArea={activeArea} onSelect={setActiveArea} />
        <SignalControl />
        <LiveStats />
        <CameraFeed />
        <Incidents />
        <IntersectionOverview />
      </div>
    </div>
  );
}