import { useState, useEffect } from "react";
import { api } from "../api";
import ReactionPicker from "../components/ReactionPicker";

export default function BenchmarkPage({ presets }) {
  const [n, setN] = useState(20);
  const [reaction, setReaction] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (presets && !reaction) {
      setReaction({ reactants: presets.reactions[0].reactants, label: presets.reactions[0].name });
    }
  }, [presets, reaction]);

  const run = async () => {
    const reactants = reaction?.reactants || presets?.reactions[0]?.reactants;
    if (!reactants) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.runBenchmark({
        reactants, n, hidden_dim: 64, T: 10,
      });
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const valDelta = result
    ? (result.summary.supervised_valency_pct - result.summary.unsupervised_valency_pct).toFixed(1)
    : 0;
  const fullDelta = result
    ? (result.summary.supervised_full_pct - result.summary.unsupervised_full_pct).toFixed(1)
    : 0;

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      <h1 className="text-[28px] font-semibold text-white mb-2">Benchmark</h1>
      <p className="text-[15px] mb-3" style={{ color: "var(--text-secondary)" }}>
        Does the constraint supervisor actually improve molecule quality? Run side-by-side trials to find out.
      </p>

      {/* ── How it works ── */}
      <div className="rounded-lg p-5 mb-8" style={{ background: "var(--bg-raised)" }}>
        <span className="text-[14px] text-white font-medium block mb-2">How this benchmark works</span>
        <div className="grid grid-cols-3 gap-6 text-[13px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          <div>
            <span className="text-green-400 font-medium">With Supervisor</span> — the AI model generates a molecule,
            and at every step the constraint engine checks the result and fixes any rule violations (bad bond counts,
            charge errors) before moving on.
          </div>
          <div>
            <span className="text-red-400 font-medium">Without Supervisor</span> — the same AI model generates a
            molecule with no checking or corrections at all. Whatever the model outputs is the final answer.
          </div>
          <div>
            <span className="text-white font-medium">Why compare?</span> — This shows how much value the constraint
            system adds. A perfect AI model wouldn't need corrections, but in practice, the supervisor catches
            mistakes and dramatically improves validity rates.
          </div>
        </div>
      </div>

      {presets && <ReactionPicker presets={presets} value={reaction} onChange={setReaction} />}

      <div className="flex items-end gap-4 mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
        <div>
          <label className="text-[13px] block mb-1.5" style={{ color: "var(--text-muted)" }}>
            Number of trials
          </label>
          <input
            type="number" value={n}
            onChange={(e) => setN(Math.min(100, Math.max(1, Number(e.target.value))))}
            min={1} max={100}
            className="w-24 rounded-md px-3 py-2 text-[14px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
            style={{ background: "var(--bg-input)", border: "1px solid var(--border)" }}
          />
        </div>
        <button
          onClick={run} disabled={loading || !presets}
          className="px-5 py-2 bg-white text-black text-[14px] font-medium rounded-md hover:bg-neutral-200 disabled:opacity-30 transition-colors"
        >
          {loading ? `Running ${n} trials...` : "Run benchmark"}
        </button>
      </div>

      {error && <p className="text-red-400 text-[14px] mb-6">{error}</p>}

      {result && (
        <>
          {/* ── Headline number ── */}
          <div className="mb-10">
            <span className="text-[36px] font-semibold text-green-400 font-mono">
              +{valDelta}%
            </span>
            <span className="text-[15px] ml-3" style={{ color: "var(--text-secondary)" }}>
              more valid molecules when the supervisor is ON ({result.n} trials)
            </span>
          </div>

          {/* ── Side-by-side bar charts ── */}
          <div className="grid grid-cols-2 gap-12 mb-10">
            <BarPair
              label="Atoms have correct bond counts"
              sublabel="(valency validity)"
              supervised={result.summary.supervised_valency_pct}
              unsupervised={result.summary.unsupervised_valency_pct}
            />
            <BarPair
              label="Mass + charge fully conserved"
              sublabel="(full conservation)"
              supervised={result.summary.supervised_full_pct}
              unsupervised={result.summary.unsupervised_full_pct}
            />
          </div>

          {/* ── Per-trial scatter ── */}
          {result.runs && (
            <div className="mb-10">
              <span className="text-[15px] text-white block mb-1">Per-trial results</span>
              <span className="text-[13px] block mb-4" style={{ color: "var(--text-muted)" }}>
                Each dot is one trial.{" "}
                <span className="text-green-400">Green</span> = valid,{" "}
                <span className="text-red-400">red</span> = invalid.
              </span>
              <RunScatter runs={result.runs} />
            </div>
          )}

          {/* ── Comparison table ── */}
          <span className="text-[15px] text-white block mb-3">Summary table</span>
          <table className="w-full text-[14px] mb-10">
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                <th className="text-left py-3 font-normal">What was checked</th>
                <th className="text-right py-3 font-normal">
                  <span className="text-green-400">With</span> supervisor
                </th>
                <th className="text-right py-3 font-normal">
                  <span className="text-red-400">Without</span> supervisor
                </th>
                <th className="text-right py-3 font-normal">Improvement</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                <td className="py-3">
                  <span className="text-white">Correct bond counts</span>
                  <span className="text-[12px] ml-2" style={{ color: "var(--text-muted)" }}>
                    (every atom's bonds ≤ its max valency)
                  </span>
                </td>
                <td className="py-3 text-right font-mono text-green-400">{result.summary.supervised_valency_pct}%</td>
                <td className="py-3 text-right font-mono text-red-400">{result.summary.unsupervised_valency_pct}%</td>
                <td className="py-3 text-right font-mono text-white">+{valDelta}%</td>
              </tr>
              <tr>
                <td className="py-3">
                  <span className="text-white">Full conservation</span>
                  <span className="text-[12px] ml-2" style={{ color: "var(--text-muted)" }}>
                    (mass + charge match between reactants and products)
                  </span>
                </td>
                <td className="py-3 text-right font-mono text-green-400">{result.summary.supervised_full_pct}%</td>
                <td className="py-3 text-right font-mono text-red-400">{result.summary.unsupervised_full_pct}%</td>
                <td className="py-3 text-right font-mono text-white">+{fullDelta}%</td>
              </tr>
            </tbody>
          </table>

          {/* ── Footer note ── */}
          <div className="rounded-lg p-4" style={{ background: "rgba(34,197,94,0.04)" }}>
            <p className="text-[13px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
              💡 <span className="text-white">Tip:</span> Train the GNN first on the{" "}
              <span className="text-white">Training</span> page — the trained weights are loaded here
              automatically when hidden dim matches. A trained model + supervisor together gives the best results.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function BarPair({ label, sublabel, supervised, unsupervised }) {
  return (
    <div>
      <span className="text-[14px] text-white block mb-1">{label}</span>
      {sublabel && (
        <span className="text-[12px] block mb-3" style={{ color: "var(--text-muted)" }}>{sublabel}</span>
      )}
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-[13px] mb-1.5">
            <span className="text-green-400">With supervisor</span>
            <span className="font-mono text-white">{supervised}%</span>
          </div>
          <div className="h-2 rounded-full" style={{ background: "var(--border)" }}>
            <div className="h-full bg-green-500 rounded-full transition-all duration-700" style={{ width: `${supervised}%` }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-[13px] mb-1.5">
            <span className="text-red-400">Without supervisor</span>
            <span className="font-mono text-white">{unsupervised}%</span>
          </div>
          <div className="h-2 rounded-full" style={{ background: "var(--border)" }}>
            <div className="h-full bg-red-500 rounded-full transition-all duration-700" style={{ width: `${unsupervised}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

function RunScatter({ runs }) {
  const n = runs.length;
  const pad = { top: 16, right: 8, bottom: 24, left: 130 };
  const dotR = Math.min(5, Math.max(3, 180 / n));
  const w = Math.max(400, n * (dotR * 2 + 2) + pad.left + pad.right);
  const rowH = 32;
  const h = pad.top + rowH * 4 + pad.bottom;

  const rows = [
    { key: "supervised_valency",      label: "With supervisor — bonds" },
    { key: "supervised_conservation", label: "With supervisor — conservation" },
    { key: "unsupervised_valency",      label: "Without supervisor — bonds" },
    { key: "unsupervised_conservation", label: "Without supervisor — conservation" },
  ];

  return (
    <div className="overflow-x-auto">
      <svg width={w} height={h}>
        {rows.map((row, ri) => {
          const y = pad.top + ri * rowH + rowH / 2;
          return (
            <g key={row.key}>
              <text x={pad.left - 8} y={y + 1} textAnchor="end" dominantBaseline="middle" fill="#737373" fontSize={11} fontFamily="Inter, sans-serif">
                {row.label}
              </text>
              <line x1={pad.left} y1={y} x2={w - pad.right} y2={y} stroke="#262626" strokeWidth={1} />
              {runs.map((r, i) => {
                const x = pad.left + (i / Math.max(1, n - 1)) * (w - pad.left - pad.right);
                const valid = r[row.key];
                return (
                  <circle key={i} cx={x} cy={y} r={dotR}
                    fill={valid ? "#22c55e" : "#ef4444"} fillOpacity={0.75}>
                    <title>Trial {i + 1} (seed {r.seed}): {valid ? "✓ valid" : "✗ invalid"}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
