import { useState } from "react";
import { api } from "../api";
import MoleculeViewer3D from "../components/MoleculeViewer3D";
import { ELEMENTS as ALL_ELEMENTS, ELEMENT_MAP } from "../data/elements";
import { Plus, Trash2 } from "lucide-react";

const ELEMENTS = ALL_ELEMENTS.map((e) => e.sym);

const MAX_VAL = Object.fromEntries(ALL_ELEMENTS.map((e) => [e.sym, e.maxVal]));

function parseFormula(formula) {
  const str = formula.trim();
  let charge = 0;
  let core = str;

  const chargeMatch = core.match(/([+-])(\d*)$/);
  if (chargeMatch) {
    const sign = chargeMatch[1] === "+" ? 1 : -1;
    const mag = chargeMatch[2] ? parseInt(chargeMatch[2]) : 1;
    charge = sign * mag;
    core = core.slice(0, -chargeMatch[0].length);
  }

  const atoms = [];
  const re = /([A-Z][a-z]?)(\d*)/g;
  let m;
  while ((m = re.exec(core)) !== null) {
    if (!m[1]) continue;
    const elem = m[1];
    const count = m[2] ? parseInt(m[2]) : 1;
    for (let i = 0; i < count; i++) {
      atoms.push({ element: elem, bonds: 0, formal_charge: 0 });
    }
  }
  if (atoms.length === 0) return null;

  if (charge !== 0) {
    const heaviest = atoms.find((a) => a.element !== "H") || atoms[0];
    heaviest.formal_charge = charge;
  }

  const remaining = atoms.map(
    (a) => MAX_VAL[a.element] || 4,
  );
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      if (remaining[i] <= 0 || remaining[j] <= 0) continue;
      if (atoms[i].element === "H" && atoms[j].element === "H") continue;
      const order = Math.min(remaining[i], remaining[j]);
      atoms[i].bonds += order;
      atoms[j].bonds += order;
      remaining[i] -= order;
      remaining[j] -= order;
    }
  }

  const name =
    formula.trim() || atoms.map((a) => a.element).join("");
  return { name, atoms };
}

function parseEquation(equation) {
  const arrow = equation.includes("->")
    ? "->"
    : equation.includes("→")
      ? "→"
      : equation.includes("=>")
        ? "=>"
        : null;
  if (!arrow) return null;

  const [lhs, rhs] = equation.split(arrow).map((s) => s.trim());
  if (!lhs || !rhs) return null;

  const reactants = lhs
    .split("+")
    .map((s) => parseFormula(s.trim()))
    .filter(Boolean);
  const products = rhs
    .split("+")
    .map((s) => parseFormula(s.trim()))
    .filter(Boolean);

  if (reactants.length === 0 || products.length === 0) return null;
  return { reactants, products };
}
export default function ConstraintChecker({ presets }) {
  const [mode, setMode] = useState("equation");
  const [reactionIdx, setReactionIdx] = useState(0);
  const [moleculeIdx, setMoleculeIdx] = useState(0);
  const [useCustom, setUseCustom] = useState(false);
  const [customName, setCustomName] = useState("custom");
  const [customAtoms, setCustomAtoms] = useState([
    { element: "C", bonds: 4, formal_charge: 0 },
  ]);
  const [equationText, setEquationText] = useState("CH3Br + OH- -> CH3OH + Br-");
  const [parsedEq, setParsedEq] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const checkEquation = async () => {
    const parsed = parseEquation(equationText);
    if (!parsed) {
      setError("Could not parse equation. Use format: A + B -> C + D");
      return;
    }
    setParsedEq(parsed);
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.checkReaction({
        reactants: parsed.reactants,
        products: parsed.products,
      });
      setResult({ ...data, type: "equation", parsed });
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const checkReaction = async () => {
    if (!presets) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const p = presets.reactions[reactionIdx];
      const data = await api.checkReaction({
        reactants: p.reactants, products: p.products,
      });
      setResult({ ...data, type: "reaction", preset: p });
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const checkMolecule = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const mol = useCustom
        ? { name: customName, atoms: customAtoms.map((a) => ({ ...a, implicit_h: 0 })) }
        : presets?.molecules[moleculeIdx];
      if (!mol) return;
      const data = await api.checkIntermediate({ molecule: mol });
      setResult({ ...data, type: "molecule", molecule: mol });
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleCheck = () =>
    mode === "equation"
      ? checkEquation()
      : mode === "reaction"
        ? checkReaction()
        : checkMolecule();

  const addAtom = () =>
    setCustomAtoms((p) => [...p, { element: "C", bonds: 4, formal_charge: 0 }]);
  const removeAtom = (i) =>
    setCustomAtoms((p) => p.filter((_, idx) => idx !== i));
  const updateAtom = (i, field, val) =>
    setCustomAtoms((p) =>
      p.map((a, idx) => (idx === i ? { ...a, [field]: val } : a)),
    );

  const checkedMolAtoms = result?.type === "molecule" ? result.molecule?.atoms : null;
  const allReactionAtoms = result?.type === "reaction" && result.preset
    ? [...result.preset.reactants.flatMap((m) => m.atoms), ...result.preset.products.flatMap((m) => m.atoms)]
    : null;

  const inputStyle = { background: "var(--bg-input)", border: "1px solid var(--border)" };

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      <h1 className="text-[28px] font-semibold text-white mb-2">Constraint Checker</h1>
      <p className="text-[15px] mb-8" style={{ color: "var(--text-secondary)" }}>
        Verify reactions or molecules against conservation axioms.
      </p>

      <div className="flex gap-4 mb-8">
        {[
          { id: "equation", label: "Equation" },
          { id: "reaction", label: "Preset" },
          { id: "molecule", label: "Molecule" },
        ].map(({ id, label }) => (
          <button
            key={id}
            onClick={() => { setMode(id); setResult(null); }}
            className={`px-3 py-1.5 rounded-md text-[13px] transition-colors ${
              mode === id ? "text-white bg-white/[0.08]" : "text-neutral-500 hover:text-neutral-300"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-7">
          {mode === "equation" ? (
            <>
              <label className="text-[13px] block mb-2" style={{ color: "var(--text-muted)" }}>Reaction equation</label>
              <input
                value={equationText}
                onChange={(e) => setEquationText(e.target.value)}
                placeholder="CH3Br + OH- -> CH3OH + Br-"
                className="w-full rounded-md px-4 py-3 text-[15px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500 mb-2"
                style={inputStyle}
                onKeyDown={(e) => { if (e.key === "Enter") handleCheck(); }}
              />
              <p className="text-[12px] mb-5" style={{ color: "var(--text-muted)" }}>
                Separate reactants/products with +. Use -&gt; or =&gt; as arrow. Charges: OH-, NH4+
              </p>
              {parsedEq && (
                <div className="grid grid-cols-2 gap-6 mb-4">
                  <div>
                    <span className="text-[13px] text-white block mb-2">Reactants</span>
                    {parsedEq.reactants.map((m, i) => <MoleculeCard key={i} mol={m} />)}
                  </div>
                  <div>
                    <span className="text-[13px] text-white block mb-2">Products</span>
                    {parsedEq.products.map((m, i) => <MoleculeCard key={i} mol={m} />)}
                  </div>
                </div>
              )}
            </>
          ) : mode === "reaction" ? (
            <>
              <label className="text-[13px] block mb-2" style={{ color: "var(--text-muted)" }}>Select reaction</label>
              <select
                value={reactionIdx}
                onChange={(e) => setReactionIdx(Number(e.target.value))}
                className="w-full rounded-md px-3 py-2.5 text-[14px] text-white focus:outline-none focus:ring-1 focus:ring-neutral-500 mb-5"
                style={inputStyle}
              >
                {presets?.reactions.map((r, i) => <option key={i} value={i}>{r.name}</option>)}
              </select>
              {presets && (
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <span className="text-[13px] text-white block mb-2">Reactants</span>
                    {presets.reactions[reactionIdx].reactants.map((m, i) => <MoleculeCard key={i} mol={m} />)}
                  </div>
                  <div>
                    <span className="text-[13px] text-white block mb-2">Products</span>
                    {presets.reactions[reactionIdx].products.map((m, i) => <MoleculeCard key={i} mol={m} />)}
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="flex items-center gap-4 mb-4 text-[14px]">
                <label className="flex items-center gap-2 cursor-pointer" style={{ color: "var(--text-secondary)" }}>
                  <input type="radio" checked={!useCustom} onChange={() => setUseCustom(false)} className="accent-green-500" /> Preset
                </label>
                <label className="flex items-center gap-2 cursor-pointer" style={{ color: "var(--text-secondary)" }}>
                  <input type="radio" checked={useCustom} onChange={() => setUseCustom(true)} className="accent-green-500" /> Custom
                </label>
              </div>
              {!useCustom ? (
                <>
                  <select value={moleculeIdx} onChange={(e) => setMoleculeIdx(Number(e.target.value))}
                    className="w-full rounded-md px-3 py-2.5 text-[14px] text-white focus:outline-none mb-4" style={inputStyle}>
                    {presets?.molecules.map((m, i) => <option key={i} value={i}>{m.name}</option>)}
                  </select>
                  {presets && <MoleculeCard mol={presets.molecules[moleculeIdx]} />}
                </>
              ) : (
                <>
                  <input value={customName} onChange={(e) => setCustomName(e.target.value)} placeholder="Molecule name"
                    className="w-full rounded-md px-3 py-2 text-[14px] text-white focus:outline-none mb-3" style={inputStyle} />
                  <div className="space-y-2 mb-3">
                    {customAtoms.map((a, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <select value={a.element} onChange={(e) => updateAtom(i, "element", e.target.value)}
                          className="rounded-md px-2 py-1.5 text-[13px] text-white focus:outline-none" style={inputStyle}>
                          {ELEMENTS.map((el) => <option key={el} value={el}>{el}</option>)}
                        </select>
                        <span className="text-[12px]" style={{ color: "var(--text-muted)" }}>bonds</span>
                        <input type="number" value={a.bonds} onChange={(e) => updateAtom(i, "bonds", Number(e.target.value))}
                          min={0} max={8} className="w-14 rounded-md px-2 py-1.5 text-[13px] font-mono text-white focus:outline-none" style={inputStyle} />
                        <span className="text-[12px]" style={{ color: "var(--text-muted)" }}>charge</span>
                        <input type="number" value={a.formal_charge} onChange={(e) => updateAtom(i, "formal_charge", Number(e.target.value))}
                          min={-3} max={3} className="w-14 rounded-md px-2 py-1.5 text-[13px] font-mono text-white focus:outline-none" style={inputStyle} />
                        <button onClick={() => removeAtom(i)} disabled={customAtoms.length <= 1}
                          className="text-neutral-600 hover:text-red-400 disabled:opacity-20 transition-colors">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                  <button onClick={addAtom} className="text-[13px] hover:text-green-400 transition-colors" style={{ color: "var(--text-muted)" }}>
                    + Add atom
                  </button>
                </>
              )}
            </>
          )}

          <button onClick={handleCheck} disabled={loading || !presets}
            className="mt-6 px-5 py-2 bg-white text-black text-[14px] font-medium rounded-md hover:bg-neutral-200 disabled:opacity-30 transition-colors">
            {loading ? "Checking..." : "Check"}
          </button>
        </div>

        <div className="col-span-5">
          {error && <p className="text-red-400 text-[14px] mb-4">{error}</p>}

          {result ? (
            <div className="space-y-5">
              <div className={`rounded-lg p-5 ${result.result.sat ? "bg-green-500/5" : "bg-red-500/5"}`}>
                <span className={`text-[18px] font-semibold ${result.result.sat ? "text-green-400" : "text-red-400"}`}>
                  {result.result.sat ? "All constraints satisfied" : "Violations found"}
                </span>
                <span className="text-[13px] ml-2 font-mono" style={{ color: "var(--text-muted)" }}>{result.elapsed_ms}ms</span>
                {result.result.violations.length > 0 && (
                  <div className="mt-3 space-y-1">
                    {result.result.violations.map((v, i) => (
                      <p key={i} className="text-[13px] text-red-300">{v}</p>
                    ))}
                  </div>
                )}
              </div>

              {checkedMolAtoms && (
                <div className="rounded-lg overflow-hidden" style={{ background: "var(--bg-raised)" }}>
                  <MoleculeViewer3D atoms={checkedMolAtoms.map(a => ({
                    ...a, formal_charge: a.formal_charge || 0,
                    implicit_h: 0, effective_valency: MAX_VAL[a.element] || 4,
                    total_bonds: a.bonds || 0,
                  }))} height={220} />
                </div>
              )}

              {result.type === "molecule" && (
                <div className="flex gap-8 text-[14px]">
                  <div>
                    <span className="block text-[12px]" style={{ color: "var(--text-muted)" }}>Mass</span>
                    <span className="font-mono text-white">{result.total_mass?.toFixed(3)} u</span>
                  </div>
                  <div>
                    <span className="block text-[12px]" style={{ color: "var(--text-muted)" }}>Charge</span>
                    <span className="font-mono text-white">{result.total_charge > 0 ? "+" : ""}{result.total_charge}</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-[14px] py-10" style={{ color: "var(--text-muted)" }}>
              Run a check to see results.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function MoleculeCard({ mol }) {
  return (
    <div className="py-2 mb-1">
      <span className="text-[14px] text-white">{mol.name}</span>
      <div className="flex flex-wrap gap-1.5 mt-1">
        {mol.atoms.map((a, i) => (
          <span key={i} className="text-[12px] font-mono px-1.5 py-0.5 rounded" style={{ background: "var(--bg-input)", color: "var(--text-secondary)" }}>
            {a.element} b{a.bonds}
            {(a.formal_charge || 0) !== 0 && <span className="text-yellow-400"> {a.formal_charge > 0 ? "+" : ""}{a.formal_charge}</span>}
          </span>
        ))}
      </div>
    </div>
  );
}
