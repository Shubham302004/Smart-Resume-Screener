import React, { useState, useEffect } from "react";
import { Upload, FileText, Loader2, Download, Trophy, Star, AlertCircle, CheckCircle, XCircle, ChevronDown, ChevronUp, BarChart2, Sparkles, Target, Brain, X } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const GRADE_CONFIG = {
  A: { border: "rgba(16,185,129,0.4)",  bg: "rgba(16,185,129,0.07)",  badge: "#10b981", text: "#6ee7b7",  label: "Excellent" },
  B: { border: "rgba(56,189,248,0.4)",  bg: "rgba(56,189,248,0.07)",  badge: "#38bdf8", text: "#7dd3fc",  label: "Good" },
  C: { border: "rgba(245,158,11,0.4)",  bg: "rgba(245,158,11,0.07)",  badge: "#f59e0b", text: "#fcd34d",  label: "Partial" },
  D: { border: "rgba(249,115,22,0.4)",  bg: "rgba(249,115,22,0.07)",  badge: "#f97316", text: "#fdba74",  label: "Weak" },
  F: { border: "rgba(239,68,68,0.4)",   bg: "rgba(239,68,68,0.07)",   badge: "#ef4444", text: "#fca5a5",  label: "Poor" },
};

const CHART_COLORS = ["#10b981", "#38bdf8", "#f59e0b", "#f97316", "#ef4444"];

function useCountUp(target, duration = 900, active = false) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!active) { setVal(0); return; }
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      setVal(Math.floor(p * target));
      if (p < 1) requestAnimationFrame(step);
      else setVal(target);
    };
    requestAnimationFrame(step);
  }, [target, active]);
  return val;
}

function ScoreBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
        <span style={{ color: "#64748b" }}>{label}</span>
        <span style={{ color: "#94a3b8", fontWeight: 600 }}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 5, borderRadius: 99, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
        <div style={{ height: "100%", borderRadius: 99, background: color, width: `${value * 100}%`, transition: "width 1s ease" }} />
      </div>
    </div>
  );
}

function SkillTag({ label, type }) {
  const cfg = {
    matched: { bg: "rgba(16,185,129,0.12)", border: "rgba(16,185,129,0.25)", color: "#6ee7b7", icon: <CheckCircle size={9} /> },
    missing: { bg: "rgba(239,68,68,0.08)",  border: "rgba(239,68,68,0.2)",   color: "#fca5a5", icon: <XCircle size={9} /> },
    bonus:   { bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.25)", color: "#fcd34d", icon: <Star size={9} /> },
  }[type];
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, fontWeight: 500, padding: "3px 9px", borderRadius: 99, background: cfg.bg, border: `1px solid ${cfg.border}`, color: cfg.color }}>
      {cfg.icon} {label}
    </span>
  );
}

function CandidateCard({ res, rank, expanded, onToggle }) {
  const g = GRADE_CONFIG[res.grade] || GRADE_CONFIG["F"];
  const animScore = useCountUp(Math.round(res.score * 100), 900, expanded);
  const displayScore = expanded ? animScore : Math.round(res.score * 100);
  const [showAllMissing, setShowAllMissing] = useState(false);

  return (
    <div style={{ borderRadius: 18, border: `1px solid ${g.border}`, background: g.bg, overflow: "hidden", transition: "box-shadow 0.3s", boxShadow: expanded ? `0 8px 32px ${g.border}` : "none" }}>

      {/* Header */}
      <div onClick={onToggle} style={{ display: "flex", alignItems: "center", gap: 14, padding: "16px 22px", cursor: "pointer", userSelect: "none" }}>

        <div style={{ flexShrink: 0, width: 28, height: 28, borderRadius: "50%", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, color: "#64748b" }}>
          {rank}
        </div>

        {res.is_best && (
          <div style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 99, background: "linear-gradient(135deg, #f59e0b, #ef4444)", color: "#fff", fontSize: 11, fontWeight: 800, letterSpacing: "0.04em" }}>
            <Trophy size={10} /> BEST FIT
          </div>
        )}

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
            <FileText size={13} style={{ color: "#94a3b8", flexShrink: 0 }} />
            <span style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{res.name}</span>
          </div>
          <span style={{ fontSize: 12, color: g.text }}>{res.verdict}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 26, fontWeight: 900, color: "#f1f5f9", lineHeight: 1, fontFamily: "'Plus Jakarta Sans', sans-serif" }}>
              {displayScore}<span style={{ fontSize: 14, fontWeight: 400, color: "#94a3b8" }}>%</span>
            </div>
            <div style={{ fontSize: 11, color: "#94a3b8" }}>overall</div>
          </div>
          <div style={{ width: 38, height: 38, borderRadius: 10, background: g.badge, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontWeight: 900, fontSize: 16, boxShadow: `0 4px 16px ${g.border}`, fontFamily: "'Plus Jakarta Sans', sans-serif" }}>
            {res.grade}
          </div>
          <div style={{ color: "#94a3b8" }}>
            {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </div>
        </div>
      </div>

      {/* Expanded */}
      {expanded && (
        <div style={{ padding: "4px 22px 22px", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginTop: 18 }}>

            {/* Scores */}
            <div>
              <p style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 16 }}>Score Breakdown</p>
              <ScoreBar label="Skills Coverage" value={res.skills_score} color="#10b981" />
              <ScoreBar label="Experience Match" value={res.experience_score} color="#38bdf8" />
              <ScoreBar label="Education Match" value={res.education_score} color="#a78bfa" />

              <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
                <div style={{ flex: 1, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: "10px 12px", textAlign: "center" }}>
                  <div style={{ fontSize: 18, fontWeight: 900, color: g.text, fontFamily: "'Plus Jakarta Sans', sans-serif" }}>{res.skill_match_pct}%</div>
                  <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>Keyword Coverage</div>
                </div>
                {res.years_experience > 0 && (
                  <div style={{ flex: 1, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: "10px 12px", textAlign: "center" }}>
                    <div style={{ fontSize: 18, fontWeight: 900, color: "#38bdf8", fontFamily: "'Plus Jakarta Sans', sans-serif" }}>~{res.years_experience}</div>
                    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>Yrs Experience</div>
                  </div>
                )}
              </div>
            </div>

            {/* Skills */}
            <div>
              <p style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 16 }}>Skills Analysis</p>

              {res.matched_skills.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <p style={{ fontSize: 11, color: "#10b981", marginBottom: 6, fontWeight: 600 }}>✓ Matched ({res.matched_skills.length})</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {res.matched_skills.map((s, i) => <SkillTag key={i} label={s} type="matched" />)}
                  </div>
                </div>
              )}

              {res.missing_skills.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <p style={{ fontSize: 11, color: "#ef4444", marginBottom: 6, fontWeight: 600 }}>✕ Missing ({res.missing_skills.length})</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5, alignItems: "center" }}>
                    {(showAllMissing ? res.missing_skills : res.missing_skills.slice(0, 8)).map((s, i) => (
                      <SkillTag key={i} label={s} type="missing" />
                    ))}
                    {res.missing_skills.length > 8 && (
                      <button
                        onClick={e => { e.stopPropagation(); setShowAllMissing(v => !v); }}
                        style={{ fontSize: 11, padding: "3px 11px", borderRadius: 99, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.2)", color: "#fca5a5", cursor: "pointer", fontWeight: 600, fontFamily: "'Plus Jakarta Sans', sans-serif", transition: "all 0.15s" }}
                        onMouseEnter={e => { e.currentTarget.style.background = "rgba(239,68,68,0.2)"; }}
                        onMouseLeave={e => { e.currentTarget.style.background = "rgba(239,68,68,0.1)"; }}>
                        {showAllMissing ? "Show less ↑" : `+${res.missing_skills.length - 8} more ↓`}
                      </button>
                    )}
                  </div>
                </div>
              )}

              {res.nice_matched.length > 0 && (
                <div>
                  <p style={{ fontSize: 11, color: "#f59e0b", marginBottom: 6, fontWeight: 600 }}>★ Bonus ({res.nice_matched.length})</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {res.nice_matched.map((s, i) => <SkillTag key={i} label={s} type="bonus" />)}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [jobDescription, setJobDescription] = useState("");
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [jdAnalysis, setJdAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState({});
  const [showChart, setShowChart] = useState(false);
  const [showAllSkills, setShowAllSkills] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const handleFiles = (newFiles) => {
    const valid = Array.from(newFiles).filter(f => f.name.endsWith(".pdf") || f.name.endsWith(".docx"));
    setFiles(valid);
  };

  const removeFile = (i) => setFiles(f => f.filter((_, idx) => idx !== i));

  const handleSubmit = async () => {
    if (!jobDescription.trim() || files.length === 0) {
      setError("Please provide a job description and at least one resume.");
      return;
    }
    setError("");
    setLoading(true);
    setResults([]);
    setJdAnalysis(null);
    setExpanded({});
    const formData = new FormData();
    formData.append("job_desc", jobDescription);
    files.forEach(f => formData.append("resumes", f));
    try {
      const res = await fetch(`${API}/match`, { method: "POST", body: formData });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setResults(data.results || []);
      setJdAnalysis(data.jd_analysis || null);
      if (data.results?.length > 0) setExpanded({ 0: true });
    } catch {
      setError("Could not connect to backend. Make sure Flask is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setJobDescription("");
    setFiles([]);
    setResults([]);
    setJdAnalysis(null);
    setExpanded({});
    setShowChart(false);
    setShowAllSkills(false);
    setError("");
    document.getElementById("file-input").value = "";
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const chartData = results.map(r => ({
    name: r.name.replace(/\.[^.]+$/, "").slice(0, 14),
    score: parseFloat((r.score * 100).toFixed(1)),
  }));

  const card = { background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 20, padding: 28 };
  const label = { display: "block", fontSize: 11, fontWeight: 700, color: "#64748b", marginBottom: 10, letterSpacing: "0.07em", textTransform: "uppercase" };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060a12; }
        textarea, input, button, span, div, p, h1, h2, h3, label {
          font-family: 'Plus Jakarta Sans', sans-serif;
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
        .fade-up { animation: fadeUp 0.5s ease forwards; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
      `}</style>

      <div style={{ minHeight: "100vh", background: "#060a12", fontFamily: "'Plus Jakarta Sans', sans-serif", color: "#e2e8f0" }}>

        {/* Background glow */}
        <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
          <div style={{ position: "absolute", top: "-15%", left: "-5%", width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 65%)" }} />
          <div style={{ position: "absolute", bottom: "-15%", right: "0%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, rgba(16,185,129,0.05) 0%, transparent 65%)" }} />
        </div>

        <div style={{ position: "relative", zIndex: 1, maxWidth: 860, margin: "0 auto", padding: "52px 20px 80px" }}>

          {/* Header */}
          <div className="fade-up" style={{ textAlign: "center", marginBottom: 52 }}>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "5px 14px", borderRadius: 99, border: "1px solid rgba(99,102,241,0.3)", background: "rgba(99,102,241,0.08)", color: "#818cf8", fontSize: 11, fontWeight: 700, letterSpacing: "0.07em", marginBottom: 20 }}>
              <Brain size={11} /> AI-POWERED SCREENING
            </div>
            <h1 style={{ fontFamily: "'Plus Jakarta Sans', sans-serif", fontSize: "clamp(28px, 4.5vw, 44px)", fontWeight: 800, letterSpacing: "-0.02em", lineHeight: 1.15, color: "#f1f5f9", marginBottom: 14 }}>
              Smart Resume Screener
            </h1>
            <p style={{ color: "#64748b", fontSize: 15, fontWeight: 400, lineHeight: 1.6 }}>
              Paste a job description · Upload resumes · Get AI-ranked candidates
            </p>
          </div>

          {/* Input Card */}
          <div className="fade-up" style={{ ...card, marginBottom: 16, animationDelay: "0.1s", opacity: 0 }}>
            <label style={label}>Job Description</label>
            <textarea
              value={jobDescription}
              onChange={e => setJobDescription(e.target.value)}
              rows={5}
              placeholder="Paste the full job description — skills, responsibilities, requirements..."
              style={{ width: "100%", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: "13px 16px", color: "#cbd5e1", fontSize: 14, resize: "none", outline: "none", lineHeight: 1.65, transition: "border-color 0.2s" }}
              onFocus={e => e.target.style.borderColor = "rgba(99,102,241,0.5)"}
              onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.07)"}
            />

            <label style={{ ...label, marginTop: 22 }}>Resumes (PDF / DOCX)</label>
            <div
              onClick={() => document.getElementById("file-input").click()}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
              style={{ border: `2px dashed ${dragOver ? "rgba(99,102,241,0.55)" : "rgba(255,255,255,0.08)"}`, borderRadius: 14, padding: "22px 20px", textAlign: "center", cursor: "pointer", transition: "all 0.2s", background: dragOver ? "rgba(99,102,241,0.05)" : "transparent" }}>
              <Upload size={20} style={{ color: "#94a3b8", margin: "0 auto 8px", display: "block" }} />
              <p style={{ color: "#94a3b8", fontSize: 13 }}>Drop files here or <span style={{ color: "#6366f1" }}>click to browse</span></p>
              <input id="file-input" type="file" multiple accept=".pdf,.docx" style={{ display: "none" }} onChange={e => handleFiles(e.target.files)} />
            </div>

            {files.length > 0 && (
              <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 5 }}>
                {files.map((f, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 13px", background: "rgba(99,102,241,0.07)", border: "1px solid rgba(99,102,241,0.18)", borderRadius: 9 }}>
                    <FileText size={12} style={{ color: "#6366f1", flexShrink: 0 }} />
                    <span style={{ flex: 1, fontSize: 12, color: "#a5b4fc", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name}</span>
                    <button onClick={e => { e.stopPropagation(); removeFile(i); }} style={{ background: "none", border: "none", cursor: "pointer", color: "#64748b", display: "flex", padding: 2 }}><X size={12} /></button>
                  </div>
                ))}
              </div>
            )}

            {error && (
              <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8, color: "#f87171", fontSize: 13 }}>
                <AlertCircle size={13} /> {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading}
              style={{ marginTop: 20, width: "100%", padding: "14px", borderRadius: 12, border: "none", cursor: loading ? "not-allowed" : "pointer", fontFamily: "'Plus Jakarta Sans', sans-serif", fontWeight: 700, fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, transition: "opacity 0.2s, transform 0.1s", opacity: loading ? 0.55 : 1, background: "linear-gradient(135deg, #6366f1 0%, #4338ca 100%)", color: "#fff", boxShadow: loading ? "none" : "0 4px 24px rgba(99,102,241,0.3)", letterSpacing: "0.01em" }}
              onMouseEnter={e => { if (!loading) e.target.style.opacity = "0.88"; }}
              onMouseLeave={e => { e.target.style.opacity = "1"; }}>
              {loading
                ? <><Loader2 size={16} className="spin" /> Analyzing resumes...</>
                : <><Sparkles size={14} /> Analyze &amp; Rank Candidates</>}
            </button>

            {(jobDescription || files.length > 0 || results.length > 0) && (
              <button
                onClick={handleReset}
                style={{ marginTop: 10, width: "100%", padding: "11px", borderRadius: 12, border: "1px solid rgba(255,255,255,0.08)", background: "transparent", cursor: "pointer", fontFamily: "'Plus Jakarta Sans', sans-serif", fontWeight: 600, fontSize: 13, color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center", gap: 7, transition: "all 0.2s" }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(239,68,68,0.35)"; e.currentTarget.style.color = "#f87171"; e.currentTarget.style.background = "rgba(239,68,68,0.05)"; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"; e.currentTarget.style.color = "#94a3b8"; e.currentTarget.style.background = "transparent"; }}>
                <X size={13} /> Reset &amp; Start Over
              </button>
            )}
          </div>

          {/* JD Analysis */}
          {jdAnalysis && (
            <div className="fade-up" style={{ opacity: 0, animationDelay: "0.05s", background: "rgba(16,185,129,0.04)", border: "1px solid rgba(16,185,129,0.14)", borderRadius: 16, padding: "18px 22px", marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 14 }}>
                <Target size={13} style={{ color: "#10b981" }} />
                <span style={{ fontSize: 11, fontWeight: 700, color: "#10b981", letterSpacing: "0.07em", textTransform: "uppercase" }}>Job Description Analysis</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(90px, 1fr))", gap: 8, marginBottom: 14 }}>
                {[
                  { label: "Skills Found", val: jdAnalysis.total_skills_detected, color: "#10b981" },
                  { label: "Nice-to-Have", val: jdAnalysis.nice_to_have.length, color: "#f59e0b" },
                  { label: "Exp. Required", val: jdAnalysis.required_years ? `${jdAnalysis.required_years}+ yrs` : "—", color: "#38bdf8" },
                  { label: "Candidates", val: results.length, color: "#a78bfa" },
                ].map((s, i) => (
                  <div key={i} style={{ background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: 10, padding: "10px", textAlign: "center" }}>
                    <div style={{ fontSize: 19, fontWeight: 900, color: s.color, fontFamily: "'Plus Jakarta Sans', sans-serif" }}>{s.val}</div>
                    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>{s.label}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5, alignItems: "center" }}>
                {(showAllSkills ? jdAnalysis.required_skills : jdAnalysis.required_skills.slice(0, 24)).map((s, i) => (
                  <span key={i} style={{ fontSize: 11, padding: "3px 9px", borderRadius: 99, background: "rgba(16,185,129,0.09)", border: "1px solid rgba(16,185,129,0.18)", color: "#6ee7b7" }}>{s}</span>
                ))}
                {jdAnalysis.required_skills.length > 24 && (
                  <button
                    onClick={() => setShowAllSkills(v => !v)}
                    style={{ fontSize: 11, padding: "3px 11px", borderRadius: 99, background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.25)", color: "#a5b4fc", cursor: "pointer", fontWeight: 600, fontFamily: "'Plus Jakarta Sans', sans-serif", transition: "all 0.15s" }}
                    onMouseEnter={e => { e.currentTarget.style.background = "rgba(99,102,241,0.22)"; }}
                    onMouseLeave={e => { e.currentTarget.style.background = "rgba(99,102,241,0.12)"; }}>
                    {showAllSkills ? "Show less ↑" : `+${jdAnalysis.required_skills.length - 24} more ↓`}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Results header */}
          {results.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", margin: "20px 0 12px" }}>
              <span style={{ fontFamily: "'Plus Jakarta Sans', sans-serif", fontSize: 18, fontWeight: 800, color: "#f1f5f9" }}>
                {results.length} Candidate{results.length !== 1 ? "s" : ""} Ranked
              </span>
              <div style={{ display: "flex", gap: 7 }}>
                {[
                  { icon: <BarChart2 size={12} />, label: showChart ? "Hide Chart" : "Chart", action: () => setShowChart(v => !v), active: showChart },
                  { icon: <Download size={12} />, label: "CSV", action: () => window.open(`${API}/download`, "_blank"), active: false },
                ].map((b, i) => (
                  <button key={i} onClick={b.action} style={{ display: "flex", alignItems: "center", gap: 5, padding: "6px 13px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.09)", background: b.active ? "rgba(99,102,241,0.18)" : "transparent", color: b.active ? "#a5b4fc" : "#929599", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "'Plus Jakarta Sans', sans-serif", transition: "all 0.15s" }}>
                    {b.icon} {b.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chart */}
          {showChart && results.length > 0 && (
            <div style={{ background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 16, padding: "18px 12px", marginBottom: 14 }}>
              <ResponsiveContainer width="100%" height={190}>
                <BarChart data={chartData} barSize={26}>
                  <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                  <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid rgba(255,255,255,0.09)", borderRadius: 10, fontSize: 12 }} labelStyle={{ color: "#777e82" }} formatter={v => [`${v}%`, "Match"]} />
                  <Bar dataKey="score" radius={[5, 5, 0, 0]}>
                    {chartData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Candidate cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {results.map((res, idx) => (
              <div key={idx} className="fade-up" style={{ opacity: 0, animationDelay: `${idx * 0.07}s` }}>
                <CandidateCard
                  res={res}
                  rank={idx + 1}
                  expanded={!!expanded[idx]}
                  onToggle={() => setExpanded(p => ({ ...p, [idx]: !p[idx] }))}
                />
              </div>
            ))}
          </div>

        </div>
      </div>
    </>
  );
}