from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
import fitz
import docx
import os
import csv
import io
import re
import json
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Smart Resume Screener API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LATEST_POINTER = os.path.join(RESULTS_DIR, "latest.txt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Upload guardrails
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILES = 10
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file
PER_RESUME_CHARS = 3000
TOTAL_PROMPT_CHARS = 24000

# Models that cannot be used with chat.completions (name fallback;
# preferred filter below uses live output_modality metadata)
NON_CHAT_SUBSTRINGS = ("whisper", "tts", "guard", "safeguard", "moderation", "embedding", "transcription", "orpheus")


def is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(s in lowered for s in NON_CHAT_SUBSTRINGS)


def model_allows_chat(m) -> bool:
    """Decide from live Groq metadata — catches speech/guard models
    whose names don't match the substring blocklist."""
    try:
        d = m.model_dump() if hasattr(m, "model_dump") else {}
        ins = [str(x).lower() for x in (d.get("input_modalities") or [])]
        outs = [str(x).lower() for x in (d.get("output_modalities") or [])]
        if not outs:  # metadata missing — fall back to name check
            return is_chat_model(getattr(m, "id", ""))
        return "text" in ins and "text" in outs and is_chat_model(getattr(m, "id", ""))
    except Exception:
        return is_chat_model(getattr(m, "id", ""))


def model_context_window(m, default: int = 32768) -> int:
    try:
        d = m.model_dump() if hasattr(m, "model_dump") else {}
        ctx = int(d.get("context_window") or default)
        return ctx if ctx > 0 else default
    except Exception:
        return default


# --------------------------------------------------
# Text Extraction with Robust Error Handling
# --------------------------------------------------
def extract_text(file_bytes: bytes, filename: str) -> str:
    try:
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                return " ".join([page.get_text() for page in doc])
            finally:
                doc.close()
        elif filename.lower().endswith(".docx"):
            d = docx.Document(io.BytesIO(file_bytes))
            parts = [p.text for p in d.paragraphs]
            # Tables are otherwise silently dropped
            for table in d.tables:
                for row in table.rows:
                    for cell in row.cells:
                        parts.append(cell.text)
            # Headers / footers
            for section in d.sections:
                for part in (section.header, section.footer):
                    for p in part.paragraphs:
                        parts.append(p.text)
            return " ".join(parts)
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
    return ""


def clamp01(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def grade_from_score(score: float) -> str:
    if score >= 0.8:
        return "A"
    if score >= 0.65:
        return "B"
    if score >= 0.5:
        return "C"
    if score >= 0.35:
        return "D"
    return "F"


def normalize_result(r: dict, fallback_name: str = "unknown") -> dict:
    """Fill LLM output gaps so sorting/CSV can never KeyError."""
    if not isinstance(r, dict):
        r = {}
    score = clamp01(r.get("score", 0.0))
    grade = r.get("grade") if r.get("grade") in ("A", "B", "C", "D", "F") else grade_from_score(score)
    skill_pct = r.get("skill_match_pct", 0.0)
    try:
        skill_pct = float(skill_pct)
    except (TypeError, ValueError):
        skill_pct = 0.0
    skill_pct = max(0.0, min(100.0, skill_pct))
    years = r.get("years_experience", 0)
    try:
        years = int(years)
    except (TypeError, ValueError):
        years = 0

    def str_list(v):
        return [str(x) for x in v] if isinstance(v, list) else []

    # matched_skills may be plain strings (old shape) or
    # {skill, evidence} objects (evidence-anchored shape)
    match_labels = []
    match_evidence = {}
    items = r.get("matched_skills", [])
    if isinstance(items, str):
        items = [items]
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                label = str(it.get("skill") or "").strip()
                ev = str(it.get("evidence") or "").strip()
                if label:
                    match_labels.append(label)
                    if ev:
                        match_evidence[label.lower()] = ev
            elif it is not None:
                s = str(it).strip()
                if s:
                    match_labels.append(s)

    def clean(skills):
        seen, out = set(), []
        for s in skills:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    return {
        "name": str(r.get("name") or fallback_name),
        "score": round(score, 4),
        "grade": grade,
        "verdict": str(r.get("verdict") or ""),
        "skills_score": round(clamp01(r.get("skills_score", 0.0)), 3),
        "experience_score": round(clamp01(r.get("experience_score", 0.0)), 3),
        "education_score": round(clamp01(r.get("education_score", 0.0)), 3),
        "skill_match_pct": round(skill_pct, 1),
        "matched_skills": clean(match_labels),
        "missing_skills": clean(str_list(r.get("missing_skills"))),
        "nice_matched": clean(str_list(r.get("nice_matched"))),
        "match_evidence": match_evidence,
        "years_experience": max(0, years),
        "summary": str(r.get("summary") or ""),
    }


# --------------------------------------------------
# Skill Aliases — deterministic safety net for LLM matching.
# The LLM sometimes marks a skill "missing" even when the resume
# contains it under a different name (sklearn vs scikit-learn).
# This pass can only rescue false-missing skills, never remove matches.
# --------------------------------------------------
SKILL_ALIASES = {
    "scikit-learn": ["sklearn", "scikit learn", "sci-kit learn"],
    "tensorflow": ["tensor flow"],
    "pytorch": ["torch", "py torch"],
    "statistics": ["stats", "statistical", "statistical analysis"],
    "neural networks": ["ann", "cnn", "rnn", "lstm", "gru", "transformer", "artificial neural"],
    "machine learning": ["ml", "ml models", "ml algorithms"],
    "deep learning": ["dl"],
    "nlp": ["natural language processing", "text processing", "text analytics"],
    "computer vision": ["cv", "image recognition", "object detection", "image processing"],
    "mathematics": ["math", "maths", "mathematical"],
    "probability": ["probabilistic", "bayesian", "bayes"],
    "exploratory data analysis": ["eda", "data exploration"],
    "feature engineering": ["feature extraction", "feature selection", "feature creation"],
    "data cleaning": ["data wrangling", "data preprocessing", "preprocessing", "data preparation"],
    "data collection": ["data gathering", "data acquisition", "web scraping", "scraping"],
    "model training": ["training models", "model fitting"],
    "model evaluation": ["model assessment", "model testing", "evaluation metrics"],
    "model deployment": ["deploy model", "model serving", "productionize"],
    "hyperparameter tuning": ["hyperparameter optimization", "grid search", "random search", "optuna"],
    "cross-validation": ["k-fold", "kfold", "cross validation"],
    "data pipeline": ["etl", "data workflow"],
    "data analysis": ["data analytics", "analytical"],
    "apis": ["api", "rest api", "restful api", "web api"],
    "aws": ["amazon web services", "ec2", "s3"],
    "azure": ["microsoft azure"],
    "gcp": ["google cloud", "google cloud platform"],
    "kubernetes": ["k8s", "container orchestration"],
    "git": ["github", "gitlab", "version control", "bitbucket"],
    "python": ["python3", "python 3", "py"],
    "javascript": ["js", "es6"],
    "postgresql": ["postgres"],
    "mongodb": ["mongo"],
    "f1-score": ["f1 score", "f-1 score", "f1", "f-score"],
    "regression": ["linear regression", "logistic regression", "regression analysis"],
    "classification": ["binary classification", "multiclass", "multi-class"],
    "clustering": ["k-means", "kmeans", "dbscan", "hierarchical clustering"],
    "random forest": ["random forests", "rf model"],
    "gradient boosting": ["xgboost", "lightgbm", "catboost", "gbm"],
    "jupyter": ["jupyter notebook", "jupyter lab", "ipynb", "notebook"],
    "pandas": ["dataframe", "data frames"],
    "matplotlib": ["pyplot", "plt"],
    "power bi": ["powerbi"],
    "mlops": ["ml ops", "ml operations"],
    "agile": ["agile methodology", "scrum"],
}


def skill_in_text(skill: str, text_lower: str) -> bool:
    """Check if a skill or any alias appears in lowercased resume text."""
    candidates = [skill.lower()] + SKILL_ALIASES.get(skill.lower(), [])
    return any(re.search(r"\b" + re.escape(p) + r"\b", text_lower) for p in candidates)


def evidence_in_text(evidence: str, text_lower: str) -> bool:
    """Check whether an LLM-provided evidence quote actually occurs in the resume."""
    ev = re.sub(r"\s+", " ", (evidence or "").lower()).strip(' "\'')
    if len(ev) < 4:
        return False
    if ev in text_lower:
        return True
    return len(ev) > 60 and ev[:60] in text_lower


STANDARD_VERDICTS = {
    "A": "Excellent match — highly recommended",
    "B": "Good match — worth interviewing",
    "C": "Partial match — review carefully",
    "D": "Weak match — significant gaps",
    "F": "Poor match — not recommended",
}


def apply_alias_correction(results: List[dict], resume_texts: List[dict], required: List[str], nice: List[str]) -> List[dict]:
    """Rescue LLM false-missing skills via deterministic alias matching.

    Only moves missing -> matched (never the reverse, so the LLM's
    semantic matches are preserved). Recomputes pct/skills_score and
    adjusts the overall score by the skills-component delta (35% weight).
    """
    texts = {rt["filename"]: (rt["text"] or "") for rt in resume_texts}
    basenames = {os.path.basename(k): v for k, v in texts.items()}

    def find_text(name: str) -> str:
        if name in texts:
            return texts[name]
        if os.path.basename(name) in basenames:
            return basenames[os.path.basename(name)]
        return ""

    req_keys = {s.lower() for s in required}
    nice_keys = {s.lower() for s in nice}

    for r in results:
        text_lower = find_text(r.get("name", "")).lower()
        if not text_lower.strip() or text_lower.strip() == "[could not extract text]":
            continue
        ev_map = r.get("match_evidence", {}) or {}

        # 1. Veto hallucinations: drop "matched" skills with neither
        #    skill/alias mention nor a verifiable evidence quote.
        vetted = []
        demoted = []
        for s in r.get("matched_skills", []):
            if skill_in_text(s, text_lower):
                vetted.append(s)
                continue
            if ev_map.get(s.lower(), "") and evidence_in_text(ev_map[s.lower()], text_lower):
                vetted.append(s)
                continue
            demoted.append(s)
        matched = vetted
        matched_keys = {s.lower() for s in matched}

        # 2. Rescue false-missing skills via alias matching
        still_missing = list(demoted)  # demoted skills rejoin missing
        for s in r.get("missing_skills", []):
            if s.lower() in matched_keys:
                continue  # duplicate across lists
            if skill_in_text(s, text_lower):
                matched.append(s)
                matched_keys.add(s.lower())
            elif s.lower() not in {m.lower() for m in still_missing}:
                still_missing.append(s)

        # 3. Gap-fill: required skills the LLM put in neither list
        for s in required:
            key = s.strip().lower()
            if key not in matched_keys and key not in {m.lower() for m in still_missing}:
                if skill_in_text(s, text_lower):
                    matched.append(key)
                    matched_keys.add(key)
                else:
                    still_missing.append(key)

        r["matched_skills"] = matched
        r["missing_skills"] = still_missing

        # Rescue bonus skills the LLM overlooked
        nice_matched = list(r.get("nice_matched", []))
        nice_keys_seen = {s.lower() for s in nice_matched}
        for s in nice:
            key = s.strip().lower()
            if key not in nice_keys_seen and skill_in_text(s, text_lower):
                nice_matched.append(key)
                nice_keys_seen.add(key)
        r["nice_matched"] = nice_matched

        # Recompute skills numbers from the corrected lists
        if req_keys:
            pct = len([s for s in matched if s.lower() in req_keys]) / len(req_keys) * 100
            new_skills = round(pct / 100, 3)
            delta = new_skills - r.get("skills_score", 0.0)
            r["skill_match_pct"] = round(pct, 1)
            r["skills_score"] = new_skills
            if abs(delta) > 1e-9:
                new_score = clamp01(r.get("score", 0.0) + 0.35 * delta)
                r["score"] = round(new_score, 4)
                new_grade = grade_from_score(new_score)
                if new_grade != r.get("grade"):
                    r["grade"] = new_grade
                    r["verdict"] = STANDARD_VERDICTS[new_grade]
    return results


def normalize_jd(jd) -> dict:
    """Fill LLM output gaps so the frontend never crashes on malformed jd_analysis."""
    if not isinstance(jd, dict):
        jd = {}

    def str_list(v):
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]

    years = jd.get("required_years")
    try:
        years = int(years) if years is not None else None
    except (TypeError, ValueError):
        years = None

    required = str_list(jd.get("required_skills"))
    total = jd.get("total_skills_detected", len(required))
    try:
        total = int(total)
    except (TypeError, ValueError):
        total = len(required)

    return {
        "required_skills": required,
        "nice_to_have": str_list(jd.get("nice_to_have")),
        "required_years": years,
        "role": str(jd.get("role") or "the role"),
        "total_skills_detected": total,
    }


def coerce_json(content: str) -> dict:
    """Parse model output, tolerating markdown fences and preamble text."""
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: largest {...} block in the response
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("No JSON object found", text, 0)


# --------------------------------------------------
# Groq LLM Powered Analysis & Scoring
# --------------------------------------------------
def analyze_with_groq(job_desc: str, resume_texts: List[dict], api_key: str, model_name: str = "llama-3.3-70b-versatile"):
    try:
        client = Groq(api_key=api_key.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Groq API Key initialization: {str(e)}")

    current_year = datetime.now().year

    # Adapt to the chosen model's context window (small models like
    # allam-2-7b would otherwise fail on max_tokens/prompt size)
    per_resume = PER_RESUME_CHARS
    total_budget = TOTAL_PROMPT_CHARS
    max_out = 4096
    try:
        for m in client.models.list().data:
            if m.id == model_name.strip():
                ctx = model_context_window(m)
                # Reserve output headroom; keep prompt within ~75% of context (~4 chars/token)
                max_out = max(1000, min(4096, ctx // 8))
                prompt_chars = max(4000, int(ctx * 3))
                per_resume = min(PER_RESUME_CHARS, prompt_chars // max(1, len(resume_texts)))
                total_budget = min(TOTAL_PROMPT_CHARS, prompt_chars)
                break
    except Exception:
        pass  # metadata lookup failed — proceed with safe defaults

    # Per-resume truncation + total budget so big batches can't blow context
    chunks = []
    budget = total_budget
    for r in resume_texts:
        if budget <= 0:
            break
        text = (r["text"] or "")[:per_resume]
        take = min(len(text), budget)
        text = text[:take]
        budget -= take
        suffix = "" if len(r["text"] or "") <= len(text) else " [truncated]"
        chunks.append(f"--- RESUME ID: {r['id']} ---\nFILENAME: {r['filename']}\nTEXT:\n{text}{suffix}")
    resumes_payload = "\n\n".join(chunks)

    prompt = f"""
You are an expert technical recruiter and AI resume screener.
Analyze the following Job Description (JD) and evaluate each candidate resume provided below.

JOB DESCRIPTION:
{job_desc}

CANDIDATES:
{resumes_payload}

Provide your analysis in strict JSON format with the following exact structure:
{{
  "jd_analysis": {{
    "required_skills": ["python", "pytorch"],
    "nice_to_have": ["docker"],
    "required_years": 2,
    "role": "Extracted Job Title",
    "total_skills_detected": 3
  }},
  "results": [
    {{
      "name": "filename.pdf",
      "score": 0.85,
      "grade": "A",
      "verdict": "Short 1-sentence verdict",
      "skills_score": 0.90,
      "experience_score": 0.80,
      "education_score": 0.70,
      "skill_match_pct": 85.0,
      "matched_skills": [{{"skill": "python", "evidence": "exact short quote from the resume proving it"}}],
      "missing_skills": ["skill2"],
      "nice_matched": ["skill3"],
      "years_experience": 3,
      "summary": "Detailed professional evaluation summary of this candidate against the role."
    }}
  ]
}}

Rules for skills (important for accurate matched/missing analysis):
- Be EXHAUSTIVE: scan the ENTIRE job description — responsibilities, requirements, and nice-to-have sections — and do not omit any tool, language, framework, platform, or methodology mentioned.
- List every skill ATOMICALLY: one tool, language, or concept per entry.
- Never group items, e.g. write ["pytorch", "tensorflow", "scikit-learn"], NOT ["ML/DL frameworks (PyTorch, TensorFlow, scikit-learn)"].
- Normalize common names to lowercase canonical forms (e.g. "python", "rest apis", "prompt engineering").
- Put genuinely optional/bonus items in nice_to_have; everything else in required_skills.
- required_years must be a number or null; role must be the job title.
- Alias awareness: treat common variants as the SAME skill (sklearn = scikit-learn, torch = pytorch, js = javascript, k8s = kubernetes, eda = exploratory data analysis, ml = machine learning). A resume mentioning any variant COUNTS as having the skill — never list it as missing. (Note: do NOT count "tf-idf" as TensorFlow, "lambda functions" as AWS, or file extensions like ".py" as anything but Python.)
- matched_skills must contain ONLY skills explicitly present in the resume text (exact or alias variant). Every entry needs an "evidence" field with an exact short quote copied from the resume. Never infer or hallucinate: if the resume implies but does not state a skill, put it in missing_skills.

Ensure the output is valid JSON only, without markdown code blocks or extra text. Current year for experience calculation is {current_year}.
"""

    messages = [
        {"role": "system", "content": "You are a precise JSON-generating recruitment assistant. Always return valid JSON."},
        {"role": "user", "content": prompt},
    ]

    def run_call(use_json_mode: bool, max_tokens: int):
        kwargs = {"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": 0.1, "seed": 42}
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    # Primary attempt, then fallbacks for models without JSON mode / smaller limits
    attempts = [(True, max_out), (False, max_out), (False, max(1000, max_out // 2))]
    last_error: Optional[Exception] = None
    for use_json_mode, max_tokens in attempts:
        try:
            completion = run_call(use_json_mode, max_tokens)
            content = completion.choices[0].message.content
            return coerce_json(content)
        except json.JSONDecodeError as jde:
            raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {str(jde)}")
        except HTTPException:
            raise
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            # Only retry on capability-style errors; auth/rate-limit errors surface immediately
            if "response_format" in msg or "json" in msg or "max_tokens" in msg or "too large" in msg or "context" in msg:
                continue
            raise HTTPException(status_code=400, detail=f"Groq API Error: {str(e)}")
    raise HTTPException(status_code=400, detail=f"Groq API Error after fallbacks: {str(last_error)}")


def write_results_csv(results: List[dict]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{stamp}.csv"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank", "name", "grade", "score", "verdict",
            "skills_score", "experience_score", "education_score",
            "skill_match_pct", "years_experience",
            "matched_skills", "missing_skills", "nice_matched", "summary",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({
                "rank": i + 1,
                "name": r["name"],
                "grade": r.get("grade", "C"),
                "score": r.get("score", 0.0),
                "verdict": r.get("verdict", ""),
                "skills_score": r.get("skills_score", 0.0),
                "experience_score": r.get("experience_score", 0.0),
                "education_score": r.get("education_score", 0.0),
                "skill_match_pct": r.get("skill_match_pct", 0.0),
                "years_experience": r.get("years_experience", 0),
                "matched_skills": "; ".join(r.get("matched_skills", [])),
                "missing_skills": "; ".join(r.get("missing_skills", [])),
                "nice_matched": "; ".join(r.get("nice_matched", [])),
                "summary": r.get("summary", ""),
            })
    # Pointer to the latest run for the legacy /download endpoint
    with open(LATEST_POINTER, "w", encoding="utf-8") as f:
        f.write(filename)
    return filename


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "Smart Resume Screener API", "version": "2.2"}


@app.post("/models")
async def get_groq_models(groq_api_key: str = Form(...)):
    if not groq_api_key.strip():
        raise HTTPException(status_code=400, detail="Groq API Key is required")
    try:
        client = Groq(api_key=groq_api_key.strip())
        response = client.models.list()
        chat = [m for m in response.data if model_allows_chat(m)]
        # Most capable first (largest context), so auto-select picks well
        chat.sort(key=lambda m: (-model_context_window(m), m.id))
        return {"models": [m.id for m in chat]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch live models from Groq: {str(e)}")


@app.post("/match")
async def match_resumes(
    job_desc: str = Form(...),
    groq_api_key: str = Form(...),
    model_name: str = Form(...),
    resumes: List[UploadFile] = File(...),
):
    if not job_desc.strip():
        raise HTTPException(status_code=400, detail="Job description is required")
    if not groq_api_key.strip():
        raise HTTPException(status_code=400, detail="Groq API Key is required")
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="Select a model via Fetch Live first.")
    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required")
    if len(resumes) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Maximum is {MAX_FILES} resumes per run.")

    resume_texts = []
    for idx, file in enumerate(resumes):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}. Only PDF and DOCX are allowed.")
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_BYTES:
            raise HTTPException(status_code=400, detail=f"File too large: {file.filename}. Maximum is 10 MB per file.")
        if not file_bytes:
            raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}.")
        extracted = extract_text(file_bytes, file.filename)
        resume_texts.append({
            "id": idx,
            "filename": file.filename,
            "text": extracted if extracted.strip() else "[Could not extract text]",
        })

    analysis_data = analyze_with_groq(job_desc, resume_texts, groq_api_key, model_name.strip())

    raw_results = analysis_data.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    results = [normalize_result(r, fallback_name=f"resume_{i}") for i, r in enumerate(raw_results)]

    jd = normalize_jd(analysis_data.get("jd_analysis"))

    # Deterministic alias safety net: rescue skills the LLM wrongly marked missing
    results = apply_alias_correction(results, resume_texts, jd.get("required_skills", []), jd.get("nice_to_have", []))

    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    if results:
        results[0]["is_best"] = True
        for r in results[1:]:
            r["is_best"] = False

    download_file = write_results_csv(results) if results else None

    # Strip internal evidence map from the API response
    for r in results:
        r.pop("match_evidence", None)

    return {
        "results": results,
        "jd_analysis": jd,
        "download_file": download_file,
    }


def _resolve_result_file(filename: str) -> str:
    if not re.fullmatch(r"results_\d{8}_\d{6}\.csv", filename or ""):
        raise HTTPException(status_code=400, detail="Invalid result file name.")
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result file not found. Run a match first.")
    return path


@app.get("/download/{filename}")
async def download_result_file(filename: str):
    path = _resolve_result_file(filename)
    return FileResponse(path, media_type="text/csv", filename="resume_screening_results.csv")


@app.get("/download")
async def download_results():
    # Legacy endpoint: serves the latest run
    if not os.path.exists(LATEST_POINTER):
        raise HTTPException(status_code=404, detail="No results yet. Run a match first.")
    with open(LATEST_POINTER, encoding="utf-8") as f:
        latest = f.read().strip()
    path = _resolve_result_file(latest)
    return FileResponse(path, media_type="text/csv", filename="resume_screening_results.csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
