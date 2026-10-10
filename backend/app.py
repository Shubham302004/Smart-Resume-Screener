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
    """Normalize one LLM semantic-score block (never carries skills or grades)."""
    if not isinstance(r, dict):
        r = {}
    years = r.get("years_experience", 0)
    try:
        years = int(years)
    except (TypeError, ValueError):
        years = 0
    return {
        "name": str(r.get("name") or fallback_name),
        "experience_score": round(clamp01(r.get("experience_score", 0.0)), 3),
        "education_score": round(clamp01(r.get("education_score", 0.0)), 3),
        "summary_score": round(clamp01(r.get("summary_score", 0.0)), 3),
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
    "docker": ["containerization", "containers", "dockerfile", "docker compose"],
    "linux": ["unix", "ubuntu"],
    "data structures": ["data structure", "dsa"],
    "algorithms": ["algorithm", "algorithmic"],
    "prompt engineering": ["prompting", "prompt design"],
    "embeddings": ["embedding", "vector embeddings", "text embeddings"],
    "transformers": ["transformer models", "attention mechanism"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "llm": ["llms", "large language model", "large language models"],
    "fine-tuning": ["finetuning", "fine tuning", "peft", "lora", "qlora"],
    "langchain": ["lang chain"],
    "langflow": [],
    "chroma": ["chromadb", "chroma db"],
    "faiss": [],
    "pinecone": [],
    "vector databases": ["vector database", "vector db", "vector store", "vector stores"],
    "huggingface": ["hugging face"],
    "openai": ["gpt", "gpt-3.5", "gpt-4", "chatgpt"],
    "groq": ["groqcloud", "groq api"],
    "anthropic": ["claude"],
    "gemini": ["google gemini"],
    "fastapi": ["fast api"],
    "flask": [],
    "streamlit": [],
    "nodejs": ["node.js", "node js"],
    "react": ["reactjs", "react.js"],
    "angular": ["angularjs"],
    "vue": ["vuejs", "vue.js"],
    "django": [],
    "spring": ["spring boot"],
    "graphql": [],
    "grpc": [],
    "kafka": ["apache kafka"],
    "rabbitmq": [],
    "microservices": ["micro services", "micro-service"],
    "spark": ["apache spark", "pyspark"],
    "hadoop": [],
    "airflow": ["apache airflow"],
    "tableau": [],
    "excel": ["ms excel", "spreadsheets", "spreadsheet"],
    "sql": [],
    "mysql": [],
    "redis": [],
    "elasticsearch": ["elastic search"],
    "sqlite": [],
    "bigquery": ["big query"],
    "snowflake": [],
    "java": [],
    "c++": ["cpp"],
    "c#": ["c sharp", "csharp"],
    "kotlin": [],
    "swift": [],
    "scala": [],
    "matlab": [],
    "php": [],
    "ruby": ["ruby on rails"],
    "typescript": ["ts"],
    "html": [],
    "css": [],
    "terraform": [],
    "jenkins": [],
    "ansible": [],
    "ci/cd": ["cicd", "ci cd", "continuous integration", "continuous deployment"],
    "bash": ["shell scripting", "shell script"],
    "numpy": ["numerical python"],
    "scipy": [],
    "seaborn": [],
    "keras": [],
    "svm": ["support vector machine", "support vector machines"],
    "reinforcement learning": ["reinforcement-learning", "rlhf"],
    "generative ai": ["genai", "gen ai"],
    "linear algebra": ["matrices", "vectors"],
    "calculus": [],
    "communication": ["verbal communication", "written communication", "presentation skills"],
    "teamwork": ["team player"],
    "collaboration": ["cross-functional", "team collaboration"],
    "problem solving": ["analytical thinking", "critical thinking", "troubleshooting"],
    "leadership": ["mentoring", "team lead"],
}


def any_variant_in_text(variants: List[str], text_lower: str) -> bool:
    return any(re.search(r"(?<![\w.])" + re.escape(v.lower()) + r"(?![\w])", text_lower) for v in variants)


def skill_in_text(skill: str, text_lower: str) -> bool:
    """Check if a skill or any alias appears in lowercased text.

    Boundaries also exclude dots so dotted tokens don't false-hit
    short aliases ("Node.js" must not match "js"; "app.py" must not
    match "py"; "tf-idf" must not match "tf").
    """
    return any_variant_in_text([skill] + SKILL_ALIASES.get(skill.lower(), []), text_lower)


NICE_CUE_WORDS = (
    "nice-to-have", "nice to have", "nice-to-haves", "bonus", "plus",
    "preferred", "desirable", "optional", "good to have", "strong plus",
    "a plus", "pluses",
)

# Section headers that switch nice-to-have context on/off for following lines
NICE_HEADERS = (
    "nice-to-have", "nice to have", "bonus", "preferred", "desirable",
    "optional", "good to have", "plus",
)
OTHER_HEADERS = (
    "about the role", "about", "responsibilities", "responsibility",
    "requirements", "requirement", "required", "qualifications",
    "experience", "education", "location", "benefits", "compensation",
    "how to apply", "role", "what you'll do", "job title",
)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[\n;]+|[.!?]+\s+", text)
    return [p.strip() for p in parts if p.strip()]


def is_header(fragment: str, headers) -> bool:
    head = fragment.strip().lower().rstrip(":").strip()
    return any(head == h or head.startswith(h + " ") or head.startswith(h + ":") for h in headers)


def extract_jd_skills(jd_text: str) -> dict:
    """Deterministic JD skill extraction — identical output for identical input.

    1. Scan the JD for every known skill/alias.
    2. Skills appearing ONLY in nice-to-have cue sentences go to nice_to_have;
       everything else found is required.
    3. Years + role via regex.
    """
    text_lower = jd_text.lower()
    sentences = split_sentences(jd_text)

    # Section-aware cue tracking: a Nice-to-Have header marks following
    # lines as cue sentences until another section header appears.
    cue_flags = []
    in_nice_section = False
    for s in sentences:
        if is_header(s, NICE_HEADERS):
            in_nice_section = True
            cue_flags.append(True)
            continue
        if is_header(s, OTHER_HEADERS):
            in_nice_section = False
            cue_flags.append(False)
            continue
        cue_flags.append(in_nice_section or any(c in s.lower() for c in NICE_CUE_WORDS))
    cue_idx = {i for i, flag in enumerate(cue_flags) if flag}

    required, nice = [], []
    for skill in sorted(SKILL_ALIASES):
        if not skill_in_text(skill, text_lower):
            continue
        variants = [skill] + SKILL_ALIASES[skill]
        in_cue = any(any_variant_in_text(variants, s.lower()) for i, s in enumerate(sentences) if i in cue_idx)
        in_plain = any(any_variant_in_text(variants, s.lower()) for i, s in enumerate(sentences) if i not in cue_idx)
        # Skill only in cue sentences -> nice-to-have; otherwise required.
        # (No cue sentences at all -> everything is required.)
        if in_cue and not in_plain and cue_idx:
            nice.append(skill)
        else:
            required.append(skill)

    exp_match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", text_lower)
    required_years = int(exp_match.group(1)) if exp_match else None

    role_match = re.search(r"(?:role|position|title|job title)[:\s]+([^\n]+)", text_lower)
    role = role_match.group(1).strip().title() if role_match else "the role"

    return {
        "required_skills": required,
        "nice_to_have": nice,
        "required_years": required_years,
        "role": role,
        "total_skills_detected": len(required) + len(nice),
    }


def extract_years_from_resume(text: str) -> int:
    """Count years of experience from date ranges (deterministic)."""
    year_ranges = re.findall(r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current)", text.lower())
    current_year = datetime.now().year
    ranges = []
    for start, end in year_ranges:
        try:
            s = int(start)
            e = current_year if end in ("present", "current") else int(end)
            if 1990 <= s <= current_year and s < e and (e - s) <= 10:
                ranges.append((s, e))
        except ValueError:
            pass
    if not ranges:
        return 0
    ranges.sort()
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return min(sum(e - s for s, e in merged), 20)


def match_resume_skills(text_lower: str, required: List[str], nice: List[str]) -> dict:
    """Deterministic skill matching — identical output for identical input."""
    matched = [s for s in required if skill_in_text(s, text_lower)]
    missing = [s for s in required if not skill_in_text(s, text_lower)]
    nice_matched = [s for s in nice if skill_in_text(s, text_lower)]
    pct = (len(matched) / len(required) * 100) if required else 0.0
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "nice_matched": nice_matched,
        "skill_match_pct": round(pct, 1),
        "skills_score": round(pct / 100, 3),
    }


def years_score(resume_years: int, required_years) -> float:
    if not required_years:
        return 1.0
    if resume_years == 0:
        return 0.5
    if resume_years >= required_years:
        return 1.0
    return resume_years / required_years


STANDARD_VERDICTS = {
    "A": "Excellent match — highly recommended",
    "B": "Good match — worth interviewing",
    "C": "Partial match — review carefully",
    "D": "Weak match — significant gaps",
    "F": "Poor match — not recommended",
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
def analyze_with_groq(job_desc: str, resume_texts: List[dict], api_key: str, model_name: str = "openai/gpt-oss-120b"):
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
You are an expert technical recruiter. Below is a Job Description (JD) and candidate resumes.
For EACH candidate, judge only the semantic fit of their EXPERIENCE, EDUCATION, and overall SUMMARY against the JD.

JOB DESCRIPTION:
{job_desc}

CANDIDATES:
{resumes_payload}

Return strict JSON with this exact structure (use each resume's exact FILENAME as "name"):
{{
  "results": [
    {{
      "name": "filename.pdf",
      "experience_score": 0.80,
      "education_score": 0.70,
      "summary_score": 0.75,
      "years_experience": 3,
      "summary": "2-3 sentence professional evaluation of this candidate against the role."
    }}
  ]
}}

Rules:
- Each *_score is a float 0.0-1.0 judging that section's relevance to the JD. Be consistent and conservative.
- years_experience: total professional years inferred from the resume (0 if a fresher/no dates).
- summary: concise recruiter-style evaluation, no scores or grades inside.
- Do NOT extract or judge individual skills — a separate deterministic system handles that.

Ensure the output is valid JSON only, without markdown code blocks or extra text. Current year is {current_year}.
"""

    messages = [
        {"role": "system", "content": "You are a precise JSON-generating recruitment assistant. Always return valid JSON."},
        {"role": "user", "content": prompt},
    ]

    def run_call(use_json_mode: bool, max_tokens: int):
        kwargs = {"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "seed": 42}
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

    # 1. Deterministic JD analysis (identical output for identical input)
    jd = extract_jd_skills(job_desc)
    required = jd["required_skills"]
    nice = jd["nice_to_have"]

    # 2. LLM semantic section scores + prose (temperature 0 + seed for stability)
    analysis_data = analyze_with_groq(job_desc, resume_texts, groq_api_key, model_name.strip())

    raw_results = analysis_data.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    sections = [normalize_result(r, fallback_name=f"resume_{i}") for i, r in enumerate(raw_results)]
    by_name = {}
    for s in sections:
        by_name.setdefault(s["name"], s)
        by_name.setdefault(os.path.basename(s["name"]), s)

    # 3. Deterministic per-candidate blend (stable numbers; only LLM
    #    semantic sub-scores and prose can vary slightly between runs)
    results = []
    for i, rt in enumerate(resume_texts):
        text_lower = (rt["text"] or "").lower()
        skills = match_resume_skills(text_lower, required, nice)

        sec = by_name.get(rt["filename"]) or by_name.get(os.path.basename(rt["filename"]))
        if sec is None and i < len(sections):
            sec = sections[i]  # order fallback if the LLM renamed files
        if sec is None:
            sec = normalize_result({}, fallback_name=rt["filename"])

        det_years = extract_years_from_resume(rt["text"] or "")
        years = det_years if det_years > 0 else sec.get("years_experience", 0)
        y_score = years_score(years, jd["required_years"])

        final = (
            0.35 * skills["skills_score"]
            + 0.30 * sec.get("experience_score", 0.0)
            + 0.15 * sec.get("education_score", 0.0)
            + 0.10 * sec.get("summary_score", 0.0)
            + 0.10 * y_score
        )
        final = round(clamp01(final), 4)
        grade = grade_from_score(final)

        results.append({
            "name": rt["filename"],
            "score": final,
            "grade": grade,
            "verdict": STANDARD_VERDICTS[grade],
            "skills_score": skills["skills_score"],
            "experience_score": sec.get("experience_score", 0.0),
            "education_score": sec.get("education_score", 0.0),
            "skill_match_pct": skills["skill_match_pct"],
            "matched_skills": skills["matched_skills"],
            "missing_skills": skills["missing_skills"],
            "nice_matched": skills["nice_matched"],
            "years_experience": years,
            "summary": sec.get("summary", ""),
        })

    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    if results:
        results[0]["is_best"] = True
        for r in results[1:]:
            r["is_best"] = False

    download_file = write_results_csv(results) if results else None

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
