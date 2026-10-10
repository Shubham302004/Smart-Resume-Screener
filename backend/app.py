from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types as genai_types
import fitz
import docx
import os
import csv
import io
import re
import json
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Smart Resume Screener API", version="3.0")

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
PER_RESUME_CHARS = 4000
TOTAL_PROMPT_CHARS = 32000

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
            for table in d.tables:
                for row in table.rows:
                    for cell in row.cells:
                        parts.append(cell.text)
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
    if score >= 0.8: return "A"
    if score >= 0.65: return "B"
    if score >= 0.5: return "C"
    if score >= 0.35: return "D"
    return "F"


def normalize_result(r: dict, fallback_name: str = "unknown") -> dict:
    if not isinstance(r, dict): r = {}
    return {
        "name": str(r.get("name") or fallback_name),
        "experience_score": round(clamp01(r.get("experience_score", 0.0)), 3),
        "education_score": round(clamp01(r.get("education_score", 0.0)), 3),
        "summary_score": round(clamp01(r.get("summary_score", 0.0)), 3),
        "years_experience": max(0, int(r.get("years_experience", 0) or 0)),
        "summary": str(r.get("summary") or ""),
    }


def coerce_json(content: str) -> dict:
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match: return json.loads(match.group(0))
        raise


# --------------------------------------------------
# Skill Vocabulary & Logic
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
    "docker": ["containerization", "containers", "dockerfile", "docker compose"],
}

NICE_CUE_WORDS = (
    "nice-to-have", "nice to have", "nice-to-haves", "bonus", "plus",
    "preferred", "desirable", "optional", "good to have", "strong plus",
    "a plus", "pluses",
)

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


def any_variant_in_text(variants: List[str], text_lower: str) -> bool:
    return any(re.search(r"(?<![\w.])" + re.escape(v.lower()) + r"(?![\w])", text_lower) for v in variants)


def skill_in_text(skill: str, text_lower: str) -> bool:
    return any_variant_in_text([skill] + SKILL_ALIASES.get(skill.lower(), []), text_lower)


def extract_jd_skills(jd_text: str) -> dict:
    text_lower = jd_text.lower()
    sentences = split_sentences(jd_text)
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
        if not skill_in_text(skill, text_lower): continue
        variants = [skill] + SKILL_ALIASES[skill]
        in_cue = any(any_variant_in_text(variants, s.lower()) for i, s in enumerate(sentences) if i in cue_idx)
        in_plain = any(any_variant_in_text(variants, s.lower()) for i, s in enumerate(sentences) if i not in cue_idx)
        if in_cue and not in_plain and cue_idx: nice.append(skill)
        else: required.append(skill)

    exp_match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", text_lower)
    return {
        "required_skills": required,
        "nice_to_have": nice,
        "required_years": int(exp_match.group(1)) if exp_match else None,
        "role": (re.search(r"(?:role|position|title|job title)[:\s]+([^\n]+)", text_lower).group(1).strip().title() 
                 if re.search(r"(?:role|position|title|job title)[:\s]+([^\n]+)", text_lower) else "the role"),
        "total_skills_detected": len(required) + len(nice),
    }


EDU_INDICATORS = (
    "university", "college", "school", "bachelor", "b.tech", "b.e.", "master", "m.tech",
    "diploma", "degree", "hsc", "ssc", "10th", "12th", "phd", "graduation",
    "academics", "coursework", "education", "institute",
)
WORK_INDICATORS = (
    "intern", "experience", "work", "employ", "freelance", "self-employed", "engineer",
    "developer", "analyst", "designer", "trainee", "assistant", "staff", "lead",
    "manager", "consultant", "sde", "backend", "frontend", "full stack", "fullstack",
    "researcher", "tutor", "member",
)

DATE_RANGE_RE = re.compile(
    r"(?:[a-z]{3,9}\s+)?(\d{4})\s*(?:[-\u2013\u2014]|\bto\b)\s*(?:[a-z]{3,9}\s+)?(\d{4}|present|current|till|now)"
)


def _nearest_distance(needle_words, window: str, anchor: int) -> int:
    best = 10**9
    for w in needle_words:
        i = window.find(w)
        if i >= 0:
            best = min(best, abs(i - anchor))
    return best


def extract_years_from_resume(text: str) -> int:
    """Count professional years from date ranges (deterministic).

    Only ranges whose nearest context is WORK (not education) count, so a
    degree's "2023 - 2026" never inflates a fresher's experience to 6 years.
    """
    lower = text.lower()
    current_year = datetime.now().year
    ranges = []

    for m in DATE_RANGE_RE.finditer(lower):
        window = lower[max(0, m.start() - 150): m.end() + 150]
        anchor = m.start() - max(0, m.start() - 150)
        edu_d = _nearest_distance(EDU_INDICATORS, window, anchor)
        work_d = _nearest_distance(WORK_INDICATORS, window, anchor)
        if edu_d < work_d:
            continue  # education date range — not experience
        try:
            s = int(m.group(1))
            e = current_year if m.group(2) in ("present", "current", "till", "now") else int(m.group(2))
        except ValueError:
            continue
        if 1990 <= s <= current_year and s < e and (e - s) <= 10:
            ranges.append((s, e))

    years = 0
    if ranges:
        ranges.sort()
        merged = [ranges[0]]
        for s, e in ranges[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        years = sum(e - s for s, e in merged)

    # Prose fallback: "5+ years of experience" written without date ranges
    prose = re.search(r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+|industry\s+)?experience", lower)
    if prose:
        try:
            years = max(years, int(prose.group(1)))
        except ValueError:
            pass

    return min(years, 20)


def match_resume_skills(text_lower: str, required: List[str], nice: List[str]) -> dict:
    matched = [s for s in required if skill_in_text(s, text_lower)]
    return {
        "matched_skills": matched,
        "missing_skills": [s for s in required if not skill_in_text(s, text_lower)],
        "nice_matched": [s for s in nice if skill_in_text(s, text_lower)],
        "skill_match_pct": round((len(matched) / len(required) * 100) if required else 0.0, 1),
        "skills_score": round((len(matched) / len(required)) if required else 0.0, 3),
    }


def years_score(resume_years: int, required_years) -> float:
    if not required_years: return 1.0
    if resume_years == 0: return 0.5
    return min(1.0, resume_years / required_years)


# --------------------------------------------------
# Gemini LLM Integration
# --------------------------------------------------
def analyze_with_gemini(job_desc: str, resume_texts: List[dict], api_key: str, model_name: str = "gemini-2.5-flash"):
    try:
        client = genai.Client(api_key=api_key.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gemini API initialization error: {str(e)}")

    current_year = datetime.now().year
    chunks = []
    budget = TOTAL_PROMPT_CHARS
    for r in resume_texts:
        if budget <= 0: break
        text = (r["text"] or "")[:PER_RESUME_CHARS]
        take = min(len(text), budget)
        budget -= take
        chunks.append(f"RESUME: {r['filename']}\n{text[:take]}")

    prompt = f"""
You are an expert technical recruiter screening resumes against a job description.
For EACH candidate, judge only three things and return a JSON object.

JOB DESCRIPTION:
{job_desc}

CANDIDATES:
{chr(10).join(chunks)}

Return strict JSON with this exact structure (use each resume's exact FILENAME as "name"):
{{
  "results": [
    {{
      "name": "filename.pdf",
      "experience_score": 0.85,
      "education_score": 0.70,
      "summary_score": 0.80,
      "years_experience": 3,
      "summary": "2-3 sentence recruiter-style evaluation of this candidate against the role."
    }}
  ]
}}

Scoring rules:
- experience_score (0.0-1.0): how relevant the candidate's actual work/projects/internships are to this JD's responsibilities. Depth, impact, and domain relevance matter more than job titles. Freshers with relevant projects should still score reasonably (0.4-0.7).
- education_score (0.0-1.0): relevance of degree, branch, institution, and coursework to the role.
- summary_score (0.0-1.0): overall fit conveyed by the whole resume (skills mix, projects, achievements).
- years_experience: total PAID professional experience in years, computed ONLY from explicit date ranges in the resume (e.g. "Jan 2023 - Present" counts). Internships and part-time roles count. Personal/academic projects do NOT count. Use 0 for freshers with no work dates. Current year is {current_year}.
- summary: concise prose only — no scores, grades, or lists inside it.
- Do NOT extract or judge individual skills — a separate deterministic system handles that.
- Output ONLY valid JSON, no markdown fences or extra text.
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        return coerce_json(response.text or "")
    except Exception as e:
        msg = str(e)
        if "404" in msg or "not found" in msg.lower() or "is not supported" in msg.lower():
            raise HTTPException(status_code=400,
                                detail=f"Model '{model_name}' is unavailable for this key. Click Fetch Live and pick another model.")
        if "API key" in msg or "403" in msg or "401" in msg or "PERMISSION_DENIED" in msg or "UNAUTHENTICATED" in msg:
            raise HTTPException(status_code=400,
                                detail="Invalid or restricted Gemini API key. Check the key at aistudio.google.com.")
        raise HTTPException(status_code=400, detail=f"Gemini error: {msg}")


def write_results_csv(results: List[dict]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{stamp}.csv"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "name", "grade", "score", "verdict", "skills_score", "experience_score", 
            "education_score", "skill_match_pct", "years_experience", "matched_skills", "missing_skills", "nice_matched", "summary"
        ])
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({
                "rank": i + 1, "name": r["name"], "grade": r["grade"], "score": r["score"], "verdict": r["verdict"],
                "skills_score": r["skills_score"], "experience_score": r["experience_score"], "education_score": r["education_score"],
                "skill_match_pct": r["skill_match_pct"], "years_experience": r["years_experience"],
                "matched_skills": "; ".join(r["matched_skills"]), "missing_skills": "; ".join(r["missing_skills"]),
                "nice_matched": "; ".join(r["nice_matched"]), "summary": r["summary"]
            })
    with open(LATEST_POINTER, "w", encoding="utf-8") as f: f.write(filename)
    return filename


@app.get("/health")
async def health(): return {"status": "ok", "service": "Smart Resume Screener API", "version": "3.0"}


# Prefer newest, strongest models at the top of the dropdown
MODEL_PREFERENCE = ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash")


@app.post("/models")
async def get_gemini_models(api_key: str = Form(...)):
    try:
        client = genai.Client(api_key=api_key.strip())
        names = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gemini model fetch failed: {str(e)}")
    gemini = sorted(n for n in names if n.startswith("gemini-"))
    if not gemini:
        raise HTTPException(status_code=400, detail="No Gemini text models available for this API key")
    preferred = [m for m in MODEL_PREFERENCE if m in gemini]
    rest = [m for m in gemini if m not in preferred]
    return {"models": preferred + rest}


@app.post("/match")
async def match_resumes(job_desc: str = Form(...), api_key: str = Form(...), model_name: str = Form(...), resumes: List[UploadFile] = File(...)):
    if not job_desc.strip():
        raise HTTPException(status_code=400, detail="Job description is required")
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key is required")
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="Model is required — click Fetch Live and pick a model")
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume")
    if len(resumes) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files: {len(resumes)}. Maximum is {MAX_FILES}.")

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
        resume_texts.append({"id": idx, "filename": file.filename, "text": extracted if extracted.strip() else "[Could not extract text]"})

    jd = extract_jd_skills(job_desc)
    analysis_data = analyze_with_gemini(job_desc, resume_texts, api_key, model_name)

    raw_sections = analysis_data.get("results", [])
    if not isinstance(raw_sections, list):
        raw_sections = []
    sections = [normalize_result(r, fallback_name=f"resume_{i}") for i, r in enumerate(raw_sections)]
    by_name = {}
    for s in sections:
        by_name.setdefault(s["name"], s)
        by_name.setdefault(os.path.basename(s["name"]), s)

    results = []
    for i, rt in enumerate(resume_texts):
        text_lower = rt["text"].lower()
        skills = match_resume_skills(text_lower, jd["required_skills"], jd["nice_to_have"])
        sec = (by_name.get(rt["filename"])
               or by_name.get(os.path.basename(rt["filename"]))
               or (sections[i] if i < len(sections) else None)
               or normalize_result({}, rt["filename"]))
        det_years = extract_years_from_resume(rt["text"])
        y = det_years  # fully deterministic — LLM never touches years
        score = round(clamp01(0.35 * skills["skills_score"] + 0.30 * sec["experience_score"] + 0.15 * sec["education_score"] + 0.10 * sec["summary_score"] + 0.10 * years_score(y, jd["required_years"])), 4)
        grade = grade_from_score(score)
        results.append({
            "name": rt["filename"], "score": score, "grade": grade, "verdict": { "A": "Highly Recommended", "B": "Good Fit", "C": "Partial Fit", "D": "Weak Fit", "F": "Poor Fit" }[grade],
            "skills_score": skills["skills_score"], "experience_score": sec["experience_score"], "education_score": sec["education_score"],
            "skill_match_pct": skills["skill_match_pct"], "matched_skills": skills["matched_skills"], "missing_skills": skills["missing_skills"],
            "nice_matched": skills["nice_matched"], "years_experience": y, "summary": sec["summary"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        results[0]["is_best"] = True
        for r in results[1:]:
            r["is_best"] = False

    return {"results": results, "jd_analysis": jd, "download_file": write_results_csv(results) if results else None}


def _resolve_result_file(filename: str) -> str:
    """Path-traversal-safe resolution of a results CSV filename."""
    if not re.fullmatch(r"results_\d{8}_\d{4,6}\.csv", filename or ""):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.normpath(os.path.join(RESULTS_DIR, filename))
    if not path.startswith(os.path.normpath(RESULTS_DIR) + os.sep) or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return path


@app.get("/download/{filename}")
async def download_result_file(filename: str):
    return FileResponse(_resolve_result_file(filename), media_type="text/csv", filename="screening_results.csv")


@app.get("/download")
async def download_results():
    if not os.path.exists(LATEST_POINTER):
        raise HTTPException(status_code=404, detail="No results yet")
    with open(LATEST_POINTER, encoding="utf-8") as f:
        return FileResponse(_resolve_result_file(f.read().strip()), media_type="text/csv", filename="screening_results.csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
