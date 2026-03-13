from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import fitz
import docx
import os
import csv
import io
import re
import spacy

app = Flask(__name__)
CORS(app)

# Load models
model = SentenceTransformer("all-mpnet-base-v2")
nlp = spacy.load("en_core_web_lg")

UPLOAD_FOLDER = "uploads"
RESULTS_FILE = "results.csv"

# --------------------------------------------------
# Tech / domain skill keywords to boost extraction
# --------------------------------------------------
TECH_KEYWORDS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin", "swift", "r", "scala", "matlab",
    # Web / Backend
    "react", "angular", "vue", "node", "nodejs", "express", "django", "flask", "fastapi", "spring", "html", "css",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "sqlite", "bigquery",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd", "git", "linux", "bash",
    # APIs & Architecture
    "rest", "graphql", "grpc", "microservices", "kafka", "rabbitmq", "apis",
    # ML / AI Core
    "machine learning", "deep learning", "nlp", "computer vision", "data science",
    "scikit-learn", "tensorflow", "pytorch", "keras", "transformers",
    "neural networks", "reinforcement learning", "generative ai", "llm", "fine-tuning",
    "regression", "classification", "clustering", "random forest",
    "gradient boosting", "xgboost", "lightgbm", "svm",
    # ML Workflow
    "feature engineering", "data preprocessing", "eda", "exploratory data analysis",
    "model training", "model evaluation", "hyperparameter tuning", "cross-validation",
    "mlops", "model deployment", "data pipeline", "training", "deployment", "inference",
    "data collection", "data cleaning",
    # Math / Stats
    "statistics", "mathematics", "linear algebra", "calculus", "probability",
    "f1-score", "accuracy", "precision", "recall",
    # Data Tools
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "spark", "hadoop", "airflow", "data engineering",
    "tableau", "power bi", "excel", "jupyter",
    # Soft Skills
    "communication", "leadership", "teamwork", "problem solving", "project management",
    "collaboration", "agile", "scrum", "jira", "devops", "data analysis",
}


# --------------------------------------------------
# Skill Aliases — variations that mean the same skill
# --------------------------------------------------
SKILL_ALIASES = {
    "scikit-learn":             ["sklearn", "scikit learn", "sci-kit learn"],
    "tensorflow":               ["tf", "tensor flow", "tensorflow2", "tf2"],
    "pytorch":                  ["torch", "py torch"],
    "statistics":               ["stats", "statistical", "statistical analysis"],
    "neural networks":          ["ann", "cnn", "rnn", "lstm", "gru", "transformer", "deep neural", "artificial neural"],
    "machine learning":         ["ml", "ml models", "ml algorithms"],
    "deep learning":            ["dl"],
    "nlp":                      ["natural language processing", "text processing", "text analytics"],
    "computer vision":          ["cv", "image recognition", "object detection", "image processing"],
    "mathematics":              ["math", "maths", "mathematical"],
    "linear algebra":           ["matrices", "vectors", "matrix operations"],
    "probability":              ["probabilistic", "bayesian", "bayes"],
    "exploratory data analysis":["eda", "data exploration", "data exploration analysis"],
    "feature engineering":      ["feature extraction", "feature selection", "feature creation"],
    "data cleaning":            ["data wrangling", "data preprocessing", "preprocessing", "data preparation"],
    "data collection":          ["data gathering", "data acquisition", "web scraping", "scraping"],
    "model training":           ["training models", "model fitting", "fit model"],
    "model evaluation":         ["model assessment", "model testing", "evaluation metrics"],
    "model deployment":         ["deploy model", "model serving", "model inference", "productionize"],
    "hyperparameter tuning":    ["hyperparameter optimization", "grid search", "random search", "optuna"],
    "cross-validation":         ["k-fold", "kfold", "cross validation"],
    "data pipeline":            ["etl", "data workflow", "data processing pipeline"],
    "data engineering":         ["data infrastructure", "data architecture"],
    "data analysis":            ["data analytics", "analytical", "analysis"],
    "apis":                     ["api", "rest api", "restful api", "web api"],
    "aws":                      ["amazon web services", "amazon aws", "ec2", "s3", "lambda"],
    "azure":                    ["microsoft azure", "azure cloud"],
    "gcp":                      ["google cloud", "google cloud platform"],
    "docker":                   ["containerization", "containers"],
    "kubernetes":               ["k8s", "container orchestration"],
    "git":                      ["github", "gitlab", "version control", "bitbucket"],
    "python":                   ["python3", "python 3", "py"],
    "javascript":               ["js", "es6", "es2015"],
    "typescript":               ["ts"],
    "postgresql":               ["postgres"],
    "mongodb":                  ["mongo"],
    "f1-score":                 ["f1 score", "f-1 score", "f1", "f-score"],
    "accuracy":                 ["model accuracy", "test accuracy"],
    "regression":               ["linear regression", "logistic regression", "regression analysis"],
    "classification":           ["binary classification", "multiclass", "multi-class"],
    "clustering":               ["k-means", "kmeans", "dbscan", "hierarchical clustering"],
    "random forest":            ["random forests", "rf model"],
    "gradient boosting":        ["xgboost", "lightgbm", "catboost", "gbm"],
    "communication":            ["verbal communication", "written communication", "presentation skills"],
    "collaboration":            ["team collaboration", "cross-functional", "teamwork"],
    "problem solving":          ["analytical thinking", "critical thinking", "troubleshooting"],
    "agile":                    ["agile methodology", "agile development"],
    "mlops":                    ["ml ops", "ml operations", "model operations"],
    "jupyter":                  ["jupyter notebook", "jupyter lab", "ipynb", "notebook"],
    "pandas":                   ["dataframe", "data frames"],
    "numpy":                    ["np", "numerical python"],
    "matplotlib":               ["pyplot", "plt"],
}


def skill_matches(skill, text):
    """Check if a skill or any of its aliases appear in the text."""
    patterns = [skill] + SKILL_ALIASES.get(skill, [])
    for p in patterns:
        if re.search(r'\b' + re.escape(p) + r'\b', text):
            return True
    return False


# --------------------------------------------------
# Text Extraction
# --------------------------------------------------
def extract_text(file_stream, filename):
    """Extract text from PDF or DOCX using in-memory stream."""
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    elif filename.lower().endswith(".docx"):
        d = docx.Document(file_stream)
        return " ".join([p.text for p in d.paragraphs])
    return ""


# --------------------------------------------------
# JD Analysis — smart auto skill extraction
# --------------------------------------------------

# Words that are never skills — filter these out from noun chunks
NON_SKILL_WORDS = {
    "experience", "ability", "knowledge", "understanding", "familiarity",
    "background", "skills", "skill", "responsibilities", "responsibility",
    "requirements", "requirement", "qualifications", "qualification",
    "role", "position", "candidate", "team", "company", "business",
    "environment", "opportunity", "passion", "attitude", "mindset",
    "degree", "bachelor", "master", "phd", "university", "college",
    "year", "years", "month", "months", "day", "days",
    "plus", "bonus", "preferred", "required", "desired", "strong",
    "good", "great", "excellent", "solid", "proven", "hands",
    "work", "working", "job", "tasks", "task", "project", "projects",
    "addition", "including", "etc", "you", "we", "our", "their",
    "this", "that", "these", "those", "they", "them", "your",
    "time", "focus", "part", "set", "use", "area", "areas",
    "level", "field", "fields", "world", "space", "gap",
}

def extract_skills_from_text(text):
    """
    Smart skill extraction using:
    1. TECH_KEYWORDS exact match (with aliases)
    2. spaCy NER — catches product names, orgs (AWS, Google, Figma etc.)
    3. spaCy noun chunks — filtered for skill-like phrases
    4. Pattern matching for common skill formats (CamelCase tools, version numbers)
    """
    text_lower = text.lower()
    doc = nlp(text)
    skills = set()

    # 1. TECH_KEYWORDS — fast exact match
    for kw in TECH_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            skills.add(kw)
        else:
            for alias in SKILL_ALIASES.get(kw, []):
                if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                    skills.add(kw)
                    break

    # 2. spaCy NER — catches named entities like tools, platforms, companies used as skills
    for ent in doc.ents:
        val = ent.text.strip().lower()
        # PRODUCT, ORG entities are often tools/frameworks
        if ent.label_ in ("PRODUCT", "ORG", "GPE") and len(val) > 1:
            if val not in NON_SKILL_WORDS and not any(c.isdigit() for c in val[:1]):
                if 2 <= len(val.split()) <= 3:
                    skills.add(val)

    # 3. Noun chunks — filtered for skill-like 1-3 word phrases
    for chunk in doc.noun_chunks:
        val = chunk.text.strip().lower()
        words = val.split()
        if not (1 <= len(words) <= 3):
            continue
        # Skip if any word is a non-skill word
        if any(w in NON_SKILL_WORDS for w in words):
            continue
        # Skip pure stopwords
        if all(token.is_stop for token in chunk):
            continue
        # Skip if starts with article/determiner
        if chunk[0].pos_ in ("DET", "PRON", "ADP"):
            continue
        # Must contain at least one NOUN or PROPN
        if not any(token.pos_ in ("NOUN", "PROPN") for token in chunk):
            continue
        # Skip very generic single words
        if len(words) == 1 and chunk[0].pos_ not in ("PROPN",) and len(val) < 4:
            continue
        skills.add(val)

    # 4. Pattern matching — CamelCase tools (e.g. FastAPI, LangChain, OpenCV)
    camel_tools = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    for tool in camel_tools:
        skills.add(tool.lower())

    # 5. Versioned tools — e.g. Python 3, GPT-4, Node.js
    versioned = re.findall(r'\b([A-Za-z][A-Za-z0-9+#.\-]+(?:\s*[\d.]+)?)\b', text)
    for v in versioned:
        val = v.strip().lower()
        if 2 < len(val) < 30 and val not in NON_SKILL_WORDS:
            if re.search(r'[a-z]{2,}', val):  # must have letters
                if val in TECH_KEYWORDS or val in [a for aliases in SKILL_ALIASES.values() for a in aliases]:
                    skills.add(val)

    # Clean up — remove empty, single chars, pure numbers
    skills = {s for s in skills if len(s) > 1 and not s.isdigit() and s not in NON_SKILL_WORDS}

    return list(skills)


def analyze_job_description(jd_text):
    text_lower = jd_text.lower()

    # Auto-extract all skills
    all_skills = extract_skills_from_text(jd_text)

    # Detect nice-to-have skills
    nice_to_have = set()
    nice_patterns = [
        r"(?:nice[- ]to[- ]have|preferred|bonus|plus|desirable)[:\s]+([^\n.]+)",
        r"([^\n.]+)(?:\s+is a plus|\s+preferred|\s+is preferred)"
    ]
    for pattern in nice_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            for skill in all_skills:
                if skill in match:
                    nice_to_have.add(skill)

    required_skills = [s for s in all_skills if s not in nice_to_have]

    # Extract years of experience
    exp_match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", text_lower)
    required_years = int(exp_match.group(1)) if exp_match else None

    # Extract role title
    role_match = re.search(r"(?:role|position|title|job title)[:\s]+([^\n]+)", text_lower)
    role = role_match.group(1).strip().title() if role_match else "the role"

    return {
        "required_skills": required_skills,
        "nice_to_have": list(nice_to_have),
        "required_years": required_years,
        "role": role,
    }


# --------------------------------------------------
# Resume Section Parser (robust)
# --------------------------------------------------
SECTION_HEADERS = {
    "skills": [
        "skill", "technical skill", "core competenc", "technology", "technologies",
        "tools", "programming", "languages", "frameworks", "expertise"
    ],
    "experience": [
        "experience", "work history", "employment", "professional background",
        "career", "work experience", "job history", "positions held"
    ],
    "education": [
        "education", "academic", "qualification", "degree", "university",
        "college", "schooling", "certification", "training"
    ],
    "projects": [
        "project", "portfolio", "work sample", "case study"
    ],
    "summary": [
        "summary", "profile", "objective", "about me", "overview", "introduction"
    ]
}

def detect_section(line):
    line_lower = line.lower().strip()
    for section, keywords in SECTION_HEADERS.items():
        for kw in keywords:
            if kw in line_lower and len(line_lower) < 60:  # headers are usually short
                return section
    return None


def split_resume_sections(text):
    sections = {k: "" for k in SECTION_HEADERS}
    sections["full"] = text
    current = None

    for line in text.split("\n"):
        detected = detect_section(line)
        if detected:
            current = detected
        elif current:
            sections[current] += " " + line

    return sections


# --------------------------------------------------
# Experience Years Extractor from Resume
# --------------------------------------------------
def extract_years_from_resume(text):
    """Try to count total years of experience from date ranges in resume."""
    year_ranges = re.findall(r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current)", text.lower())
    current_year = 2025
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
    # Merge overlapping ranges to avoid double counting
    ranges.sort()
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return min(sum(e - s for s, e in merged), 20)  # cap at 20 years


# --------------------------------------------------
# Main Matching API
# --------------------------------------------------
@app.route("/match", methods=["POST"])
def match_resumes():
    job_desc = request.form.get("job_desc", "").strip()
    resumes = request.files.getlist("resumes")

    if not job_desc:
        return jsonify({"error": "Job description is required"}), 400
    if not resumes:
        return jsonify({"error": "At least one resume is required"}), 400

    # Deep JD analysis
    jd_analysis = analyze_job_description(job_desc)
    required_skills = jd_analysis["required_skills"]
    nice_to_have = jd_analysis["nice_to_have"]
    required_years = jd_analysis["required_years"]

    # Encode full JD + individual components for richer comparison
    jd_emb = model.encode(job_desc, convert_to_tensor=True)

    results = []

    for file in resumes:
        fname = file.filename
        file_bytes = io.BytesIO(file.read())

        # Extract text from in-memory stream
        text_raw = extract_text(file_bytes, fname)
        text = text_raw.lower()

        if not text.strip():
            results.append({
                "name": fname,
                "score": 0,
                "grade": "F",
                "verdict": "Could not extract text from file",
                "skills_score": 0,
                "experience_score": 0,
                "education_score": 0,
                "skill_match_pct": 0,
                "matched_skills": [],
                "missing_skills": required_skills,
                "nice_matched": [],
                "years_experience": 0,
                "summary": "Unable to parse this resume."
            })
            continue

        sections = split_resume_sections(text)

        # --- Semantic scoring (experience, education, summary only) ---
        def safe_encode(content, fallback):
            return model.encode(content.strip() if content.strip() else fallback, convert_to_tensor=True)

        exp_emb     = safe_encode(sections["experience"] + " " + sections["projects"], text)
        edu_emb     = safe_encode(sections["education"], text)
        summary_emb = safe_encode(sections["summary"] + " " + sections["full"], text)

        # semantic scores for experience, education, summary only
        exp_score     = util.cos_sim(jd_emb, exp_emb).item()
        edu_score     = util.cos_sim(jd_emb, edu_emb).item()
        summary_score = util.cos_sim(jd_emb, summary_emb).item()

        # --- Skill matching — extract skills from resume then compare ---
        resume_skills = set(extract_skills_from_text(text_raw))

        def resume_has_skill(s):
            if s.lower() in resume_skills:
                return True
            if re.search(r'\b' + re.escape(s) + r'\b', text):
                return True
            for alias in SKILL_ALIASES.get(s, []):
                if re.search(r'\b' + re.escape(alias) + r'\b', text):
                    return True
            return False

        matched_skills = [s for s in required_skills if resume_has_skill(s)]
        missing_skills = [s for s in required_skills if not resume_has_skill(s)]
        nice_matched   = [s for s in nice_to_have if resume_has_skill(s)]

        skill_match_pct = (len(matched_skills) / len(required_skills) * 100) if required_skills else 0

        # Skills score = keyword coverage (honest, not semantic)
        skills_score = skill_match_pct / 100

        # --- Experience years ---
        resume_years = extract_years_from_resume(text)
        years_score = 1.0
        if required_years:
            if resume_years == 0:
                years_score = 0.5
            elif resume_years >= required_years:
                years_score = 1.0
            else:
                years_score = resume_years / required_years

        # --- Final weighted score ---
        # skills keyword (35%) + experience semantic (30%) + education semantic (15%) + summary semantic (10%) + years (10%)
        final_score = (
            0.35 * skills_score +
            0.30 * exp_score +
            0.15 * edu_score +
            0.10 * summary_score +
            0.10 * years_score
        )

        # --- Grade (recalibrated for cosine similarity range 0.4–0.75) ---
        if final_score >= 0.65:
            grade = "A"
        elif final_score >= 0.52:
            grade = "B"
        elif final_score >= 0.40:
            grade = "C"
        elif final_score >= 0.28:
            grade = "D"
        else:
            grade = "F"

        # --- Verdict summary ---
        if grade == "A":
            verdict = "Excellent match — highly recommended"
        elif grade == "B":
            verdict = "Good match — worth interviewing"
        elif grade == "C":
            verdict = "Partial match — review carefully"
        elif grade == "D":
            verdict = "Weak match — significant gaps"
        else:
            verdict = "Poor match — not recommended"

        results.append({
            "name": fname,
            "score": round(final_score, 4),
            "grade": grade,
            "verdict": verdict,
            "skills_score": round(skills_score, 3),
            "experience_score": round(exp_score, 3),
            "education_score": round(edu_score, 3),
            "skill_match_pct": round(skill_match_pct, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "nice_matched": nice_matched,
            "years_experience": resume_years,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Mark the best candidate
    if results:
        results[0]["is_best"] = True
        for r in results[1:]:
            r["is_best"] = False

    # Write CSV
    if results:
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "rank", "name", "grade", "score", "verdict",
                "skills_score", "experience_score", "education_score",
                "skill_match_pct", "years_experience",
                "matched_skills", "missing_skills", "nice_matched"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, r in enumerate(results):
                writer.writerow({
                    "rank": i + 1,
                    "name": r["name"],
                    "grade": r["grade"],
                    "score": r["score"],
                    "verdict": r["verdict"],
                    "skills_score": r["skills_score"],
                    "experience_score": r["experience_score"],
                    "education_score": r["education_score"],
                    "skill_match_pct": r["skill_match_pct"],
                    "years_experience": r["years_experience"],
                    "matched_skills": "; ".join(r["matched_skills"]),
                    "missing_skills": "; ".join(r["missing_skills"]),
                    "nice_matched": "; ".join(r["nice_matched"]),
                })

    # Also return JD analysis for frontend display
    return jsonify({
        "results": results,
        "jd_analysis": {
            "required_skills": required_skills,
            "nice_to_have": nice_to_have,
            "required_years": required_years,
            "role": jd_analysis["role"],
            "total_skills_detected": len(required_skills),
        }
    })


# --------------------------------------------------
# CSV Download
# --------------------------------------------------
@app.route("/download", methods=["GET"])
def download_results():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results yet. Run a match first."}), 404
    return send_file(RESULTS_FILE, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)