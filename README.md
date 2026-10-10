# 🎯 Smart Resume Screener (Groq LLM + FastAPI)

An AI-powered resume screening tool that automatically analyzes job descriptions, extracts required skills, and ranks candidates with human-level accuracy using **Groq LLM**.

---

## 🚀 Features

- **Groq LLM Semantic Scoring** — Experience, education, and summary fit judged by the LLM (temperature 0 + fixed seed for stability).
- **Deterministic Skill Matching** — Required/nice-to-have skills extracted from the JD by code (same input = same output, every run), matched against resumes with alias awareness (`sklearn` = `scikit-learn`, `torch` = `pytorch`). Skill numbers can never hallucinate or drift between runs.
- **Grade System** — Candidates graded A–F with professional plain-English verdicts and detailed AI evaluation summaries.
- **Best Fit Badge** — Top candidate is highlighted automatically.
- **Score Breakdown** — Skills coverage %, experience match, and education match per candidate.
- **Matched / Missing / Bonus Skills** — Clearly shown per candidate with expandable lists.
- **CSV Export** — Download full screening results as a spreadsheet.
- **Drag & Drop UI** — Clean dark interface with animated scores and interactive comparison chart.

---

## 🛠️ Tech Stack

**Backend**
- Python + FastAPI + Uvicorn
- Groq Python SDK (`groq`)
- PyMuPDF (PDF parsing)
- python-docx (DOCX parsing)

**Frontend**
- React + Vite
- Tailwind CSS
- Recharts
- Lucide React

---

## ⚙️ Setup & Run

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python app.py
```

FastAPI will start on `http://127.0.0.1:5000` (interactive API docs available at `http://127.0.0.1:5000/docs`).

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
echo VITE_API_URL=http://127.0.0.1:5000 > .env

# Run development server
npm run dev
```

Frontend will start on `http://localhost:5173`.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/match` | Upload JD, Groq API key, model choice, & resumes (max 10 files, 10 MB each, PDF/DOCX); get ranked LLM results + per-run CSV |
| GET | `/download` | Download the latest screening results as CSV |
| GET | `/download/{filename}` | Download a specific run's CSV |
| POST | `/models` | Fetch live chat-capable Groq models for your API key |
| GET | `/health` | Health check |

> **First run:** paste your Groq key, click **Fetch Live**, pick a model from the dropdown, then analyze. Model IDs change over time, so always fetch live instead of typing one from memory.

---

## 👤 Author

Built by **Shubham Ghodekar**
