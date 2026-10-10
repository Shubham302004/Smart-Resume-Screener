# 🎯 Smart Resume Screener (Groq LLM + FastAPI)

An AI-powered resume screening tool that automatically analyzes job descriptions, extracts required skills, and ranks candidates with human-level accuracy using **Groq LLM**.

---

## 🚀 Features

- **Groq LLM-Powered Analysis** — Leverages state-of-the-art models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b`) for ultra-accurate semantic matching, grading, and candidate evaluation.
- **Smart Skill Detection** — Automatically extracts required and nice-to-have skills from any Job Description.
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
