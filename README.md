# 🎯 Smart Resume Screener

An AI-powered resume screening tool that automatically analyzes job descriptions, extracts required skills, and ranks candidates based on how well their resumes match.

---

## 🚀 Features

- **Auto Skill Detection** — Extracts skills from any JD automatically using spaCy NER + noun chunks + keyword matching. No hardcoded lists.
- **Semantic Scoring** — Uses `sentence-transformers` to semantically match resume sections (experience, education) against the JD
- **Smart Skill Matching** — Alias matching so `sklearn` matches `scikit-learn`, `torch` matches `pytorch`, etc.
- **Grade System** — Candidates graded A–F with a plain-English verdict
- **Best Fit Badge** — Top candidate is highlighted automatically
- **Score Breakdown** — Skills coverage %, experience match, education match per candidate
- **Matched / Missing / Bonus Skills** — Clearly shown per candidate with expandable lists
- **CSV Export** — Download full results as a spreadsheet
- **Drag & Drop UI** — Clean dark interface with animated scores and bar chart

---

## 🛠️ Tech Stack

**Backend**
- Python + Flask
- sentence-transformers (`all-mpnet-base-v2`)
- spaCy (`en_core_web_lg`)
- PyMuPDF (PDF parsing)
- python-docx (DOCX parsing)

**Frontend**
- React + Vite
- Tailwind CSS
- Recharts
- Lucide React

---

## ⚙️ Setup & Run

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Run
python app.py
```

Flask will start on `http://127.0.0.1:5000`

---

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
echo VITE_API_URL=http://127.0.0.1:5000 > .env

# Run
npm run dev
```

Frontend will start on `http://localhost:5173`

---

## 📁 Project Structure

```
Smart-Resume-Screener/
├── backend/
│   ├── app.py              # Flask API
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/match` | Upload JD + resumes, get ranked results |
| GET | `/download` | Download results as CSV |

---

## 📊 How Scoring Works

| Component | Weight | Method |
|-----------|--------|--------|
| Skills Coverage | 35% | Keyword + alias matching |
| Experience Match | 30% | Semantic similarity |
| Education Match | 15% | Semantic similarity |
| Summary Match | 10% | Semantic similarity |
| Years of Experience | 10% | Date range extraction |

---

## 👤 Author

Built by **Shubham Ghodekar**
