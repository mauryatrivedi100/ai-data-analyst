# AI Data Analyst

A full-stack web application that enables users to upload CSV datasets and perform an end-to-end data science workflow — all without writing code. Includes data cleaning, exploratory data analysis, visualization, machine learning model training, AI-powered insights (via local Llama 3.2), and PDF report generation.

![Tech Stack](https://img.shields.io/badge/React-19-blue) ![Tech Stack](https://img.shields.io/badge/Flask-3.1-green) ![Tech Stack](https://img.shields.io/badge/Python-3.11+-yellow) ![Tech Stack](https://img.shields.io/badge/Tailwind-4-purple) ![Tech Stack](https://img.shields.io/badge/Ollama-Llama_3.2-orange)

## Features

- **Dataset Upload** — Drag-and-drop or browse CSV files (up to 50 MB)
- **Analysis Dashboard** — Row/column counts, data types, missing values, descriptive statistics
- **Data Cleaning** — Handle missing values, remove duplicates, detect outliers, rename/drop/convert columns, encode categoricals, scale numericals
- **Visualization** — Histograms, scatter plots, line charts, bar charts, pie charts, box plots, correlation heatmaps
- **Machine Learning** — Train classification (Logistic Regression, Decision Tree, Random Forest) and regression models with evaluation metrics
- **AI Insights** — Generate natural language analysis using your local Llama 3.2 model via Ollama
- **PDF Reports** — Download a comprehensive PDF report of your analysis session

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite 6, Tailwind CSS 4, Recharts 3, React Router 7, Axios |
| Backend | Flask 3.1, Pandas, NumPy, scikit-learn, fpdf2 |
| AI | Ollama + Llama 3.2 (local inference) |
| Testing | pytest, Hypothesis (property-based), Vitest, fast-check |

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Ollama** (for AI insights feature)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Frontend setup

```bash
cd frontend
npm install
```

### 4. Ollama setup (for AI Insights)

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
ollama serve
```

## Running the Application

### Start the backend (Terminal 1)

```bash
cd backend
python app.py
```

The Flask server starts at `http://localhost:5000`.

### Start the frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

The React app starts at `http://localhost:5173`.

### Open in browser

Navigate to `http://localhost:5173`

## Running Tests

### Backend tests

```bash
cd backend
python -m pytest tests/ -v
```

### Frontend tests

```bash
cd frontend
npx vitest run
```

## Project Structure

```
ai-data-analyst/
├── backend/
│   ├── app.py              # Flask application entry point
│   ├── routes.py           # API route definitions
│   ├── cleaning.py         # Data cleaning operations
│   ├── eda.py              # Exploratory data analysis computations
│   ├── ml.py               # Machine learning training & evaluation
│   ├── insights.py         # AI insights via Ollama/Llama 3.2
│   ├── report.py           # PDF report generation
│   ├── utils.py            # Shared utility functions
│   ├── requirements.txt    # Python dependencies
│   ├── uploads/            # Uploaded CSV files (gitignored)
│   ├── reports/            # Generated PDF reports (gitignored)
│   └── tests/              # Backend test suite
├── frontend/
│   ├── src/
│   │   ├── pages/          # React page components
│   │   ├── components/     # Shared UI components
│   │   ├── services/       # API service modules
│   │   ├── hooks/          # Custom React hooks
│   │   └── contexts/       # React Context providers
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a CSV file |
| GET | `/summary` | Dataset summary metadata |
| GET | `/statistics` | Descriptive statistics |
| POST | `/clean` | Apply cleaning operation |
| GET | `/download-cleaned` | Download cleaned CSV |
| GET | `/visualizations` | Get chart data |
| POST | `/train` | Train ML model |
| GET | `/feature-importance` | Feature importance scores |
| POST | `/generate-insights` | Generate AI insights |
| GET | `/download-report` | Download PDF report |
| POST | `/reset-session` | Clear session data |

## FAQ

**Q: What file formats are supported?**
Only CSV files are supported. The file must have a `.csv` extension and be between 1 byte and 50 MB in size.

**Q: Do I need an internet connection to use this app?**
No. Everything runs locally — the Flask backend, the React frontend, and the AI model (Ollama). No data leaves your machine.

**Q: The AI Insights feature isn't working. What do I check?**
Make sure Ollama is running (`ollama serve`) and that you've pulled the model (`ollama pull llama3.2`). The app connects to `http://localhost:11434`. If insights are slow, that's normal — local LLM inference takes 30–120 seconds depending on your hardware.

**Q: Can I use a different LLM model instead of Llama 3.2?**
Yes. Edit `backend/insights.py` and change the `OLLAMA_MODEL` variable to any model you have installed in Ollama (e.g., `mistral`, `llama3.1`, `gemma2`).

**Q: My file uploaded but the Analysis page shows no data.**
Restart the Flask backend (`Ctrl+C` then `python app.py`). This can happen if the server encountered an error during processing. Also ensure your CSV is properly formatted with comma delimiters.

**Q: Can I upload a new dataset without restarting the app?**
Yes. On the Home page, click "Start New Analysis" to clear the current dataset and all analysis results, then upload a new file.

**Q: What machine learning algorithms are available?**
Classification: Logistic Regression, Decision Tree, Random Forest. Regression: Linear Regression, Decision Tree, Random Forest. All use an 80/20 train/test split with a fixed random seed for reproducibility.

**Q: What does the PDF report include?**
The report includes only the steps you've completed: dataset summary, cleaning operations performed, model evaluation metrics, feature importance (for tree-based models), and AI-generated insights. Sections you haven't used are omitted.

**Q: How are outliers detected?**
Using the IQR (Interquartile Range) method. Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged as outliers.

**Q: Is my data stored permanently?**
Uploaded files are stored in `backend/uploads/` for the duration of your session. They are not sent to any external service. Use "Start New Analysis" to delete the file, or manually clear the `uploads/` folder.

**Q: Can I deploy this to a server?**
Yes, but you'd need Ollama running on that server for AI insights. For production, replace `app.run(debug=True)` with a proper WSGI server like Gunicorn, and build the frontend with `npm run build` to serve static files.

**Q: The app is slow with large datasets.**
Pandas operations on large files (>10 MB) can be CPU-intensive. Consider cleaning your data to reduce row count, or run on a machine with more RAM. The backend processes everything in-memory.

**Q: Feature importance says "only available for tree-based models." Why?**
Logistic Regression and Linear Regression don't expose `feature_importances_` in scikit-learn. Use Decision Tree or Random Forest to see feature importance rankings.

## License

MIT
