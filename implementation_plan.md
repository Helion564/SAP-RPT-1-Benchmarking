# Automated Benchmarking Web Application

We will build a beautiful, modern web application that allows users to upload their own CSV datasets, automatically benchmarks all available models (XGBoost, LightGBM, CatBoost, SAP RPT-1 simulator) against the uploaded data, and recommends the best model for their specific use case.

## User Review Required

> [!IMPORTANT]
> **Architecture Decision**: I propose using **FastAPI** for the backend (since our benchmarking code is already in Python) and a custom **Vanilla HTML/CSS/JS** frontend. This allows us to create a highly polished, responsive, and visually stunning user interface with dynamic animations and charts, avoiding the generic look of tools like Streamlit. Do you approve of this tech stack?

> [!NOTE]
> **Execution Time**: Benchmarking multiple models on an uploaded CSV might take a few seconds to a minute depending on the size of the dataset. We will add a beautiful loading animation on the frontend to keep the user engaged while the backend processes the data.

## Open Questions

1. Do you want to limit the size of the uploaded CSV files (e.g., max 5MB) to prevent the server from running out of memory during cross-validation?
2. Are you okay with continuing to use the SAP RPT-1 Simulator (k-NN) for the web demo, or do you have your Hugging Face token ready to plug in the real SAP RPT-1 model?

## Proposed Changes

We will create a new directory `webapp/` in the repository to house the full stack.

### Backend (FastAPI)

The backend will expose an endpoint to receive the CSV, save it temporarily, and run the evaluation pipeline from `demo_benchmark.py` (adapted for a single dynamic dataset).

#### [NEW] [webapp/main.py](file:///c:/Users/HP/SAP-RPT1-Benchmarking19/webapp/main.py)
- Setup FastAPI server.
- Define a `POST /benchmark` endpoint that accepts a `UploadFile`.
- Logic to read the CSV into a pandas DataFrame.
- Auto-detect the task type (classification vs. regression) based on the target column (which we can assume is the last column by default, or let the user specify).
- Run the 5-fold cross-validation loop for XGBoost, LightGBM, CatBoost, and SAP-RPT1 (sim).
- Calculate the "Best Model" based on the primary metric (ROC-AUC or R²).
- Return the structured JSON results to the frontend.

#### [NEW] [webapp/requirements.txt](file:///c:/Users/HP/SAP-RPT1-Benchmarking19/webapp/requirements.txt)
- Dependencies: `fastapi`, `uvicorn`, `python-multipart` (along with existing ML libs).

---

### Frontend (HTML/CSS/JS)

A stunning, glassmorphism-inspired UI with smooth transitions, drag-and-drop file upload, and dynamic charts.

#### [NEW] [webapp/static/index.html](file:///c:/Users/HP/SAP-RPT1-Benchmarking19/webapp/static/index.html)
- Modern layout with a hero section.
- A drag-and-drop zone for CSV files.
- A results dashboard area that remains hidden until data is returned.
- Recommendation banner highlighting the best model.

#### [NEW] [webapp/static/style.css](file:///c:/Users/HP/SAP-RPT1-Benchmarking19/webapp/static/style.css)
- Vibrant, curated dark-mode color palette.
- Smooth micro-animations for buttons and the upload zone.
- Loading spinners and transitions.

#### [NEW] [webapp/static/app.js](file:///c:/Users/HP/SAP-RPT1-Benchmarking19/webapp/static/app.js)
- Handle drag-and-drop events and file selection.
- Send the `FormData` via `fetch` API to the `/benchmark` endpoint.
- Parse the JSON response.
- Render dynamic comparison charts using `Chart.js`.
- Render the metrics table and the "Best Model Recommendation" card.

## Verification Plan

### Automated Tests
- Start the FastAPI server using `uvicorn`.
- We will use the `browser_subagent` to navigate to the web app, upload one of our generated datasets (e.g., `datasets/iris_X.csv` combined with target, or just `datasets/iris.csv`), and verify that the results render correctly on the screen.

### Manual Verification
- The user can open `http://localhost:8000` in their local browser.
- The user can upload their own custom CSV file.
- The user can verify that the UI gracefully handles the processing time and displays a gorgeous recommendation dashboard at the end.
