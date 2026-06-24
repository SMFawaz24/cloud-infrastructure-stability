# Cloud Infrastructure Stability Analysis

A time-series anomaly detection system for cloud infrastructure metrics, built with a Flask web dashboard, Z-Score statistical detection, and an Isolation Forest machine learning model. Includes MLflow experiment tracking and an optional Hugging Face Spaces deployment.

**Shreyans Modi — RA2311026010720**  
**Syed Mohammad Fawaz — RA2311026010780**  
SRM Institute of Science and Technology  
B.Tech Computer Science and Engineering, 2024

---

## What it does

The system simulates 200 minutes of cloud server activity across four metrics — CPU usage, memory usage, disk usage, and network latency. It injects six realistic anomaly events into the data, each affecting only the metrics that would actually spike in that kind of incident (a memory leak raises CPU and memory, a network issue only raises latency). Two independent detectors then scan the data and raise alerts with severity classifications. All results are served through a multi-page analytics dashboard.

The web dashboard has five sections:

- **Overview** — eight KPI cards covering detection counts, recall, and precision, with sparkline charts for each metric
- **Time Series** — four annotated full charts, one per metric, with markers for ground truth anomalies, Z-Score alerts, and Isolation Forest alerts overlaid
- **Alerts** — a filterable table showing every detected alert with timestamp, metric values, severity, which detector fired, and whether it was a true positive
- **Analytics** — recall and precision comparison bars, severity breakdown chart, Z-Score over time with threshold line, Pearson correlation heatmap, and a method agreement doughnut chart
- **Statistics** — descriptive stats table (mean, std, min, P25, P50, P75, P95, max) and a box plot for all four metrics

---

## Project structure

```
cloud-infrastructure-stability/
│
├── webapp/
│   ├── server.py               Flask backend — runs the pipeline, exposes JSON API
│   └── templates/
│       └── index.html          Full dashboard — HTML, CSS, JS, Chart.js
│
├── src/
│   ├── detection.py            Core engine — data generation, Z-Score, Isolation Forest
│   ├── visualize.py            Matplotlib chart generation (batch and live modes)
│   └── __init__.py
│
├── mlflow_tracking/
│   ├── train.py                MLflow-integrated run — logs params, metrics, model
│   └── export_model.py         Exports the trained model from MLflow to .pkl
│
├── huggingface/
│   ├── app.py                  Gradio web app for Hugging Face Spaces deployment
│   └── requirements.txt
│
├── app.py                      Standalone Gradio app — runs locally without HF
├── run.py                      Terminal-only script — no browser, saves chart to disk
└── requirements.txt
```

---

## Getting started

**Python 3.8 or later is required.**

```bash
git clone https://github.com/SMFawaz24/cloud-infrastructure-stability.git
cd cloud-infrastructure-stability

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the project

### Web dashboard (recommended)

Starts the Flask server and opens the full analytics dashboard.

```bash
python webapp/server.py
```

Open `http://127.0.0.1:5050` in your browser.

### Terminal only

Runs the detection pipeline, prints all alerts, and saves the 4-panel chart to `outputs/stability_chart.png`.

```bash
python run.py
```

### Gradio app (local)

Runs an interactive web interface where you can submit metric readings manually and see live detection results.

```bash
python app.py
```

Open `http://127.0.0.1:7860`.

### MLflow experiment tracking

```bash
# Run the pipeline with full MLflow logging
python mlflow_tracking/train.py

# Open the MLflow tracking UI
mlflow ui
```

Open `http://127.0.0.1:5000` to see logged parameters, metrics, the saved chart artifact, and the registered Isolation Forest model.

```bash
# Export the trained model for use in the Gradio app
python mlflow_tracking/export_model.py
```

---

## Detection methods

### Z-Score

Each metric is standardised across the dataset. A reading is flagged as anomalous if any metric's absolute Z-Score exceeds the threshold — meaning it sits that many standard deviations away from the column mean.

```
Z = (value − mean) / standard deviation
```

The threshold is 2.5 in batch mode (`run.py`, `train.py`) and 1.8 in the interactive Gradio app, where the session history is shorter and a lower threshold gives more responsive detection.

### Isolation Forest

An ensemble of random decision trees partitions the data by selecting random features and split values. Points in sparse regions of the feature space require fewer splits to isolate — they have shorter average path lengths — and are scored as anomalies. The model is trained once on 500 simulated normal readings and serialised to `isolation_forest_model.pkl` by `export_model.py`.

The contamination parameter is set to 0.03, reflecting the 3% anomaly rate in the simulated data.

### Severity classification

Detected anomalies are graded based on CPU and memory thresholds:

| Level | Condition |
|---|---|
| CRITICAL | CPU above 90% or memory above 90% |
| HIGH | CPU above 75% or memory above 75% |
| MODERATE | Any other flagged reading |

---

## MLflow tracked values

Every run of `mlflow_tracking/train.py` logs the following:

| Category | Key |
|---|---|
| Parameters | `n_data_points`, `random_seed`, `zscore_threshold`, `iforest_contamination`, `detection_methods` |
| Metrics | `zscore_recall`, `iforest_recall`, `zscore_detections`, `iforest_detections`, `zscore_true_positives`, `iforest_true_positives`, `injected_anomalies` |
| Artifacts | `stability_chart.png` — the full 4-panel annotated chart |
| Model Registry | `CloudStabilityIForest` — the trained Isolation Forest, loadable via `mlflow.sklearn.load_model()` |

---

## Hugging Face deployment

After running `mlflow_tracking/export_model.py`, upload these three files to a new Gradio Space on [huggingface.co/spaces](https://huggingface.co/spaces):

```
huggingface/app.py
huggingface/requirements.txt
isolation_forest_model.pkl
```

Set the SDK to **Gradio** during Space creation. The app will build automatically and be live at `https://huggingface.co/spaces/your-username/your-space-name` within two minutes.

---

## AWS EC2 deployment

For a persistent public URL, the Flask dashboard can be deployed on an EC2 t2.micro instance (free tier eligible) behind nginx.

1. Launch an Ubuntu 22.04 t2.micro instance with HTTP/HTTPS allowed in the security group
2. Allocate and associate an Elastic IP
3. SSH in and run:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv nginx unzip
# Upload the project zip, unzip, create venv, pip install -r requirements.txt
```

4. Create a systemd service pointing to `webapp/server.py`
5. Configure nginx to proxy port 80 to port 5050
6. Navigate to the Elastic IP in a browser

Estimated cost from $100 AWS credits: under $5 for a short-term demo deployment.

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy, pandas | Data generation and manipulation |
| scipy | Z-Score computation |
| scikit-learn | Isolation Forest model |
| matplotlib | Batch chart generation |
| flask | Web server and JSON API |
| mlflow | Experiment tracking and model registry |
| gradio | Interactive web UI for the Gradio app |
| joblib | Model serialisation |

---

## License

MIT
