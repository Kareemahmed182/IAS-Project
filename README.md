
#  Intelligent Adaptive System for Smart Home Environment Management

This project models and manages the indoor **temperature and humidity** of a virtual smart home using **machine learning** and adaptive control techniques.

##  Project Structure

```
IAS_PART2/
├── backend/
│   ├── models/               # LSTM and Random Forest models
│   ├── control/              # Environment control logic (ventilation & heaters)
│   ├── utils/                # Preprocessing, metrics, and plotting
├── data/                     # Raw CSV dataset
├── output/                   # Prediction CSVs and plots
├── control_simulation.py     # Applies heater/ventilation control logic
├── main.py                   # Entry point: runs full pipeline
├── README.md                 # This file
```

##  How to Run

### 1. Create a Virtual Environment
```bash
python -m venv my_env
```

### 2.  Activate Environment
```bash
# Windows
my_env\Scripts\activate
```

### 3.  Install Dependencies
```bash
pip install -r requirements.txt
```

(If `requirements.txt` is not included, install manually):
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow seaborn
```

### 4. ️ Run the Project
```bash
python main.py
```

##  Features

- **LSTM model** to predict indoor temperature and humidity every 15 minutes.
- **Random Forest** as a comparison baseline.
- **Control logic** simulating:
  - Mechanical ventilation (resets indoor to outdoor conditions).
  - Storage heaters (+0.5°C for 4h, max 2x/day).
- **Chronological data split** for realism.
- **MAE-optimized deep learning**.

##  Output

- `output/predictions_lstm.csv`: Raw LSTM predictions
- `output/adjusted_predictions.csv`: Control-adjusted environment
- `output/control_plot.png`: Timeline of control events

##  Evaluation

- Metrics: MAE, RMSE, R²
- LSTM outperforms RF in final version
- Data preprocessing and feature cleaning significantly improved results
