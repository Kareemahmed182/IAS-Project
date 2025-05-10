
#  Intelligent Adaptive System (IAS) for Virtual Home Environment Management

##  Project Overview

This project implements an intelligent adaptive system designed to predict and manage **indoor temperature and humidity** in a virtual home environment using machine learning techniques. The system leverages time-series modeling to make environmental control decisions using **mechanical ventilation** and **storage heaters**, aiming to optimize **user comfort**.

---

##  Project Structure

```plaintext
IAS-Project/
│
├── backend/
│   ├── control/              # Environmental control logic (heaters, ventilation)
│   ├── models/               # Machine learning models (LSTM, baselines)
│   └── utils/                # Helper functions for preprocessing, plotting, etc.
│
├── data/                     # Raw and processed datasets
├── my_env/                   # Optional virtual environment config
├── output/                   # CSV output predictions
│
├── control_simulation.py     # Script to apply control logic to predictions
├── main.py                   # Main pipeline: preprocessing, training, inference
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```

---

##  Features

- **Predictive Modeling**:
  - Uses **LSTM** for time-series prediction.
  - Benchmarked against baseline ML models (e.g., XGBoost, Linear Regression).

- **Control Logic**:
  - **Mechanical Ventilation**: Aligns indoor conditions with outdoor values.
  - **Storage Heaters**: Raises indoor temperature by 0.5°C per 15 minutes (limited to 2 uses/day).

- **Data Processing**:
  - Handles missing and noisy data.
  - Performs feature selection based on correlation analysis.

- **Output Generation**:
  - CSV file with 15-minute interval predictions for indoor temperature and humidity.

---

##  How to Run

###  Prerequisites

- Python 3.8 or higher
- Install required packages:

```bash
pip install -r requirements.txt
```

###  Running the System

1. Ensure the dataset is placed in the `data/` directory.
2. Execute the pipeline using:

```bash
python main.py
```

3. The output will be saved to:

```plaintext
output/predicted_indoor_conditions.csv
```

---

##  Output Format

Each row in the output CSV contains:

```plaintext
Timestamp, Predicted_Indoor_Temperature, Predicted_Indoor_Humidity
```

---

##  Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

These are calculated for both temperature and humidity. Optional plots can visualize prediction accuracy.

---

##  Adaptive System Logic

- **Mechanical Ventilation**:  
  Applies outdoor conditions to indoor readings when active. Effect lasts exactly 15 minutes.

- **Storage Heaters**:  
  Increase temperature by 0.5°C per 15 minutes, persist for 4 hours. Limited to 2 activations/day.  
  **Note**: Ventilation cancels heating effects during overlap.


