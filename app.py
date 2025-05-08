import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.radio("Select Model", ["LSTM", "Random Forest"])
show_data = st.sidebar.checkbox("Show Raw Predictions")

# Title
st.title("🏡 Smart Home Environment Monitor")
st.markdown("An adaptive system using LSTM + RF to manage indoor conditions intelligently.")

# Load appropriate prediction file
if model_choice == "LSTM":
    preds = pd.read_csv("output/predictions_lstm.csv")
    mae_temp, mae_hum = "1.49", "2.86"
    r2_temp, r2_hum = "0.45", "0.62"
else:
    preds = pd.read_csv("output/predictions_rf.csv")
    mae_temp, mae_hum = "1.57", "4.44"
    r2_temp, r2_hum = "0.40", "0.27"

if show_data:
    st.dataframe(preds.head())

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Predictions", "🧠 Control Simulation"])

with tab1:
    st.metric("MAE - Temperature (°C)", mae_temp)
    st.metric("MAE - Humidity (%)", mae_hum)
    st.metric("R² - Temperature", r2_temp)
    st.metric("R² - Humidity", r2_hum)

with tab2:
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(preds["Actual_Temperature"], label="Actual", color="tomato")
    axs[0].plot(preds["Pred_Temperature"], label="Predicted", color="black")
    axs[0].set_title("Indoor Temperature")
    axs[1].plot(preds["Actual_Humidity"], label="Actual", color="teal")
    axs[1].plot(preds["Pred_Humidity"], label="Predicted", color="black")
    axs[1].set_title("Indoor Humidity")
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    controls = pd.read_csv("output/adjusted_predictions.csv")
    if "heater_status" in controls.columns and "ventilation_status" in controls.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(controls["heater_status"], label="Heater", color="orange")
        ax2.plot(controls["ventilation_status"], label="Ventilation", color="skyblue")
        ax2.set_title("Control Logic Over Time")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Control data not available.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by Kareem | Powered by Streamlit + LSTM + RF")
