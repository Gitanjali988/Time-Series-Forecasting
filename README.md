# ⏳ Time Series Forecasting of Electricity & Truck Production

This repository presents a comprehensive time series analysis project aimed at forecasting electricity production and total truck manufacturing using statistical modeling techniques like ARIMA and SARIMA.

---

## 📊 Objective

To analyze seasonal and non-seasonal datasets using time series methods:
- Detect stationarity
- Plot ACF and PACF to guide model selection
- Apply ARIMA/SARIMA models
- Forecast and visualize future values
- Evaluate residuals and model performance

---

## 📄 Datasets Used

- **📈 Seasonal**: [Total Truck Production](https://fred.stlouisfed.org/series/G17MVSFTTRUCKS) (Jan 1996 – Jun 2022)
- **⚡ Non-Seasonal**: [Electricity Production](https://www.kaggle.com/datasets/shenba/time-series-datasets?resource=download&select=Electric_Production.csv) (Jan 1985 – Jan 2018)

---

## Visualization

![image](https://github.com/user-attachments/assets/ca7c06a2-858c-476a-98e2-b34df484f09e)

---
## 🧠 Methodology

1. **Time Series Visualization**
2. **Stationarity Check** (ADF Test)
3. **ACF & PACF Plotting** for lag correlation analysis
4. **Model Fitting**: AR, MA, ARIMA, SARIMA
5. **Forecasting** future values
6. **Residual Diagnostics** and performance evaluation

---

## 📉 Key Findings

- **Electricity production** data was **non-stationary** and required differencing before modeling.
- **Truck production** was **stationary**, suitable for direct modeling.
- ARIMA/SARIMA models were selected using AIC scores and lag structure.
- Forecasts showed strong alignment with actual values, especially for seasonal data.

---

## 📓 Report

📄 [Download the Full Report (PDF)](./report/Time%20Series%20Analysis%20Report.pdf)

---

## 🔧 Tools & Libraries

- Python (pandas, statsmodels, matplotlib, seaborn)
- ARIMA, SARIMA modeling
- Jupyter Notebooks

---

## 📌 Author

**Fnu Gitanjali**  
*Data Engineer* 
📍 *Seattle, WA*  
📧 gitanjali.gitu72@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/gitanjali-fnu/) 

---

## ⚖️ License

© 2025 Fnu Gitanjali – All rights reserved.  
No part of this repository may be copied or used without explicit permission.  


