# 🛠️ Predictive Maintenance API & Dashboard

A production-grade machine learning system that predicts industrial equipment failure in real-time.

## 🌟 Features
- **Accurate Predictions:** Random Forest model achieving **96.5% accuracy**.
- **Data Resilience:** Custom cleaning pipeline that handles "ERR" strings and missing sensor data.
- **RESTful API:** High-performance inference engine built with **FastAPI**.
- **Audit Logging:** Every prediction is logged to a **SQLite** database for monitoring.
- **Live Dashboard:** Interactive HTML frontend for instant failure visualization.

## 🚀 Quick Start
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Train Model:** `python train_model.py`
3. **Launch API:** `uvicorn main:app --reload`
4. **View Dashboard:** Open `index.html` in your browser.
