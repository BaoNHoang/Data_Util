# 🧠 Patient Data Cleaner & Scaler Utility

This repository provides a simple and modular Python utility for **loading**, **cleaning**, and **scaling** patient health data for machine learning tasks.

It includes:
- **`data_utils.py`** — functions for reading, cleaning, and scaling CSV datasets  
- **`patients.csv`** — sample dataset of patients with health metrics  
- **`main.py`** — example script demonstrating how to load and process the data

---

## 🚀 Features

✅ Automatically cleans messy CSV data  
✅ Drops missing or empty rows and columns  
✅ Keeps only numeric columns (for ML models)  
✅ Scales data using `StandardScaler`  
✅ Optionally balances classes using `RandomOverSampler`  
✅ Saves a cleaned, ready-to-use dataset

---

## 📂 Project Structure
```
📁 your-project/
├── data_utils.py # CSV loader + scaler functions
├── main.py # main script to run the process
├── patients.csv # example dataset
└── README.md # this file
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/patient-data-utils.git
cd patient-data-utils
python -m venv venv
```
### 2️⃣ Create a Virtual Environment (recommended)
```
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows
```
### 3️⃣ Install Dependencies
```
pip install pandas scikit-learn imbalanced-learn
```
