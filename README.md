# 🧠 Term Deposit Prediction - ML App (FastAPI + Sklearn)

This project is a **Machine Learning-based prediction system** built using **FastAPI** that predicts whether a bank client will subscribe to a **term deposit**. The prediction is based on client demographic, financial, and marketing campaign data.

---

## 📊 Dataset

The dataset used is a version of the **Bank Marketing Dataset**, which includes features like:

- **Age, Job, Marital Status, Education**
- **Default, Housing Loan, Personal Loan**
- **Contact Type, Campaign Duration**
- **Previous Campaign Info**

📄 Detailed preprocessing, EDA, model selection, and training are provided in [`notebook_v1.ipynb`](Notebook_v1.ipynb).

---

## ✅ Features

- 🔍 **EDA + Preprocessing Pipeline**
  - OneHotEncoding (categorical)
  - Imputation for missing values
  - StandardScaler (numerical)

- 🤖 **ML Models Tried**
  - Random Forest
  - XGBoost
  - Gradient Boosting

- 🌐 **FastAPI UI**
  - Jinja2 HTML form for user input
  - Auto preprocessing and prediction
  - Displays prediction result with friendly UI

- 🚫 **No Docker Required** (Docker optional)
  - Easily run locally with `uvicorn`

---

## 🛠 Tech Stack

| Layer        | Tools & Libraries                     |
|--------------|----------------------------------------|
| Backend API  | FastAPI, Pydantic, Uvicorn            |
| ML/EDA       | Scikit-learn, XGBoost, Pandas, NumPy  |
| Frontend     | HTML (Jinja2 templates)               |
| Visualization| Matplotlib, Seaborn                   |
| Dev Environment | Python 3.8+                        |

---

## 📂 Project Structure

