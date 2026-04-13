# AI Fraud Detection System

Detects fraudulent credit card transactions using Machine Learning (Random Forest).

## Tech Stack
Python, Flask, Scikit-learn, SQLite, JWT, bcrypt

## Project Structure
```
├── app.py              # Server + API + Database (all-in-one)
├── train_model.py      # Train ML model (one-time)
├── fraud_model.pkl     # Trained model
├── requirements.txt
├── .gitignore
└── static/
    ├── style.css        # All styles
    ├── index.html       # Landing page
    ├── login.html       # Login page
    └── account-balance.html  # Dashboard
```

## Run
```
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000 — Login: `admin` / `1234`

## Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud