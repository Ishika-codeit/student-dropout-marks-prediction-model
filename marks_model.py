import pandas as pd
import joblib

# Load trained objects
marks_model = joblib.load("marks_model.pkl")
encoders = joblib.load("marks_encoders.pkl")
scaler = joblib.load("marks_scaler.pkl")

# Load training columns
df_train = pd.read_csv("cleaned_dataset.csv")
X = df_train.drop(columns=["marks_range", "dropout"], errors="ignore")

def predict_marks(student_dict, columns):
    df_new = pd.DataFrame([student_dict])

    # Fill missing columns
    for col in columns:
        if col not in df_new.columns:
            df_new[col] = df_train[col].mode()[0] if col in df_train.columns else "unknown"

    df_new = df_new[columns]

    # Encode categorical
    for col in df_new.select_dtypes(include="object").columns:
        if col in encoders:
            le = encoders[col]
            df_new[col] = df_new[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    df_scaled = scaler.transform(df_new)
    return marks_model.predict(df_scaled)[0]
