import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, r2_score
)

# =========================
# 🎨 BACKGROUND STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🎓 AI Student Performance Prediction System")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Student_Performance.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# =========================
# ADD INTERNET USAGE (SIMULATED)
# =========================
np.random.seed(42)
df["internet_usage"] = np.random.randint(0, 10, size=len(df))

# =========================
# FEATURE ENGINEERING
# =========================
df["overall_score"] = (
    df["math_score"] +
    df["science_score"] +
    df["english_score"]
) / 3

df["grade"] = df["overall_score"].apply(
    lambda x: "A" if x >= 70 else
              "B" if x >= 60 else
              "C" if x >= 50 else "D"
)

# Encode grades for R2
grade_mapping = {"A": 4, "B": 3, "C": 2, "D": 1}
reverse_mapping = {v: k for k, v in grade_mapping.items()}

df["target"] = df["grade"].map(grade_mapping)

features = [
    "age",
    "study_hours",
    "attendance_percentage",
    "math_score",
    "science_score",
    "english_score",
    "internet_usage"
]

X = df[features]
y = df["target"]

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = train_model(X_train, y_train)

y_pred_encoded = model.predict(X_test)

# Convert to labels
y_pred = [reverse_mapping[i] for i in y_pred_encoded]
y_test_labels = [reverse_mapping[i] for i in y_test]

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test_labels, y_pred) * 100
precision = precision_score(y_test_labels, y_pred, average="weighted") * 100
recall = recall_score(y_test_labels, y_pred, average="weighted") * 100
f1 = f1_score(y_test_labels, y_pred, average="weighted") * 100

r2 = r2_score(y_test, y_pred_encoded)

cm = confusion_matrix(y_test_labels, y_pred, labels=["A","B","C","D"])

# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio("Navigation", [
    "🎯 Predict",
    "📊 Metrics",
    "📈 Graphs + Recommendations"
])

# =========================
# PAGE 1 - PREDICT
# =========================
if page == "🎯 Predict":

    st.header("Enter Student Details")

    age = st.number_input("Age", 10, 30, 18)
    study_hours = st.number_input("Study Hours", 0, 10, 2)
    attendance = st.number_input("Attendance %", 0, 100, 75)

    math = st.number_input("Math Score", 0, 100, 50)
    science = st.number_input("Science Score", 0, 100, 50)
    english = st.number_input("English Score", 0, 100, 50)

    internet_usage = st.slider("Internet Usage (hours/day)", 0, 10, 3)

    if st.button("Predict"):

        input_df = pd.DataFrame([[ 
            age, study_hours, attendance,
            math, science, english,
            internet_usage
        ]], columns=features)

        pred_encoded = model.predict(input_df)[0]
        pred_grade = reverse_mapping[pred_encoded]

        proba = model.predict_proba(input_df)
        confidence = np.max(proba) * 100

        pred_score = (math + science + english) / 3

        st.session_state["grade"] = pred_grade
        st.session_state["score"] = pred_score
        st.session_state["confidence"] = confidence

        st.success("Prediction Completed! Go to Metrics Page")

# =========================
# PAGE 2 - METRICS
# =========================
elif page == "📊 Metrics":

    st.header("📊 Model Performance")

    st.write(f"Accuracy: {accuracy:.2f}%")
    st.write(f"Precision: {precision:.2f}%")
    st.write(f"Recall: {recall:.2f}%")
    st.write(f"F1 Score: {f1:.2f}%")
    st.write(f"R² Score: {r2:.2f}")

    if "confidence" in st.session_state:
        st.write(f"Confidence: {st.session_state['confidence']:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["A","B","C","D"],
                yticklabels=["A","B","C","D"],
                cmap="Blues", ax=ax)

    st.pyplot(fig)

    if "grade" in st.session_state:
        st.subheader("🎯 Your Prediction")
        st.success(f"Grade: {st.session_state['grade']}")
        st.info(f"Overall Score: {st.session_state['score']:.2f}")

# =========================
# PAGE 3 - GRAPHS
# =========================
elif page == "📈 Graphs + Recommendations":

    st.header("📈 Insights Dashboard")

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Feature Importance
    ax[0,0].barh(features, model.feature_importances_)
    ax[0,0].set_title("Feature Importance")

    # Score Distribution
    sns.histplot(df["overall_score"], kde=True, ax=ax[0,1])
    ax[0,1].set_title("Score Distribution")

    # Correlation
    sns.heatmap(df[features + ["overall_score"]].corr(),
                cmap="coolwarm", ax=ax[1,0])
    ax[1,0].set_title("Correlation Heatmap")

    # Study Hours vs Score
    ax[1,1].scatter(df["study_hours"], df["overall_score"])
    ax[1,1].set_title("Study Hours vs Score")

    st.pyplot(fig)

    st.subheader("💡 Recommendations")

    st.write("📌 Study at least 2–3 hours daily")
    st.write("📌 Improve attendance above 85%")
    st.write("📌 Reduce excessive internet usage")
    st.write("📌 Focus on weak subjects")