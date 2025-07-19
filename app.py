import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Burnout Predictor", layout="wide")
st.title(" Burnout Prediction App")

# Upload CSV
st.sidebar.header("Upload Employee Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if 'Attrition' not in df.columns:
        st.warning("Dataset must contain 'Attrition' column as target.")
    else:
        st.subheader(" Exploratory Data Analysis")
        st.write("Attrition Count:")
        st.bar_chart(df['Attrition'].value_counts())

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        st.write("Correlation Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), ax=ax, cmap='coolwarm')
        st.pyplot(fig)

        st.subheader("Burnout Prediction")
        df_model = df.copy()
        df_model = pd.get_dummies(df_model, drop_first=True)

        if 'Attrition_Yes' in df_model.columns:
            X = df_model.drop('Attrition_Yes', axis=1)
            y = df_model['Attrition_Yes']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.subheader(" Try Prediction with Custom Input")
            sample_input = {}
            for col in X.columns:
                sample_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

            input_df = pd.DataFrame([sample_input])
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0][1]

            st.success("Prediction: {} (Burnout Risk Score: {:.2f}%)".format(
                "YES" if pred == 1 else "NO", pred_proba * 100))

else:
    st.info("Upload a CSV file to get started")
