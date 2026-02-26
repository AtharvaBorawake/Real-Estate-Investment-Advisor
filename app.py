import streamlit as st
import joblib
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Real Estate ML", layout="wide")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    clf_model = joblib.load("models/classification_model.pkl")
    reg_model = joblib.load("models/regression_model.pkl")
    return clf_model, reg_model

clf_model, reg_model = load_models()

@st.cache_data
def load_data():
    return pd.read_csv("data/df_clean.csv")

df_clean = load_data()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Investment Prediction","Data Insights", "Feature Importance","Model Information","About Creator"]
)

if page == "Home":
    st.title("🏡 Real Estate Investment Advisor")
    
    st.markdown("""
    ### 📌 Overview
    This application helps users make **data-driven real estate investment decisions** by:

    - Predicting whether a property is a **good investment**
    - Forecasting **future property prices**
    - Explaining **key factors influencing property value**

    ### 🔧 Technologies Used
    - Machine Learning (Classification & Regression)
    - XGBoost, Random Forest, Gradient Boosting
    - MLflow for experiment tracking
    - Streamlit for deployment

    
    """)
elif page == "Data Insights":
    st.title("📊 Data Insights & Exploratory Analysis")

    st.markdown("""
    This section presents key exploratory insights derived from the cleaned real estate dataset.
    These visualizations help understand pricing behavior, feature relationships, and property-type differences.
    """)

    # 1️⃣ Price per SqFt Distribution
    st.subheader("📈 Distribution of Price per SqFt")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        df_clean["Price_per_SqFt"],
        bins=100,
        kde=True,
        ax=ax
    )
    ax.set_xlabel("Price per SqFt")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.caption(
        "Most properties are priced below ₹20,000 per SqFt, "
        "with frequency decreasing as price increases, indicating right-skewed distribution."
    )

    st.markdown("---")

    # 2️⃣ Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap of Numerical Features")

    num_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        df_clean[num_cols].corr(),
        cmap="coolwarm",
        annot=False,
        ax=ax
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.caption(
        "The heatmap highlights relationships between numerical features. "
        "Property size and price show positive correlation, while age tends to negatively impact value."
    )

    st.markdown("---")

    # 3️⃣ Price per SqFt by Property Type
    st.subheader("🏠 Price per SqFt by Property Type")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        x="Property_Type",
        y="Price_per_SqFt",
        data=df_clean,
        inner="quartile",
        ax=ax
    )
    ax.set_xlabel("Property Type")
    ax.set_ylabel("Price per SqFt")
    st.pyplot(fig)

    st.caption(
        "Violin plots show similar median pricing across property types, "
        "with high-price outliers present in all categories."
    )
    
    st.markdown("### Distribution of Property Prices (Lakhs)")
    fig, ax = plt.subplots()
    sns.histplot(df_clean["Price_in_Lakhs"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Distribution of Property Sizes (SqFt)")
    fig, ax = plt.subplots()
    sns.histplot(df_clean["Size_in_SqFt"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Average Price per SqFt by City (Top 10)")
    top_cities = (
        df_clean.groupby("City")["Price_per_SqFt"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    top_cities.plot(kind="bar", ax=ax)
    ax.set_ylabel("Price per SqFt")
    st.pyplot(fig)

    st.markdown("### Correlation Between Size and Price")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df_clean["Size_in_SqFt"],
        y=df_clean["Price_in_Lakhs"],
        alpha=0.3,
        ax=ax
    )
    st.pyplot(fig)
    
elif page == "Investment Prediction":

    st.header("📊 Enter Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        bhk = st.number_input("BHK", 1, 10, 2)
        size = st.number_input("Size in SqFt", 300, 10000, 1200)
        age = st.number_input("Property Age", 0, 50, 5)
        price_sqft = st.number_input("Price per SqFt", 1000, 50000, 8000)

    with col2:
        amenities = st.multiselect(
            "Select Amenities",
            ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
        )

    with col3:
        furnish_status = st.selectbox(
            "Furnishing",
            ["Unfurnished", "Semi-furnished", "Furnished"]
        )

        transport_level = st.selectbox(
            "Transport Connectivity",
            ["Low", "Medium", "High"]
        )

        locality_score = st.slider("Locality Price Score", 0.0, 1.0, 0.5) #demo for now
        city_score = st.slider("City Price Score", 0.0, 1.0, 0.5)   #demo for now 

    # ---- Feature Engineering (same as notebook) ----
    amenity_count = len(amenities)
    playground = int("Playground" in amenities)
    gym = int("Gym" in amenities)
    garden = int("Garden" in amenities)
    pool = int("Pool" in amenities)
    clubhouse = int("Clubhouse" in amenities)

    furnish_map = {
        "Unfurnished": 0,
        "Semi-furnished": 1,
        "Furnished": 2
    }

    transport_map = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    furnish_score = furnish_map[furnish_status]
    transport_score = transport_map[transport_level]

    input_data = [[
        bhk,
        size,
        age,
        price_sqft,
        amenity_count,
        playground,
        gym,
        garden,
        pool,
        clubhouse,
        furnish_score,
        transport_score,
        locality_score,
        city_score
    ]]

    # ---- Prediction ----
    if st.button("🔮 Predict"):
        invest_pred = clf_model.predict(input_data)[0]
        invest_prob = clf_model.predict_proba(input_data)[0][1]
        future_price = reg_model.predict(input_data)[0]

        st.subheader("📈 Results")

        if invest_pred == 1:
            st.success(f"✅ Good Investment (Confidence: {invest_prob*100:.2f}%)")
        else:
            st.error(f"❌ Risky Investment (Confidence: {(1-invest_prob)*100:.2f}%)")

        st.info(f"💰 Predicted Future Price (5 Years): ₹ {future_price:.2f} Lakhs") #Not working as expected will be updated soon
        

elif page == "Feature Importance":
    st.title("🔍 Feature Importance")

    st.markdown("""
    This section displays the most influential features used by the
    **XGBoost Classification Model** to evaluate investment quality.
    """)

    try:
        # Get XGBoost booster
        booster = clf_model.get_booster()

        # Get feature importance (gain-based)
        importance_dict = booster.get_score(importance_type="gain")

        if len(importance_dict) == 0:
            st.warning("No feature importance data available.")
        else:
            # Convert directly to DataFrame (NO index parsing)
            importance_df = (
                pd.DataFrame.from_dict(
                    importance_dict,
                    orient="index",
                    columns=["Importance"]
                )
                .reset_index()
                .rename(columns={"index": "Feature"})
                .sort_values("Importance", ascending=False)
            )

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=importance_df,
                x="Importance",
                y="Feature",
                ax=ax
            )
            ax.set_title("Top Influential Features (XGBoost Classifier)")
            st.pyplot(fig)

            st.caption(
                "Higher importance indicates stronger influence on the investment decision."
            )

    except Exception as e:
        st.error("Feature importance could not be displayed.")
        st.code(str(e))    
    

elif page == "Model Information":
    st.title("🧠 Model & Experiment Tracking")

    st.markdown("""
    ### Models Used
    **Classification**
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost (Final)

    **Regression**
    - Linear Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Extra Trees Regressor
    - XGBoost Regressor (Final)

    ### Experiment Tracking
    - MLflow used to track parameters, metrics, and artifacts
    - Best-performing models registered in MLflow Model Registry
    """)
  
elif page == "About Creator":

    st.header("👨‍💻 About the Creator")

    st.markdown("""
    **Name:** Atharva Borawake 
    **Project:** Real Estate Investment Advisor  
    **Domain:** Machine Learning & Data Science  

    ### 🔧 Skills Demonstrated
    - Python & Pandas
    - Exploratory Data Analysis (EDA)
    - Feature Engineering
    - Classification & Regression Models
    - XGBoost, Random Forest, Ensemble Learning
    - MLflow (Experiment Tracking & Model Registry)
    - Streamlit Application Development

    ### 🎯 Project Objective
    To help real-estate investors make **data-driven decisions** by:
    - Predicting future property value
    - Identifying profitable investments
    - Understanding key influencing factors

    ### 📌 Disclaimer
    This application is built for **educational and analytical purposes only** and should not be considered financial advice.
    """)

    st.success("Thank you for exploring this project 🙌")