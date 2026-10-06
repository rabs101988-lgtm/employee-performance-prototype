import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ADUSTECH Employee Performance Prototype",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f8f6;
    }

    /* ================= SIDEBAR ================= */

    [data-testid="stSidebar"] {
        background-color: #064d2d !important;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .sidebar-title {
        color: #ffffff !important;
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 25px;
    }

    .sidebar-item {
        color: #ffffff !important;
        font-size: 16px;
        font-weight: 700;
        padding: 10px 0;
    }

    .sidebar-section {
        color: #ffffff !important;
        font-size: 14px;
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .sidebar-info {
        color: #ffffff !important;
        font-size: 13px;
        line-height: 1.8;
    }

    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.30);
        margin: 20px 0;
    }

    /* ================= HEADER ================= */

    .adustech-header {
        background: linear-gradient(135deg, #006b3c, #0b8f4d);
        padding: 28px 35px;
        border-radius: 0 0 18px 18px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    .header-title {
        color: #ffffff !important;
        font-size: 30px;
        font-weight: 800;
        margin: 0;
    }

    .header-subtitle {
        color: #e9fff3 !important;
        font-size: 15px;
        margin-top: 8px;
        line-height: 1.6;
    }

    /* ================= SECTION TITLES ================= */

    .section-title {
        color: #075b35 !important;
        font-size: 23px;
        font-weight: 750;
        border-left: 5px solid #0b8f4d;
        padding-left: 12px;
        margin-top: 25px;
        margin-bottom: 18px;
    }

    /* ================= RESULT CARD ================= */

    .result-card {
        background: linear-gradient(135deg, #078743, #0b9b52);
        padding: 35px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.13);
    }

    .result-label {
        color: #ffffff !important;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .result-value {
        color: #ffffff !important;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    /* ================= PROBABILITY CARDS ================= */

    .prob-card {
        background: #ffffff;
        border-left: 5px solid #0b8f4d;
        border-radius: 12px;
        padding: 23px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }

    .prob-title {
        color: #14532d !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .prob-value {
        color: #0b8f4d !important;
        font-size: 29px;
        font-weight: 800;
    }

    /* ================= INFORMATION CARDS ================= */

    .info-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #d9e8df;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    .info-title {
        color: #075b35 !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-text {
        color: #374151 !important;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ================= RESEARCH NOTICE ================= */

    .research-notice {
        background: #ecfdf5;
        border: 1px solid #86efac;
        border-left: 5px solid #0b8f4d;
        border-radius: 10px;
        padding: 17px 20px;
        color: #14532d !important;
        font-size: 14px;
        line-height: 1.6;
        margin: 15px 0 25px 0;
    }

    /* ================= BUTTON ================= */

    div.stButton > button {
        background-color: #087f42 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
    }

    div.stButton > button:hover {
        background-color: #056b36 !important;
        color: #ffffff !important;
    }

    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        color: #6b7280 !important;
        font-size: 12px;
        padding: 25px 0 10px 0;
        margin-top: 35px;
        border-top: 1px solid #d9e8df;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(
    "models/employee_performance_decision_tree_pipeline.pkl"
)

if not MODEL_PATH.exists():
    st.error(
        "The saved model could not be found. Please ensure that "
        "models/employee_performance_decision_tree_pipeline.pkl "
        "exists in the repository."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load the saved model: {e}")
    st.stop()


# ============================================================
# GET PREPROCESSOR AND CLASSIFIER
# ============================================================

try:
    preprocessor = model.named_steps["preprocessor"]
except Exception:
    st.error(
        "The saved model does not contain the expected "
        "'preprocessor' step."
    )
    st.stop()

if "model" in model.named_steps:
    tree_model = model.named_steps["model"]
elif "classifier" in model.named_steps:
    tree_model = model.named_steps["classifier"]
else:
    st.error(
        "The saved pipeline does not contain a 'model' "
        "or 'classifier' step."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="adustech-header"><div class="header-title">Aliko Dangote University of Science and Technology</div><div class="header-subtitle">Employee Performance Classification Research Prototype<br>Wudil, Kano State, Nigeria</div></div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">Navigation</div><div class="sidebar-item">🏠 Home</div><div class="sidebar-item">👤 Employee Classification</div><div class="sidebar-item">📊 Results</div><div class="sidebar-item">🔍 SHAP Explanation</div><div class="sidebar-item">ℹ️ About</div><hr class="sidebar-divider"><div class="sidebar-section">Research Prototype</div><div class="sidebar-info">MSc Data Science Research<br>ADUSTECH Application Context<br>IBM HR Analytics Dataset</div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RESEARCH NOTICE
# ============================================================

st.markdown(
    """
    <div class="research-notice"><strong>Research Prototype Notice:</strong> This application is an MSc research prototype for classifying observed employee performance categories using the IBM HR Analytics Employee Attrition &amp; Performance dataset. The dataset is fictional and is not an ADUSTECH employee database. The application should therefore not be interpreted as an operational ADUSTECH human-resource decision-making system.</div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="section-title">Employee Performance Classification</div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter the available employee attributes below to obtain "
    "a classification of the observed performance category."
)


# ============================================================
# EMPLOYEE INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-title">Employee Information</div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30,
        step=1
    )

    DailyRate = st.number_input(
        "Daily Rate",
        min_value=0,
        max_value=2000,
        value=800,
        step=1
    )

    DistanceFromHome = st.number_input(
        "Distance From Home",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    Education = st.number_input(
        "Education",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    EnvironmentSatisfaction = st.number_input(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3,
        step=1
    )

    HourlyRate = st.number_input(
        "Hourly Rate",
        min_value=0,
        max_value=200,
        value=65,
        step=1
    )

    JobInvolvement = st.number_input(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3,
        step=1
    )

    JobLevel = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    JobSatisfaction = st.number_input(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3,
        step=1
    )


with col2:

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0,
        max_value=200000,
        value=5000,
        step=100
    )

    MonthlyRate = st.number_input(
        "Monthly Rate",
        min_value=0,
        max_value=30000,
        value=14000,
        step=100
    )

    NumCompaniesWorked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    RelationshipSatisfaction = st.number_input(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3,
        step=1
    )

    StockOptionLevel = st.number_input(
        "Stock Option Level",
        min_value=0,
        max_value=3,
        value=1,
        step=1
    )

    TotalWorkingYears = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=50,
        value=10,
        step=1
    )

    TrainingTimesLastYear = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )

    WorkLifeBalance = st.number_input(
        "Work Life Balance",
        min_value=1,
        max_value=4,
        value=3,
        step=1
    )

    YearsAtCompany = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )


with col3:

    YearsInCurrentRole = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )

    YearsSinceLastPromotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    YearsWithCurrManager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    Attrition = st.selectbox(
        "Attrition",
        ["No", "Yes"]
    )

    BusinessTravel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    Department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    EducationField = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    JobRole = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    OverTime = st.selectbox(
        "Over Time",
        ["No", "Yes"]
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [{
        "Age": Age,
        "Attrition": Attrition,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }]
)


# ============================================================
# CLASSIFICATION BUTTON
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

predict_button = st.button(
    "🔎 Classify Employee Performance",
    use_container_width=True
)


# ============================================================
# CLASSIFICATION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
        else:
            probabilities = None

        if int(prediction) == 0:
            prediction_label = "Excellent"
        else:
            prediction_label = "Outstanding"


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        excellent_prob = 0.0
        outstanding_prob = 0.0

        if probabilities is not None:

            classes = list(model.classes_)

            if 0 in classes:
                excellent_prob = probabilities[
                    classes.index(0)
                ]

            if 1 in classes:
                outstanding_prob = probabilities[
                    classes.index(1)
                ]


        # ====================================================
        # CLASSIFICATION RESULT
        # ====================================================

        st.markdown(
            """
            <div class="section-title">Classification Result</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card"><div class="result-label">Classified Performance Category</div><div class="result-value">{prediction_label.upper()}</div></div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.markdown(
            """
            <div class="section-title">Classification Probabilities</div>
            """,
            unsafe_allow_html=True
        )

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:

            st.markdown(
                f"""
                <div class="prob-card"><div class="prob-title">Excellent</div><div class="prob-value">{excellent_prob:.2%}</div></div>
                """,
                unsafe_allow_html=True
            )

        with probability_col2:

            st.markdown(
                f"""
                <div class="prob-card"><div class="prob-title">Outstanding</div><div class="prob-value">{outstanding_prob:.2%}</div></div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.markdown(
            """
            <div class="info-card"><div class="info-title">Interpretation</div><div class="info-text">The displayed probabilities are model estimates for the two observed performance categories represented in the experimental dataset. They should not be interpreted as guarantees or as evidence of an employee's true future performance.</div></div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown(
            """
            <div class="section-title">SHAP Explanation</div>
            """,
            unsafe_allow_html=True
        )

        try:

            transformed_input = preprocessor.transform(
                input_data
            )

            if hasattr(
                transformed_input,
                "toarray"
            ):
                transformed_dense = (
                    transformed_input.toarray()
                )
            else:
                transformed_dense = np.asarray(
                    transformed_input
                )

            try:
                feature_names = (
                    preprocessor
                    .get_feature_names_out()
                )
            except Exception:
                feature_names = [
                    f"Feature {i + 1}"
                    for i in range(
                        transformed_dense.shape[1]
                    )
                ]

            explainer = shap.TreeExplainer(
                tree_model
            )

            shap_values = explainer.shap_values(
                transformed_dense
            )

            # ------------------------------------------------
            # SHAP output handling
            # ------------------------------------------------

            if isinstance(
                shap_values,
                list
            ):

                if len(shap_values) > 1:
                    values = np.asarray(
                        shap_values[1]
                    )[0]
                else:
                    values = np.asarray(
                        shap_values[0]
                    )[0]

            else:

                values_array = np.asarray(
                    shap_values
                )

                if values_array.ndim == 3:

                    if values_array.shape[-1] > 1:
                        values = values_array[
                            0, :, 1
                        ]
                    else:
                        values = values_array[
                            0, :, 0
                        ]

                elif values_array.ndim == 2:

                    values = values_array[0]

                else:

                    values = values_array.flatten()


            values = np.asarray(
                values
            ).flatten()


            # ------------------------------------------------
            # Match dimensions
            # ------------------------------------------------

            if len(values) != len(
                feature_names
            ):

                minimum = min(
                    len(values),
                    len(feature_names)
                )

                values = values[:minimum]
                feature_names = (
                    feature_names[:minimum]
                )


            # ------------------------------------------------
            # SHAP dataframe
            # ------------------------------------------------

            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": values,
                    "Absolute SHAP Value":
                        np.abs(values)
                }
            )

            shap_df = (
                shap_df
                .sort_values(
                    "Absolute SHAP Value",
                    ascending=False
                )
                .head(15)
            )


            st.write(
                "The following features had the largest "
                "absolute SHAP contributions for this "
                "classification."
            )

            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ].reset_index(
                    drop=True
                ),
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # SHAP BAR CHART
            # ------------------------------------------------

            chart_df = (
                shap_df
                .sort_values(
                    "Absolute SHAP Value",
                    ascending=True
                )
            )

            st.bar_chart(
                chart_df.set_index(
                    "Feature"
                )[
                    "Absolute SHAP Value"
                ]
            )

            st.caption(
                "SHAP values describe the contribution of "
                "model features to the classification for "
                "this input. They indicate model association "
                "and should not be interpreted as causal effects."
            )


        except Exception as shap_error:

            st.warning(
                "The classification was completed, but "
                "the SHAP explanation could not be generated "
                f"for this input. Details: {shap_error}"
            )


        # ====================================================
        # SUBMITTED INFORMATION
        # ====================================================

        st.markdown(
            """
            <div class="section-title">Submitted Employee Information</div>
            """,
            unsafe_allow_html=True
        )

        display_data = (
            input_data
            .T
            .reset_index()
        )

        display_data.columns = [
            "Variable",
            "Value"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # PROTOTYPE INFORMATION
        # ====================================================

        st.markdown(
            """
            <div class="section-title">Prototype Information</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card"><div class="info-title">Selected Model</div><div class="info-text">Decision Tree classifier selected using the highest mean five-fold stratified cross-validation Macro F1 among the four evaluated algorithms.</div></div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card"><div class="info-title">Experimental Dataset</div><div class="info-text">IBM HR Analytics Employee Attrition &amp; Performance dataset. The dataset is a public fictional dataset used for the experimental evaluation.</div></div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card"><div class="info-title">Application Context</div><div class="info-text">Aliko Dangote University of Science and Technology (ADUSTECH), Wudil, Kano State. ADUSTECH is the intended application context and is not the source of the experimental employee records.</div></div>
            """,
            unsafe_allow_html=True
        )


    except Exception as error:

        st.error(
            "An error occurred while processing the "
            "employee information."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">MSc Data Science Research Prototype<br>Machine-Learning-Based Employee Performance Classification<br>Aliko Dangote University of Science and Technology (ADUSTECH) Application Context</div>
    """,
    unsafe_allow_html=True
)
