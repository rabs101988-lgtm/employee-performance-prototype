
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ADUSTECH Employee Performance Classification",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ADUSTECH-INSPIRED GREEN DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f4f7f5;
        color: #1f2933;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* ================= HEADER ================= */

    .adustech-header {
        background-color: #087f3f;
        color: white;
        padding: 18px 25px;
        border-radius: 0 0 10px 10px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
    }

    .adustech-logo {
        width: 68px;
        height: 68px;
        min-width: 68px;
        background-color: white;
        color: #087f3f;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 11px;
        font-weight: 800;
        border: 3px solid #dcefe4;
    }

    .adustech-name {
        font-size: 25px;
        font-weight: 800;
    }

    .adustech-subtitle {
        font-size: 15px;
        margin-top: 3px;
    }

    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] {
        background-color: #063d25;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-title {
        color: white !important;
        font-size: 20px;
        font-weight: 700;
        padding: 10px 5px 15px 5px;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 15px;
    }

    .sidebar-text {
        color: #dcefe4 !important;
        font-size: 13px;
        line-height: 1.6;
    }

    /* ================= PAGE TITLES ================= */

    .page-title {
        color: #075b2d;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .page-subtitle {
        color: #667085;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* ================= SECTION HEADINGS ================= */

    .section-heading {
        color: #075b2d;
        font-size: 21px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 14px;
        border-left: 5px solid #087f3f;
        padding-left: 10px;
    }

    /* ================= CARDS ================= */

    .info-card {
        background-color: white;
        border: 1px solid #dfe7e2;
        border-radius: 9px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .card-title {
        color: #075b2d;
        font-size: 18px;
        font-weight: 700;
    }

    /* ================= RESULT ================= */

    .result-card {
        background-color: #087f3f;
        color: white;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 15px 0 25px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    .result-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .result-value {
        font-size: 38px;
        font-weight: 850;
        margin-top: 8px;
    }

    /* ================= PROBABILITIES ================= */

    .prob-card {
        background-color: white;
        border: 1px solid #dfe7e2;
        border-left: 5px solid #087f3f;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 7px rgba(0,0,0,0.04);
    }

    .prob-title {
        color: #667085;
        font-size: 14px;
        font-weight: 600;
    }

    .prob-value {
        color: #075b2d;
        font-size: 27px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* ================= NOTICE ================= */

    .notice {
        background-color: #edf7f1;
        border-left: 5px solid #087f3f;
        padding: 15px 18px;
        border-radius: 6px;
        color: #344054;
        font-size: 14px;
        line-height: 1.6;
        margin: 18px 0;
    }

    /* ================= BUTTON ================= */

    .stButton > button {
        background-color: #087f3f;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 700;
        padding: 10px 20px;
    }

    .stButton > button:hover {
        background-color: #075b2d;
        color: white;
    }

    /* ================= FOOTER ================= */

    .footer {
        background-color: #063d25;
        color: white;
        padding: 16px;
        margin-top: 35px;
        border-radius: 7px;
        text-align: center;
        font-size: 13px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "employee_performance_decision_tree_pipeline.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Please make sure "
        "'employee_performance_decision_tree_pipeline.pkl' "
        "is inside the models folder."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as model_error:
    st.error("The saved model could not be loaded.")
    st.stop()


# ============================================================
# IDENTIFY MODEL PIPELINE COMPONENTS
# ============================================================

try:
    preprocessor = model.named_steps["preprocessor"]

    if "model" in model.named_steps:
        tree_model = model.named_steps["model"]
    elif "classifier" in model.named_steps:
        tree_model = model.named_steps["classifier"]
    else:
        st.error(
            "The saved pipeline does not contain the expected "
            "model/classifier step."
        )
        st.stop()

except Exception:
    st.error(
        "The saved model pipeline structure could not be read."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
  
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Navigation</div>',
        unsafe_allow_html=True
    )

    st.markdown("🏠  **Home**")
    st.markdown("👤  **Employee Classification**")
    st.markdown("📊  **Results**")
    st.markdown("🔍  **SHAP Explanation**")
    st.markdown("ℹ️  **About**")

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-text">

        <strong>Research Prototype</strong><br><br>

        MSc Data Science Research<br>
        ADUSTECH Application Context<br>
        IBM HR Analytics Dataset<br><br>

        Selected Model:<br>
        <strong>Decision Tree Classifier</strong>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="page-title">'
    'Employee Performance Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Classification of observed employee performance categories '
    'using the selected machine-learning model.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# RESEARCH NOTICE
# ============================================================

st.markdown(
    """
    <div class="notice">

    <strong>Research Prototype Notice:</strong><br>

    This application is a standalone MSc research prototype.
    The experimental model was developed using the fictional IBM HR Analytics
    Employee Attrition & Performance dataset. It does not use actual ADUSTECH
    employee records and should not be used as an autonomous employment
    decision-making system.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EMPLOYEE INPUT
# ============================================================

st.markdown(
    '<div class="section-heading">Employee Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=35
    )

    Attrition = st.selectbox(
        "Attrition",
        ["No", "Yes"]
    )

    BusinessTravel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    DailyRate = st.number_input(
        "Daily Rate",
        min_value=0,
        value=800
    )

    Department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    DistanceFromHome = st.number_input(
        "Distance From Home",
        min_value=0,
        value=5
    )

    Education = st.number_input(
        "Education",
        min_value=1,
        max_value=5,
        value=3
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

    EnvironmentSatisfaction = st.number_input(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    HourlyRate = st.number_input(
        "Hourly Rate",
        min_value=0,
        value=65
    )

    JobInvolvement = st.number_input(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3
    )

    JobLevel = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2
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

    JobSatisfaction = st.number_input(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000
    )

    MonthlyRate = st.number_input(
        "Monthly Rate",
        min_value=0,
        value=15000
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    NumCompaniesWorked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        value=2
    )

    OverTime = st.selectbox(
        "Over Time",
        ["No", "Yes"]
    )

    RelationshipSatisfaction = st.number_input(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    StockOptionLevel = st.number_input(
        "Stock Option Level",
        min_value=0,
        max_value=3,
        value=1
    )

    TotalWorkingYears = st.number_input(
        "Total Working Years",
        min_value=0,
        value=10
    )

    TrainingTimesLastYear = st.number_input(
        "Training Times Last Year",
        min_value=0,
        value=3
    )

    WorkLifeBalance = st.number_input(
        "Work Life Balance",
        min_value=1,
        max_value=4,
        value=3
    )

    YearsAtCompany = st.number_input(
        "Years At Company",
        min_value=0,
        value=5
    )

    YearsInCurrentRole = st.number_input(
        "Years In Current Role",
        min_value=0,
        value=3
    )


# ============================================================
# REMAINING VARIABLES
# ============================================================

col4, col5, col6 = st.columns(3)

with col4:

    YearsSinceLastPromotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        value=1
    )

with col5:

    YearsWithCurrManager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        value=2
    )

with col6:

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        Classification Categories
        </div>

        <p>
        The model classifies the observed performance category as:
        </p>

        <strong>Excellent</strong><br>
        <strong>Outstanding</strong>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [
        {
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
        }
    ]
)


# ============================================================
# CLASSIFICATION
# ============================================================

st.markdown(
    '<div class="section-heading">Performance Classification</div>',
    unsafe_allow_html=True
)

if st.button("CLASSIFY EMPLOYEE PERFORMANCE"):

    try:

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        excellent_probability = probabilities[0]
        outstanding_probability = probabilities[1]

        if prediction == 1:
            result = "OUTSTANDING"
        else:
            result = "EXCELLENT"


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Classified Performance Category
                </div>

                <div class="result-value">
                    {result}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.markdown(
            '<div class="section-heading">'
            'Classification Probabilities'
            '</div>',
            unsafe_allow_html=True
        )

        p1, p2 = st.columns(2)

        with p1:

            st.markdown(
                f"""
                <div class="prob-card">

                    <div class="prob-title">
                        Excellent
                    </div>

                    <div class="prob-value">
                        {excellent_probability:.2%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:

            st.markdown(
                f"""
                <div class="prob-card">

                    <div class="prob-title">
                        Outstanding
                    </div>

                    <div class="prob-value">
                        {outstanding_probability:.2%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILITY INTERPRETATION
        # ====================================================

        st.markdown(
            """
            <div class="notice">

            <strong>Interpretation:</strong><br>

            The displayed probabilities are model-generated estimates
            associated with the two observed performance categories.
            They should not be interpreted as certainty or as evidence
            of an employee's actual or future performance.

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown(
            '<div class="section-heading">'
            'SHAP Explanation'
            '</div>',
            unsafe_allow_html=True
        )

        try:

            transformed_input = preprocessor.transform(input_data)

            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.TreeExplainer(tree_model)

            shap_result = explainer(transformed_input)

            shap_values = shap_result.values


            # ------------------------------------------------
            # HANDLE DIFFERENT SHAP OUTPUT SHAPES
            # ------------------------------------------------

            if shap_values.ndim == 3:

                # Binary classification:
                # rows × features × classes

                local_values = shap_values[0, :, 1]

            elif shap_values.ndim == 2:

                local_values = shap_values[0]

            else:

                local_values = np.asarray(
                    shap_values
                ).reshape(-1)


            # ------------------------------------------------
            # CREATE SHAP DATAFRAME
            # ------------------------------------------------

            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": local_values
                }
            )

            shap_df["Absolute SHAP"] = (
                shap_df["SHAP Value"].abs()
            )

            shap_df = (
                shap_df
                .sort_values(
                    "Absolute SHAP",
                    ascending=False
                )
                .head(10)
            )


            # ------------------------------------------------
            # SHAP EXPLANATION NOTICE
            # ------------------------------------------------

            st.markdown(
                """
                <div class="info-card">

                <div class="card-title">
                Top Factors Contributing to the Classification
                </div>

                <p>
                SHAP values indicate how the model's input features
                contributed to the classification for this particular
                employee record. They describe model behaviour and
                should not be interpreted as causal effects.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # SHAP BAR CHART
            # ------------------------------------------------

            st.bar_chart(
                shap_df.set_index("Feature")["SHAP Value"]
            )


            # ------------------------------------------------
            # SHAP TABLE
            # ------------------------------------------------

            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


        except Exception:

            st.warning(
                "The SHAP explanation could not be generated "
                "for this classification."
            )


        # ====================================================
        # SUBMITTED INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-heading">'
            'Submitted Employee Information'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            input_data.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )


    except Exception as prediction_error:

        st.error(
            "The classification could not be completed. "
            "Please check the entered information and try again."
        )


# ============================================================
# PROTOTYPE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Prototype Information'
    '</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)


with info1:

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        Selected Model
        </div>

        <p>
        Decision Tree Classifier
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        Classification
        </div>

        <p>
        Excellent / Outstanding
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

        <div class="card-title">
        Explainability
        </div>

        <p>
        SHAP TreeExplainer
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <strong>MSc Research Prototype</strong>
    &nbsp; | &nbsp;
    ADUSTECH Application Context
    &nbsp; | &nbsp;
    Machine-Learning-Based Employee Performance Classification

    <br>

    IBM HR Analytics Employee Attrition & Performance Dataset

    </div>
    """,
    unsafe_allow_html=True
)
