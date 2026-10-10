# employee-performance-prototype
MSc research prototype for employee performance-rating classification
# Employee Performance-Rating Classification Prototype

## 1. Project Overview

This repository contains the software prototype and supporting materials developed for an MSc research study on the classification of observed employee performance ratings using machine-learning techniques.

The study compares four classification algorithms: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM).

## 2. Research Context

The intended institutional application context is Aliko Dangote University of Science and Technology (ADUSTECH), Wudil, Kano State, Nigeria. However, the experiments use a publicly available IBM HR Analytics dataset rather than actual ADUSTECH employee records.

Consequently, the experimental results describe the performance of the models on the selected dataset and should not be interpreted as empirical findings about ADUSTECH employees.

## 3. Dataset

The study uses the IBM HR Analytics Employee Attrition & Performance dataset obtained from Kaggle.

**Dataset source:** [Insert the exact Kaggle dataset URL used in this study.]

The dataset is used for methodological experimentation. Its original variables and limitations should be considered when interpreting the results.

## 4. Classification Algorithms

The study examines the following algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## 5. Experimental Evaluation

The experimental methodology uses a stratified training and testing split and five-fold stratified cross-validation.

Macro F1-score is the primary evaluation metric. Additional evaluation measures include accuracy, precision, recall, weighted F1-score, and confusion matrices, where applicable.

## 6. Prototype

The repository includes a Streamlit application (`app.py`) and the associated dependency specification (`requirements.txt`). The `models/` directory contains the model-related files provided with the project.

The prototype is intended for research demonstration and is not a production human-resources decision-making system.

## 7. Installation and Execution

1. Clone or download this repository.
2. Install the required Python packages using the provided dependency file.
3. Verify that the required model files are present.
4. Start the Streamlit application using:

   `streamlit run app.py`

The application requires a compatible Python environment and all model files referenced by the source code.

## 8. Reproducibility and Limitations

The repository supports transparency by documenting the prototype and making its available implementation files accessible for inspection.

Reproduction of the experimental results depends on the availability of the appropriate dataset, model files, software dependencies, and experimental configuration. The findings are limited to the dataset and methodology used in this study.

## 9. Academic Use

This repository accompanies an MSc research project. It is intended to support academic review and methodological transparency. The prototype should not be used as the sole basis for employment, promotion, disciplinary, or other consequential human-resources decisions.
