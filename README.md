# Heart Failure Prediction Using Machine Learning

**Course**: ENCS5341 - Machine Learning and Data Science  
**Institution**: Electrical and Computer Engineering Department, Birzeit University

* * *

## Table of Contents

*   [Project Overview](#project-overview)
*   [Dataset Description](#dataset-description)
*   [Models and Methods](#models-and-methods)
*   [Results](#results)
*   [Conclusion](#conclusion)
*   [Installation and Usage](#installation-and-usage)
*   [Repository Structure](#repository-structure)
*   [Contributors](#contributors)
*   [License](#license)
*   [Acknowledgments](#acknowledgments)

* * *

## Project Overview

Cardiovascular diseases (CVDs) are the leading cause of death globally, claiming approximately 17.9 million lives each year. Early detection and management are crucial to reduce mortality rates. This project explores the application of machine learning techniques to predict the likelihood of heart failure in patients using clinical and demographic data.

The primary objectives of this project are:
*   To implement and evaluate different machine learning models for predicting heart failure.
*   To perform exploratory data analysis (EDA) to understand the dataset.
*   To tune hyperparameters for optimal model performance.
*   To analyze the performance of the best model and interpret the results.

## Dataset Description

The dataset used in this project is the **Heart Failure Prediction Dataset** obtained from Kaggle. It contains **918** patient records with **12** attributes, including both numerical and categorical features.

### Features:
*   `Age`: Age of the patient (years)
*   `Sex`: Sex of the patient (`M`, `F`)
*   `ChestPainType`: Type of chest pain experienced
*   `RestingBP`: Resting blood pressure (mm Hg)
*   `Cholesterol`: Serum cholesterol (mm/dl)
*   `FastingBS`: Fasting blood sugar (`1` if > 120 mg/dl, else `0`)
*   `RestingECG`: Resting electrocardiogram results
*   `MaxHR`: Maximum heart rate achieved
*   `ExerciseAngina`: Exercise-induced angina (`Y`, `N`)
*   `Oldpeak`: ST depression induced by exercise relative to rest
*   `ST_Slope`: The slope of the peak exercise ST segment
*   `HeartDisease`: Output class (`1` for presence, `0` for absence)

### Data Preprocessing and EDA:
*   **Handling Missing Values**: The dataset had no missing values.
*   **Outlier Detection**: Identified and treated outliers using the Interquartile Range (IQR) method.
*   **Encoding Categorical Variables**: Converted categorical variables to numerical using label encoding.
*   **Feature Selection**: Used Sequential Feature Selection with a Gradient Boosting Classifier to select the most relevant features.

## Models and Methods

### Baseline Model: K-Nearest Neighbors (KNN)
*   **K Values Tested**: 1 to 10
*   **Best K Value**: 7
*   **Performance**:
    *   Accuracy: 81.16%
    *   Precision: 84.83%
    *   Recall: 80.39%
    *   F1 Score: 82.55%

### Advanced Models
1. **Random Forest (Selected Model)**
    *   **Hyperparameter Tuning**:
        *   Number of Estimators: 50
        *   Criterion: 'gini'
        *   Max Depth: None
        *   Min Samples Split: 4
        *   Min Samples Leaf: 1
    *   **Performance**:
        *   Training Accuracy: 98.13%
        *   Testing Accuracy: 84.06%
        *   Cross-Validation Accuracy: 88.17%
        *   Precision: 86.58%
        *   Recall: 84.31%
        *   F1 Score: 85.43%

2. **Support Vector Machine (SVM)**
    *   **Performance**:
        *   Training Accuracy: 91.90%
        *   Testing Accuracy: 82.97%
        *   Cross-Validation Accuracy: 87.24%
        *   Precision: 82.75%
        *   Recall: 82.89%
        *   F1 Score: 82.81%

3. **Multilayer Perceptron (MLP)**
    *   **Performance**:
        *   Training Accuracy: 90.34%
        *   Testing Accuracy: 83.70%
        *   Cross-Validation Accuracy: 86.93%
        *   Precision: 85.06%
        *   Recall: 85.62%
        *   F1 Score: 85.34%

4. **Logistic Regression**
    *   **Performance**:
        *   Training Accuracy: 85.67%
        *   Testing Accuracy: 81.16%
        *   Cross-Validation Accuracy: 84.43%
        *   Precision: 84.83%
        *   Recall: 80.39%
        *   F1 Score: 82.55%

## Results

The **Random Forest** model outperformed the other models, achieving the highest accuracy and balanced performance across precision, recall, and F1-score.

### Classification Report for Random Forest:

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Negative (0)   | 81%       | 84%    | 82%      | 123     |
| Positive (1)   | 87%       | 84%    | 85%      | 153     |
| **Accuracy**    |           |        | **84%**  | 276     |

### Key Findings:
*   The Random Forest model effectively captured the complex relationships in the data.
*   Hyperparameter tuning significantly improved model performance.
*   The selected features contributed positively to the model's predictive capability.

## Conclusion

This project demonstrates the potential of machine learning models in predicting heart failure risk using clinical data. The Random Forest model, with optimized hyperparameters, provided the best performance. The findings underscore the importance of feature selection and hyperparameter tuning in developing effective predictive models in healthcare.

### Model Limitations:
*   Overfitting: The high training accuracy indicates potential overfitting.
*   Data Imbalance: Slight imbalance in the target classes may affect model performance.
*   Generalizability: The model's applicability to other datasets or populations requires further validation.

## Installation and Usage

### Prerequisites
*   Python 3.x
*   Jupyter Notebook or JupyterLab
*   Required Python libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation Steps
1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/heart-failure-prediction.git
    ```
2. **Navigate to the Project Directory**
    ```bash
    cd heart-failure-prediction
    ```
3. **Install Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook
Open the Jupyter Notebook in your preferred environment:
```bash
jupyter notebook Heart_Failure_Prediction.ipynb
```
Execute the cells sequentially to reproduce the analysis and results.

## Repository Structure
```
heart-failure-prediction/
├── data/
│   └── heart_failure_data.csv
├── images/
│   └── eda_plots/
│       ├── distribution_age.png
│       ├── correlation_heatmap.png
│       └── ...
├── Project_Description.pdf
├── Heart_Failure_Prediction - ML Report.pdf
├── Heart_Failure_Prediction - ML Code.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```
*   **data/**: Contains the dataset used in the project.
*   **images/**: Visualizations and plots generated during EDA. (Not Available yet) 
*   **Project_Description.pdf**: Detailed project description document.
*   **Heart_Failure_Prediction - ML Report.pdf**: Final report summarizing the machine learning analysis and results.
*   **Heart_Failure_Prediction - ML Code.ipynb**: The main Jupyter Notebook with all code and analysis.
*   **requirements.txt**: List of Python libraries required.
*   **LICENSE**: License information.

## Contributors
*   **Eyab Ghifari**
*   **Hamza Awashra**

**Instructor:** Dr. Yazan Abu Farha

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
*   **Dataset Source:** Heart Failure Prediction Dataset by Fedesoriano on Kaggle
*   **Institution:** Birzeit University
*   **Course:** ENCS5341 - Machine Learning and Data Science

* * *

_This project was completed as part of the coursework for ENCS5341 at Birzeit University, aiming to apply machine learning techniques to a real-world healthcare problem._
