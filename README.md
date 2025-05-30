# Intrusion-Detection-System-Hacknova-2024-
This project implements an Intrusion Detection System (IDS) using machine learning techniques. The goal of the system is to classify different types of network attacks using network traffic data. The model is trained using a Random Forest classifier to detect various network attacks based on labeled data and its associated features. Additionally, we are using Streamlit to create an interactive web dashboard that allows users to visualize the data and model predictions.

## Libraries Used

This project uses several Python libraries for data processing, machine learning, model evaluation, and dashboard visualization:

### 1. **Pandas**
   - **Purpose**: Used for data manipulation and analysis. It provides data structures like `DataFrame` for handling and analyzing structured data efficiently.
   - **Installation**: `pip install pandas`
   - **Usage**: Loading and preprocessing the dataset, handling missing values, and performing feature extraction.
   
### 2. **NumPy**
   - **Purpose**: Provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
   - **Installation**: `pip install numpy`
   - **Usage**: Used for numerical operations like creating arrays, performing calculations, and handling feature data.

### 3. **Scikit-learn**
   - **Purpose**: A machine learning library that includes simple and efficient tools for data mining and data analysis. It includes algorithms for classification, regression, clustering, and more.
   - **Installation**: `pip install scikit-learn`
   - **Usage**: 
     - **LabelEncoder**: For encoding categorical variables into numeric values.
     - **RandomForestClassifier**: Used as the model for training and predicting network attack types.
     - **train_test_split**: For splitting the dataset into training and testing sets.
     - **accuracy_score** and **classification_report**: For model evaluation.

### 4. **Joblib**
   - **Purpose**: Used for serializing Python objects (saving and loading models and transformers).
   - **Installation**: `pip install joblib`
   - **Usage**: Saving and loading the trained machine learning model and label encoder to disk.

### 5. **pyarrow**
   - **Purpose**: This library is used for handling the Parquet file format, enabling fast data processing.
   - **Installation**: `pip install pyarrow`
   - **Usage**: Reading Parquet files for efficient data loading.

### 6. **Streamlit**
   - **Purpose**: Used for creating interactive web applications and dashboards for machine learning projects.
   - **Installation**: `pip install streamlit`
   - **Usage**: Building a user-friendly interface for the IDS, allowing users to input data and visualize results. Streamlit will display model predictions and graphs dynamically.

### 7. **Matplotlib**
   - **Purpose**: A plotting library used for creating static, animated, and interactive visualizations in Python.
   - **Installation**: `pip install matplotlib`
   - **Usage**: Used for creating various plots such as bar charts, line plots, and histograms to visualize data and model performance.

### 8. **Seaborn**
   - **Purpose**: A Python data visualization library based on Matplotlib, which provides a high-level interface for drawing attractive statistical graphics.
   - **Installation**: `pip install seaborn`
   - **Usage**: Used for creating more advanced visualizations like correlation heatmaps and distributions.

### 9. **Plotly**
   - **Purpose**: A graphing library that makes interactive, publication-quality graphs online.
   - **Installation**: `pip install plotly`
   - **Usage**: Used for creating interactive plots, especially for visualizing model predictions and performance metrics.

## Project Workflow

1. **Data Loading and Preprocessing**:
   - The dataset is loaded from a Parquet file using the `pyarrow` engine.
   - Categorical features like `Label` and `ClassLabel` are encoded into numeric values using `LabelEncoder` from `scikit-learn`.
   - Missing values are checked and handled appropriately.

2. **Feature Selection**:
   - Features are extracted by dropping the target columns (`Label` and `ClassLabel`) from the dataset.
   - The `Label` column is selected as the target variable for the classification task.

3. **Model Training**:
   - The dataset is split into training and testing sets using `train_test_split`.
   - A Random Forest classifier (`RandomForestClassifier`) is used to train the model on the training data. The model is trained with parallel processing enabled by setting `n_jobs=-1` for faster training.

4. **Model Evaluation**:
   - The model’s performance is evaluated using accuracy and classification metrics provided by `accuracy_score` and `classification_report` from `scikit-learn`.

5. **Model Saving**:
   - The trained model and label encoder are saved using `joblib` for later use in the application (for example, in a Streamlit dashboard).

6. **Dashboard Creation**:
   - Streamlit is used to create an interactive web dashboard where users can upload their own data, visualize the attack types, and view real-time predictions of network traffic based on the trained model.
   - Various plots and graphs are displayed using `Matplotlib`, `Seaborn`, and `Plotly` for a better understanding of the data and model performance.

## Requirements

To install the required libraries, run the following command:
```bash
pip install pandas numpy scikit-learn joblib pyarrow streamlit matplotlib seaborn plotly
