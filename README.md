# Intrusion Detection System - 

**Intrusion-Detection-System-Hacknova-2024** is a **Machine Learning-based Network Intrusion Detection System (IDS)**. This project leverages network traffic data to identify and classify various types of cyber attacks using a **Random Forest Classifier**, and provides an intuitive, interactive dashboard built with **Streamlit**.

---

## Project Overview

The system detects different types of network intrusions by analyzing key features extracted from network flows such as:

- Flow Duration
- Total Forward Packets
- Fwd Packets Length Total
- Flow Bytes per Second
- Flow Packets per Second

Using these features, the model classifies network traffic into categories like **benign traffic** or specific attack types (e.g., DDoS, Portscan, Botnet, Webattack, etc.).

This project includes two main components:
1. **Model Training Script**: Trains the Random Forest classifier on a sampled dataset.
2. **Streamlit Dashboard**: Provides an interactive UI for real-time prediction and data visualization.

---

## Features

✅ Machine Learning Model trained using **Random Forest Classifier**  
✅ Predicts multiple types of network attacks  
✅ Interactive **Streamlit Dashboard** for predictions and visualizations  
✅ Includes **bar charts**, **pie charts**, and **correlation heatmap** for insights  
✅ Uses **LabelEncoder** to decode predictions back to human-readable labels  
✅ Optimized for memory usage and fast inference  

---

##  Technologies Used

| Technology | Description |
|----------|-------------|
| **Python** | Core programming language |
| **Pandas** | Data manipulation and preprocessing |
| **Scikit-Learn** | Machine learning algorithms and evaluation metrics |
| **Joblib** | Model and encoder serialization |
| **Streamlit** | Building the interactive web dashboard |
| **Seaborn & Matplotlib** | Data visualization tools |
| **PyArrow** | Efficient Parquet file reading |

---

##  Dataset

The dataset used in this project is a **sampled version** of the `cic-collection.parquet` dataset (CICIDS), containing labeled network traffic records. We sampled **1% of the full dataset** to reduce training time while maintaining meaningful patterns for classification.

For more information about the dataset: [CICIDS Dataset](https://www.unb.ca/cic/datasets/index.html) 

---

##  Model Training

### Steps:
1. Load and sample the dataset (`cic-collection.parquet`)
2. Preprocess data:
   - Handle missing values
   - Optimize memory usage (float32/int32 conversion)
   - Encode categorical labels using `LabelEncoder`
##  Results 

The model achieved the following performance metrics on the test set:

| Metric              | Score      |
|---------------------|------------|
| **Accuracy**        | ~90%       |
| **Precision (avg)** | ~0.88      |
| **Recall (avg)**    | ~0.85      |
| **F1-Score (avg)**  | ~0.86      |

> ⚠️ *Note:* These scores may vary depending on dataset sampling, feature selection, and model hyperparameters.

## Future Improvements
Incorporate more features for better accuracy
Use more advanced models like XGBoost or Deep Learning
Add real-time packet capture and analysis
Implement alerting mechanisms
Deploy the dashboard online using Streamlit Sharing or Docker

