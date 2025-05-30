import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Intrusion Detection System", page_icon=":shield:", layout="wide")

# Load the trained model
model = joblib.load(r'E:\ADVI\Code\python\IDS\trained_model.joblib')

# Load the LabelEncoder for decoding predictions
label_encoder = joblib.load(r'E:\ADVI\Code\python\IDS\label_encoder.joblib')iir

# Streamlit App Title
st.title("🌐 Intrusion Detection System Dashboard")
st.markdown("A **Machine Learning** application to detect and classify network intrusions. 🚀")
st.sidebar.title("🔧 Configuration Panel")
st.sidebar.markdown("Use the options below to make predictions and explore the data.")

# User Inputs for Prediction based on the selected 5 features
st.sidebar.header("🌟 Input Features for Prediction")
feature_1 = st.sidebar.slider("Flow Duration", min_value=0.0, max_value=1000000.0, value=500000.0, step=1000.0)
feature_2 = st.sidebar.slider("Total Fwd Packets", min_value=0, max_value=50000, value=2000, step=100)
feature_3 = st.sidebar.slider("Fwd Packets Length Total", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
feature_4 = st.sidebar.slider("Flow Bytes/s", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
feature_5 = st.sidebar.slider("Flow Packets/s", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)

# Combine user inputs into a NumPy array for prediction
input_features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])

# Prediction when the button is pressed
if st.sidebar.button("✨ Predict Attack Type"):
    prediction = model.predict(input_features)
    predicted_label = label_encoder.inverse_transform(prediction)
    st.markdown(f"### 🎯 Prediction Result: **{predicted_label[0]}**")
    st.success(f"The predicted attack type is **{predicted_label[0]}**.")

# Section for Data Insights and Visualizations
st.markdown("---")
st.markdown("## 📊 Data Insights and Visualizations")

# Example data for visualization (you can replace this with your dataset's summary statistics)
example_data = {
    "Attack Type": ["Benign", "Botnet", "DDoS", "Portscan", "Webattack"],
    "Count": [50, 10, 20, 15, 5],
}
df_viz = pd.DataFrame(example_data)

# Bar Chart
st.subheader("Attack Type Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="Attack Type", y="Count", data=df_viz, palette="viridis", ax=ax)
ax.set_title("Distribution of Attack Types", fontsize=16, fontweight="bold")
st.pyplot(fig)

# Pie Chart
st.subheader("Proportions of Attack Types")
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(
    df_viz["Count"],
    labels=df_viz["Attack Type"],
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("muted", len(df_viz)),
)
ax_pie.set_title("Proportions of Attack Types", fontsize=16, fontweight="bold")
st.pyplot(fig_pie)

# Heatmap (example correlation matrix)
st.subheader("Feature Correlation Heatmap")
example_features = np.random.rand(10, 5)  # Replace with your dataset's feature matrix
example_df = pd.DataFrame(example_features, columns=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"])
correlation_matrix = example_df.corr()

fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax_heatmap)
ax_heatmap.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
st.pyplot(fig_heatmap)

# Footer Section
st.markdown("---")
st.markdown(
    """
    🎉 Thank you for using our **Intrusion Detection System**!  
    Designed with 💻 and ❤️ for Hacknova 2024.
    """
)
