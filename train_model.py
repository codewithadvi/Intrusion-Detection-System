import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Replace with the actual path to your Parquet file
df = pd.read_parquet(r'E:\ADVI\Code\python\IDS\cic-collection.parquet', engine='pyarrow')

# Sample 1% of the dataset for faster training
df_sampled = df.sample(frac=0.01, random_state=42)  # 1% of data

# Display the first few rows of the dataset
print(df_sampled.head())

# Check for missing valuest
print(df_sampled.isnull().sum())

# Check the data types and optimize memory usage
for col in df_sampled.select_dtypes(include=['float64']).columns:
    df_sampled[col] = df_sampled[col].astype('float32')

for col in df_sampled.select_dtypes(include=['int64']).columns:
    df_sampled[col] = df_sampled[col].astype('int32')

# Encode 'Label' and 'ClassLabel' columns
label_encoder = LabelEncoder()
df_sampled['Label'] = label_encoder.fit_transform(df_sampled['Label'])
df_sampled['ClassLabel'] = label_encoder.fit_transform(df_sampled['ClassLabel'])

# Select the features you want to use for training (5 features in this case)
selected_features = [
    'Flow Duration', 'Total Fwd Packets', 'Fwd Packets Length Total', 
    'Flow Bytes/s', 'Flow Packets/s'
]

# Ensure that only the selected features are used
X = df_sampled[selected_features]  # Features (only 5 selected features)
Y = df_sampled['Label']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1)  # Reduce trees to 10

# Fit the model on training data
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model's performance
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# Save the trained RandomForest model
joblib.dump(model, 'trained_model.joblib')

# Save the label encoder for future use
joblib.dump(label_encoder, 'label_encoder.joblib')

print("Model and LabelEncoder saved successfully!")
