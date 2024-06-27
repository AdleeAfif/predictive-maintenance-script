import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load datasets
telemetry = pd.read_csv('azure_data/PdM_telemetry.csv', parse_dates=['datetime'])
errors = pd.read_csv('azure_data/PdM_errors.csv', parse_dates=['datetime'])
failures = pd.read_csv('azure_data/PdM_failures.csv', parse_dates=['datetime'])
maint = pd.read_csv('azure_data/PdM_maint.csv', parse_dates=['datetime'])
machines = pd.read_csv('azure_data/PdM_machines.csv')

# Preprocessing: Aggregate telemetry data (e.g., daily averages)
telemetry['date'] = telemetry['datetime'].dt.date
daily_telemetry = telemetry.groupby(['machineID', 'date']).agg({
    'volt': 'mean',
    'rotate': 'mean',
    'pressure': 'mean',
    'vibration': 'mean'
}).reset_index()

# Create rolling statistics (e.g., 7-day rolling mean)
rolling_features = daily_telemetry.groupby('machineID')[['volt', 'rotate', 'pressure', 'vibration']].rolling(window=7).mean().reset_index()
rolling_features.columns = ['machineID', 'date', 'volt_rolling_mean', 'rotate_rolling_mean', 'pressure_rolling_mean', 'vibration_rolling_mean']

# Merge daily telemetry with rolling features
features = pd.merge(daily_telemetry, rolling_features, on=['machineID', 'date'], how='left')

# Merge errors and maint with features
features = pd.merge(features, errors, left_on=['machineID', 'date'], right_on=['machineID', errors['datetime'].dt.date], how='left')
features = pd.merge(features, maint, left_on=['machineID', 'date'], right_on=['machineID', maint['datetime'].dt.date], how='left')

# Add machine information
features = pd.merge(features, machines, on='machineID', how='left')

# Fill NaN values for numeric columns with 0
features[['errorID', 'comp']] = features[['errorID', 'comp']].fillna('none')
features[['volt', 'rotate', 'pressure', 'vibration',
          'volt_rolling_mean', 'rotate_rolling_mean', 'pressure_rolling_mean', 'vibration_rolling_mean']] = features[['volt', 'rotate', 'pressure', 'vibration',
                                                                                                                     'volt_rolling_mean', 'rotate_rolling_mean', 'pressure_rolling_mean', 'vibration_rolling_mean']].fillna(0)

# Ensure datetime columns are excluded from the feature matrix
columns_to_drop = ['datetime_x', 'datetime_y', 'datetime']
columns_present = [col for col in columns_to_drop if col in features.columns]
features = features.drop(columns=columns_present)

# Convert categorical variables to numeric using one-hot encoding
features = pd.get_dummies(features, columns=['model', 'errorID', 'comp'], drop_first=True)

# Prepare the feature set and labels
# Create a feature for days_to_failure based on the failures data
features['datetime'] = pd.to_datetime(features['date'])
failures['datetime'] = pd.to_datetime(failures['datetime'])
features['days_to_failure'] = features.apply(
    lambda row: (failures.loc[(failures['machineID'] == row['machineID']) & (failures['datetime'] > row['datetime'])]['datetime'].min() - row['datetime']).days 
    if len(failures.loc[(failures['machineID'] == row['machineID']) & (failures['datetime'] > row['datetime'])]) > 0 
    else np.nan, axis=1)

# Create a feature for component failure
def next_failure_component(row):
    future_failures = failures.loc[(failures['machineID'] == row['machineID']) & (failures['datetime'] > row['datetime'])]
    return future_failures['failure'].iloc[0] if not future_failures.empty else 'none'

features['next_failure_component'] = features.apply(next_failure_component, axis=1)

# Print unique components to check
print("Unique components:", features['next_failure_component'].unique())

# Drop rows where days_to_failure is NaN (no failure recorded after the date)
features = features.dropna(subset=['days_to_failure'])

# Define features and targets
X = features.drop(columns=['datetime', 'date', 'days_to_failure', 'next_failure_component'])

# Initialize LabelEncoder
le = LabelEncoder()

# Convert component labels to numeric
features['next_failure_component_encoded'] = le.fit_transform(features['next_failure_component'])

# Print the mapping between labels and numeric values
component_mapping = {index: label for index, label in enumerate(le.classes_)}
print("Component Mapping:", component_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train_days, y_test_days, y_train_component, y_test_component = train_test_split(
    X, features['days_to_failure'], features['next_failure_component_encoded'], test_size=0.2, random_state=42
)

# Train separate models for each task
model_days = RandomForestRegressor(n_estimators=100, random_state=42)
model_component = RandomForestClassifier(n_estimators=100, random_state=42)

model_days.fit(X_train, y_train_days)
model_component.fit(X_train, y_train_component)

# Make predictions
y_pred_days = model_days.predict(X_test)
y_pred_component = model_component.predict(X_test)

# Evaluate the models
mae_days = mean_absolute_error(y_test_days, y_pred_days)
accuracy_component = accuracy_score(y_test_component, y_pred_component)

print(f'Mean Absolute Error for days to failure: {mae_days}')
print(f'Accuracy for component prediction: {accuracy_component}')

# Print classification report for component prediction
print("Classification Report for Component Prediction:")
print(classification_report(y_test_component, y_pred_component, target_names=le.classes_))

# Predict remaining days and components for the entire dataset
predicted_days = model_days.predict(X)
predicted_components_encoded = model_component.predict(X)

# Convert component predictions back to labels
predicted_component_names = le.inverse_transform(predicted_components_encoded)

# Add predictions to the features dataframe
features['predicted_days_to_failure'] = predicted_days
features['predicted_failure_component'] = predicted_component_names

# Display the predictions for multiple machines
print("\nPredictions:")
print(features[['machineID', 'datetime', 'predicted_days_to_failure', 'predicted_failure_component']].head(20))  # Display the first 20 predictions

# Optional: Feature Importance
feature_importance_days = pd.Series(model_days.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importance_component = pd.Series(model_component.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 10 Features for Days to Failure Prediction:")
print(feature_importance_days.head(10))

print("\nTop 10 Features for Component Prediction:")
print(feature_importance_component.head(10))

# Save the models and label encoder for future use
# import joblib

# joblib.dump(model_days, 'model_days.pkl')
# joblib.dump(model_component, 'model_component.pkl')
# joblib.dump(le, 'label_encoder.pkl')

# print("\nModels and label encoder saved.")