import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset from a CSV file
df = pd.read_csv('predictive_maintenance.csv')

# Display the DataFrame to check the changes
print(df.head())

# Feature selection
X = df[['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']]
y_maintenance = df['maintenance_required']
y_reason = df['maintenance_reason']

# Encode categorical labels
le = LabelEncoder()
y_reason_encoded = le.fit_transform(y_reason)

# Combine the targets
y_combined = pd.DataFrame({'maintenance_required': y_maintenance, 'maintenance_reason': y_reason_encoded})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_model = MultiOutputClassifier(rf, n_jobs=-1)
multi_target_model.fit(X_train, y_train)

# Make predictions
y_pred = multi_target_model.predict(X_test)

# Evaluate the model
accuracy_maintenance = accuracy_score(y_test["maintenance_required"], y_pred[:, 0])
accuracy_reason = accuracy_score(y_test["maintenance_reason"], y_pred[:, 1])

print(f'Accuracy for maintenance required: {accuracy_maintenance}')
print(f'Accuracy for maintenance reason: {accuracy_reason}')

# Ensure the target names and labels match
target_names = le.classes_
labels = list(range(len(target_names)))

print("Classification report for maintenance required:")
print(classification_report(y_test["maintenance_required"], y_pred[:, 0]))

print("Classification report for maintenance reason:")
print(classification_report(y_test["maintenance_reason"], y_pred[:, 1], target_names=target_names, labels=labels, zero_division=0))

