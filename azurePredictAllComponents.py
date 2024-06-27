import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load the datasets
errors = pd.read_csv('azure_data/PdM_errors.csv', parse_dates=['datetime'])
failures = pd.read_csv('azure_data/PdM_failures.csv', parse_dates=['datetime'])
machines = pd.read_csv('azure_data/PdM_machines.csv')
maint = pd.read_csv('azure_data/PdM_maint.csv', parse_dates=['datetime'])
telemetry = pd.read_csv('azure_data/PdM_telemetry.csv', parse_dates=['datetime'])

# Preprocess the data
# Handle missing values
errors = errors.dropna(subset=['datetime', 'machineID', 'errorID'])
failures = failures.dropna(subset=['datetime', 'machineID', 'failure'])
machines = machines.dropna(subset=['machineID', 'model', 'age'])
maint = maint.dropna(subset=['datetime', 'machineID', 'comp'])
telemetry = telemetry.dropna(subset=['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration'])

# Convert categorical variables to numerical
machines['model'] = machines['model'].astype('category').cat.codes
errors['errorID'] = errors['errorID'].astype('category').cat.codes

# Split the datetime column into separate features
errors['date'] = errors['datetime'].dt.date
errors['time'] = errors['datetime'].dt.time
failures['date'] = failures['datetime'].dt.date
failures['time'] = failures['datetime'].dt.time
maint['date'] = maint['datetime'].dt.date
maint['time'] = maint['datetime'].dt.time
telemetry['date'] = telemetry['datetime'].dt.date
telemetry['time'] = telemetry['datetime'].dt.time

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scale numerical columns for machines DataFrame
machine_numerical_cols = ['age']
machines[machine_numerical_cols] = scaler.fit_transform(machines[machine_numerical_cols])

# Scale numerical columns for telemetry DataFrame
telemetry_numerical_cols = ['volt', 'rotate', 'pressure', 'vibration']
telemetry[telemetry_numerical_cols] = scaler.fit_transform(telemetry[telemetry_numerical_cols])

# Identify the components for each machine
components = pd.concat([failures['failure'].str.extract('comp(\\d+)', expand=False),
                        maint['comp'].str.extract('comp(\\d+)', expand=False)], ignore_index=True).drop_duplicates()

# Train a model for each component
for component in components:
    # Prepare the training data for the current component
    comp_failures = failures[failures['failure'] == f'comp{component}']
    comp_maint = maint[maint['comp'] == f'comp{component}']
    
    # Create separate lists for each feature
    time_since_failure = []
    time_since_maint = []
    volt_data = []
    rotate_data = []
    pressure_data = []
    vibration_data = []
    y_train = []
    
    for _, row in comp_maint.iterrows():
        machine_id = row['machineID']
        maint_date = row['datetime']
        
        # Find the previous failure and maintenance events for this machine and component
        prev_failures = comp_failures[
            (comp_failures['machineID'] == machine_id) &
            (comp_failures['datetime'] < maint_date)
        ]['datetime'].tolist()
        prev_maint = comp_maint[
            (comp_maint['machineID'] == machine_id) &
            (comp_maint['datetime'] < maint_date)
        ]['datetime'].tolist()
        
        # Extract the last telemetry data before the maintenance event
        telemetry_data = telemetry[
            (telemetry['machineID'] == machine_id) &
            (telemetry['datetime'] < maint_date)
        ]
        if not telemetry_data.empty:
            telemetry_data = telemetry_data.iloc[-1][['volt', 'rotate', 'pressure', 'vibration']].values
        else:
            telemetry_data = [0, 0, 0, 0]
        
        # Compute the time since the last failure and maintenance
        time_since_failure.append((maint_date - max(prev_failures, default=pd.Timestamp(1970, 1, 1))) / pd.Timedelta(days=1) if prev_failures else 0)
        time_since_maint.append((maint_date - max(prev_maint, default=pd.Timestamp(1970, 1, 1))) / pd.Timedelta(days=1) if prev_maint else 0)
        
        # Append the telemetry data to separate lists
        volt_data.append(telemetry_data[0])
        rotate_data.append(telemetry_data[1])
        pressure_data.append(telemetry_data[2])
        vibration_data.append(telemetry_data[3])
        
        # Compute the target variable (remaining days until failure)
        next_failure = comp_failures[
            (comp_failures['machineID'] == machine_id) &
            (comp_failures['datetime'] > maint_date)
        ]['datetime']
        if not next_failure.empty:
            remaining_days = (next_failure.min() - maint_date) / pd.Timedelta(days=1)
        else:
            remaining_days = 365  # Assign a default value (e.g., 365 days or any other reasonable value)
        y_train.append(remaining_days)
    
    # Create the feature matrix X_train
    X_train = np.column_stack((time_since_failure, time_since_maint, volt_data, rotate_data, pressure_data, vibration_data))
    
    # Train a model (e.g., Random Forest Regressor)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Make predictions for each machine
    for machine_id in machines['machineID']:
        latest_telemetry = telemetry[telemetry['machineID'] == machine_id].iloc[-1]
        
        # Find the previous failure and maintenance events for this machine and component
        prev_failures = comp_failures[
            (comp_failures['machineID'] == machine_id)
        ]['datetime'].tolist()
        prev_maint = comp_maint[
            (comp_maint['machineID'] == machine_id)
        ]['datetime'].tolist()
        
        # Compute the time since the last failure and maintenance
        time_since_failure = (latest_telemetry['datetime'] - max(prev_failures, default=pd.Timestamp(1970, 1, 1))) / pd.Timedelta(days=1) if prev_failures else 0
        time_since_maint = (latest_telemetry['datetime'] - max(prev_maint, default=pd.Timestamp(1970, 1, 1))) / pd.Timedelta(days=1) if prev_maint else 0
        
        # Extract the latest telemetry data
        features = [time_since_failure, time_since_maint,
                    latest_telemetry['volt'], latest_telemetry['rotate'],
                    latest_telemetry['pressure'], latest_telemetry['vibration']]
        
        # Make a prediction
        remaining_days = model.predict([features])[0]
        
        print(f"Machine ID: {machine_id}, Component: comp{component}, Remaining days: {int(remaining_days)}")