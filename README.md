# -VR-Time-Series-Classification-for-Skill-Level-Assessment-

# Time Series Classification with Random Forest for VR Expertise Level

This repository contains Python code for classifying time series data from VR device accelerometers, aimed at identifying user expertise levels ("expert," "intermediate," or "novice"). The classification is based on time series recordings, organized into different CSV files for each participant. The workflow includes data preparation, feature extraction, model training, classification, and the evaluation of new time series. Below is a step-by-step explanation of the code.

## Table of Contents
1. [Setup](#setup)
2. [Data Loading and Labeling](#data-loading-and-labeling)
3. [Time Conversion and Interpolation](#time-conversion-and-interpolation)
4. [Feature Extraction](#feature-extraction)
5. [Model Training and Saving](#model-training-and-saving)
6. [Classification of New Series with Scoring](#classification-of-new-series-with-scoring)
7. [Batch Classification of Multiple Series](#batch-classification-of-multiple-series)
8. [Usage](#usage)

---

### Setup

Ensure you have the required libraries installed. The code relies on:
```bash
pip install numpy pandas scikit-learn scipy joblib
```

### Data Loading and Labeling

Data for "expert" and "novice" users are stored in separate directories, each containing multiple CSV files. The `load_data` function reads each file from the directory and adds a label for classification.

```python
import pandas as pd
import glob
import os

# Directories containing data
expert_dir = './trajectories/expert'
novice_dir = './trajectories/novice'

# Function to load data and add labels
def load_data(directory, label):
    data_list = []
    for filepath in glob.glob(os.path.join(directory, "*.csv")):
        df = pd.read_csv(filepath)
        df['label'] = label  # Add class label
        data_list.append(df)
    return data_list

# Load data from expert and novice directories
expert_data = load_data(expert_dir, label="expert")
novice_data = load_data(novice_dir, label="novice")

# Combine lists into one DataFrame and inspect
all_data = expert_data + novice_data
print("Data loaded from expert and novice.")
print(f"Total time series: {len(all_data)}")
print("Example data:", all_data[0].head())
```

### Time Conversion and Interpolation

Each CSV file includes a timestamp, which we convert to relative time in milliseconds. This conversion is performed using `convert_to_relative_time`, followed by interpolating each series to a standard length using `interpolate_series`.

```python
import numpy as np
from scipy.interpolate import interp1d

# Convert timestamp to relative time in milliseconds
def convert_to_relative_time(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
    df['time'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() * 1000
    return df.drop(columns=['Timestamp'])  # Remove original column

# Apply the function to each series in the list
all_data = [convert_to_relative_time(df) for df in all_data]

# Calculate the average length of expert series
mean_length = int(np.mean([len(df) for df in expert_data]))

# Interpolate series to a standard length
def interpolate_series(df, target_length):
    interpolated_df = pd.DataFrame()
    common_time = np.linspace(0, df['time'].iloc[-1], target_length)
    interpolated_df['time'] = common_time
    for col in df.columns:
        if col != 'time' and col != 'label':  # Exclude label if present
            f = interp1d(df['time'], df[col], kind='linear', fill_value="extrapolate")
            interpolated_df[col] = f(common_time)
    if 'label' in df.columns:
        interpolated_df['label'] = df['label'].iloc[0]
    return interpolated_df

# Apply interpolation to all series
all_data_interpolated = [interpolate_series(df, mean_length) for df in all_data]
print("Data after interpolation:")
print(all_data_interpolated[0].head())
```

### Feature Extraction

The `extract_features` function extracts summary statistics (mean, standard deviation, min, max, median) from each series to create feature vectors for model training and testing.

```python
# Function to extract summary statistics from each series
def extract_features(data_list):
    features = []
    labels = []
    for df in data_list:
        feature_vector = []
        for col in df.columns:
            if col != 'time' and col != 'label':
                feature_vector.extend([
                    df[col].mean(),
                    df[col].std(),
                    df[col].min(),
                    df[col].max(),
                    df[col].median(),
                ])
        features.append(feature_vector)
        labels.append(df['label'].iloc[0])
    return np.array(features), np.array(labels)

# Extract features and labels
X_train, y_train = extract_features(all_data_interpolated)
```

### Model Training and Saving

We train a `RandomForestClassifier` model on the extracted features, then save the trained model using `joblib`.

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
model_filename = './models/random_forest_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved at {model_filename}")
```

### Classification of New Series with Scoring

The function `classify_series_with_score` takes a new time series, processes it, and assigns a classification score based on the probability of the series being classified as "expert." The score is then mapped to a label ("expert," "intermediate," or "novice").

```python
def classify_series_with_score(new_series, model, target_length):
    new_series = convert_to_relative_time(new_series)
    interpolated_series = interpolate_series(new_series, target_length)
    
    # Extract summary statistics
    feature_vector = []
    for col in interpolated_series.columns:
        if col != 'time' and col != 'label':
            feature_vector.extend([
                interpolated_series[col].mean(),
                interpolated_series[col].std(),
                interpolated_series[col].min(),
                interpolated_series[col].max(),
                interpolated_series[col].median(),
            ])
    
    X_new = np.array(feature_vector).reshape(1, -1)
    proba = model.predict_proba(X_new)[0]
    proba_expert = proba[0]
    
    # Convert probability to score
    if proba_expert >= 0.85:
        score = 7
    elif proba_expert >= 0.70:
        score = 6
    elif proba_expert >= 0.55:
        score = 5
    elif proba_expert >= 0.40:
        score = 4
    elif proba_expert >= 0.25:
        score = 3
    elif proba_expert >= 0.10:
        score = 2
    else:
        score = 1

    label = "expert" if score >= 6 else "intermediate" if score >= 4 else "novice"
    return label, score, proba_expert

# Example usage
csv_directory = './trajectories/test_serie/'
for csv_file in glob.glob(f"{csv_directory}/*.csv"):
    new_series_df = pd.read_csv(csv_file)
    label, score, proba_expert = classify_series_with_score(new_series_df, model, mean_length)
    print(f"Classification for {csv_file}:")
    print(f"Label: {label}")
    print(f"Score on a scale of 1 to 7: {score}")
    print(f"Probability of being an expert: {proba_expert:.2f}\n")
```

### Usage

To run the code, place your CSV files in the specified folders, set up the directory structure as shown, and execute the Python script. The output will classify each time series, displaying the label, score, and probability of expertise.

### Directory Structure

```
project/
├── models/
│   └── random_forest_model.joblib
├── trajectories/
│   ├── expert/
│   ├── novice/
│   └── test_serie/
└── main.py
```



