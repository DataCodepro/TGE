import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load and preprocess the dataset
data = pd.read_csv('TGE_price.csv')

# Fill missing values
data.fillna(0, inplace=True)

# Map categorical columns to numerical values
narrative_mapping = {name: idx for idx, name in enumerate(data['narrative'].unique())}
type_mapping = {name: idx for idx, name in enumerate(data['type'].unique())}
cex_1_mapping = {name: idx for idx, name in enumerate(data['cex_1'].unique())}
cex_2_mapping = {name: idx for idx, name in enumerate(data['cex_2'].unique())}
cex_3_mapping = {name: idx for idx, name in enumerate(data['cex_3'].unique())}

data['narrative'] = data['narrative'].map(narrative_mapping)
data['type'] = data['type'].map(type_mapping)
data['cex_1'] = data['cex_1'].map(cex_1_mapping)
data['cex_2'] = data['cex_2'].map(cex_2_mapping)
data['cex_3'] = data['cex_3'].map(cex_3_mapping)

# Define features and target variables
features = data[['narrative', 'type', 'inst_ico', 'comm_ico', 'listing_price', 'inst_vesting_perc', 'pub_vesting_perc', 'cex_1', 'cex_2', 'cex_3', 'x_comm']]
targets = data[['hr1_high', 'hr1_low', 'hr2_high', 'hr2_low', 'hr3_high', 'hr3_low']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Streamlit app
st.title('TGE Price Prediction Dashboard')

# Input features from the user
st.sidebar.header('Input Features')

def user_input_features():
    narrative = st.sidebar.selectbox('Narrative', options=list(narrative_mapping.keys()), index=0)
    type_ = st.sidebar.selectbox('Type', options=list(type_mapping.keys()), index=0)
    inst_ico = st.sidebar.slider('Inst ICO', 0, 1, 0)
    comm_ico = st.sidebar.slider('Comm ICO', 0, 1, 0)
    
    # Replace sliders with number inputs for manual entry
    listing_price = st.sidebar.number_input('Listing Price', min_value=0.000000, value=0.000001, format="%.6f")
    inst_vesting_perc = st.sidebar.slider('Inst Vesting %', 1, 100, 1)
    pub_vesting_perc = st.sidebar.slider('Pub Vesting %', 1, 100, 1)
    cex_1 = st.sidebar.selectbox('CEX 1', options=list(cex_1_mapping.keys()), index=0)
    cex_2 = st.sidebar.selectbox('CEX 2', options=list(cex_2_mapping.keys()), index=0)
    cex_3 = st.sidebar.selectbox('CEX 3', options=list(cex_3_mapping.keys()), index=0)
    
    # Replace slider with number input for manual entry
    x_comm = st.sidebar.number_input('X Comm', min_value=1, value=1)

    features_dict = {
        'narrative': narrative_mapping[narrative],
        'type': type_mapping[type_],
        'inst_ico': inst_ico,
        'comm_ico': comm_ico,
        'listing_price': listing_price,
        'inst_vesting_perc': inst_vesting_perc,
        'pub_vesting_perc': pub_vesting_perc,
        'cex_1': cex_1_mapping[cex_1],
        'cex_2': cex_2_mapping[cex_2],
        'cex_3': cex_3_mapping[cex_3],
        'x_comm': x_comm
    }
    
    return pd.DataFrame(features_dict, index=[0])

input_features = user_input_features()

# Make prediction
prediction = model.predict(input_features)

# Display results
st.subheader('Prediction Results')
st.write('Predicted HR1 Low and High Prices:')
st.write(f"HR1 Low: {prediction[0][1]}")
st.write(f"HR1 High: {prediction[0][0]}")
st.write('Predicted HR2 Low and High Prices:')
st.write(f"HR2 Low: {prediction[0][3]}")
st.write(f"HR2 High: {prediction[0][2]}")
st.write('Predicted HR2 Low and High Prices:')
st.write(f"HR3 Low: {prediction[0][5]}")
st.write(f"HR3 High: {prediction[0][5]}")
