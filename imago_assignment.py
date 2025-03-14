import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Load the dataset
df = pd.read_csv(r"D:\TASK-ML-INTERN.csv")

# Separate features (X) and target (y)
X = df.drop(columns=['vomitoxin_ppb'])  # Features
y = df['vomitoxin_ppb']  # Target variable
spectral_data = df.iloc[:, 1:-1]

# Standardize the features (PCA requires standardized data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(spectral_data)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Variance explained by the top principal components
explained_variance = pca.explained_variance_ratio_
print(f"Variance explained by each component: {explained_variance}")

# Visualize the reduced data (2D scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.title('PCA Reduced Data (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Vomitoxin (ppb)')
plt.show()

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 1: Implement an Attention Mechanism
def build_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    # Reshape input to 3D for MultiHeadAttention
    x = Reshape((input_shape[0], 1))(inputs)  # Shape: (batch_size, sequence_length, num_features)
    x = Dense(128, activation='relu')(x)
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

# Compile and train the attention model
attention_model = build_attention_model((X_train.shape[1],))
attention_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
attention_history = attention_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the attention model
y_pred_attention = attention_model.predict(X_test)
mae_attention = mean_absolute_error(y_test, y_pred_attention)
rmse_attention = np.sqrt(mean_squared_error(y_test, y_pred_attention))
r2_attention = r2_score(y_test, y_pred_attention)
print(f"Attention Model - MAE: {mae_attention}, RMSE: {rmse_attention}, R²: {r2_attention}")

# Compare with the feedforward neural network
def build_feedforward_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model

feedforward_model = build_feedforward_model((X_train.shape[1],))
feedforward_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
feedforward_history = feedforward_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the feedforward model
y_pred_feedforward = feedforward_model.predict(X_test)
mae_feedforward = mean_absolute_error(y_test, y_pred_feedforward)
rmse_feedforward = np.sqrt(mean_squared_error(y_test, y_pred_feedforward))
r2_feedforward = r2_score(y_test, y_pred_feedforward)
print(f"Feedforward Model - MAE: {mae_feedforward}, RMSE: {rmse_feedforward}, R²: {r2_feedforward}")

# Step 2: Create a Streamlit App for Interactive Predictions
def streamlit_app():
    st.title("Spectral Data Analysis and Prediction")
    uploaded_file = st.file_uploader("Upload your spectral data (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        # Preprocess data
        spectral_columns = df.columns[1:-1]  # Exclude 'hsi_id' and 'vomitoxin_ppb'
        df[spectral_columns] = scaler.transform(df[spectral_columns])

        # Make predictions using the attention model
        X_uploaded = df[spectral_columns]
        y_pred_uploaded = attention_model.predict(X_uploaded)
        df['Predicted_Vomitoxin_ppb'] = y_pred_uploaded

        # Display predictions
        st.write("Predictions:")
        st.write(df[['hsi_id', 'vomitoxin_ppb', 'Predicted_Vomitoxin_ppb']])

        # Visualize predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(df['vomitoxin_ppb'], df['Predicted_Vomitoxin_ppb'], alpha=0.5)
        plt.plot([min(df['vomitoxin_ppb']), max(df['vomitoxin_ppb'])], 
                 [min(df['vomitoxin_ppb']), max(df['vomitoxin_ppb'])], color='red', linestyle='--')
        plt.title('Actual vs Predicted Vomitoxin (ppb)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        st.pyplot(plt)

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()