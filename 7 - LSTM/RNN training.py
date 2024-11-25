import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(df, seq_length=24):
    """Preprocess the data and create sequences"""
    # Convert power data to numpy array
    power_data = df['Global_active_power'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    power_data_scaled = scaler.fit_transform(power_data)
    
    # Create sequences
    X, y = create_sequences(power_data_scaled, seq_length)
    
    # Split into train and validation sets (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, X_val, y_train, y_val, scaler

def build_model(seq_length, n_features=1):
    """Build the LSTM model"""
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, 
             input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate(df, seq_length=24, epochs=50, batch_size=32):
    """Train and evaluate the model"""
    # Preprocess data
    X_train, X_val, y_train, y_val, scaler = preprocess_data(df, seq_length)
    
    # Build model
    model = build_model(seq_length)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, scaler

def make_predictions(model, data, scaler, seq_length):
    """Make predictions using the trained model"""
    predictions = model.predict(data)
    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Example usage
if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('household_power_consumption.txt', sep=';', 
                     parse_dates={'datetime': ['Date', 'Time']},
                     infer_datetime_format=True)
    
    # Train the model
    seq_length = 24  # predict based on last 24 time steps
    model, history, scaler = train_and_evaluate(df, seq_length)
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()