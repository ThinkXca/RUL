#!/usr/bin/env python3
"""
Battery voltage and time difference feature extraction using TLSTM_AE
Complete implementation based on T_LSTM_AE model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from TLSTM_AE import T_LSTM_AE

class BatteryTLSTMAEFeatureExtractor:
    """Battery feature extractor based on TLSTM_AE"""
    
    def __init__(self, input_dim=1, output_dim=32, output_dim2=16, output_dim3=1,
                 hidden_dim=64, hidden_dim2=32, hidden_dim3=16, dropout_rate=0.1):
        """
        Initialize TLSTM_AE feature extractor
        
        Args:
            input_dim: Input dimension (voltage)
            output_dim: First encoder layer output dimension
            output_dim2: Second encoder layer output dimension  
            output_dim3: Final decoder output dimension
            hidden_dim: First layer hidden dimension
            hidden_dim2: Second layer hidden dimension (main feature dimension)
            hidden_dim3: Third layer hidden dimension
            dropout_rate: Dropout rate
        """
        self.model = T_LSTM_AE(
            input_dim=input_dim,
            output_dim=output_dim,
            output_dim2=output_dim2,
            output_dim3=output_dim3,
            hidden_dim=hidden_dim,
            hidden_dim2=hidden_dim2,
            hidden_dim3=hidden_dim3,
            dropout_rate=dropout_rate
        )
        
        # Save model configuration
        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'output_dim2': output_dim2,
            'output_dim3': output_dim3,
            'hidden_dim': hidden_dim,
            'hidden_dim2': hidden_dim2,
            'hidden_dim3': hidden_dim3,
            'dropout_rate': dropout_rate
        }
        
        self.voltage_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.is_trained = False
        
        # Initialization complete
    
    def load_battery_data(self, csv_file, sequence_length=10):
        """
        Load and preprocess battery data (random sampling of 50 data points)
        
        Args:
            csv_file: Data file path
            sequence_length: Target sequence length (default 50)
            
        Returns:
            voltage_data: Voltage sequence data [n_batteries, seq_len, 1]
            time_data: Time difference sequence data [n_batteries, seq_len, 1]
            battery_info: Battery information list
        """
        print(f"Loading battery data from: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"Error: Data file not found {csv_file}")
            return None, None, None
        
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['time_minutes', 'voltage_V']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None, None, None
        
        # Handle battery ID
        if 'cell_id' not in df.columns:
            # If no cell_id, try other possible ID columns
            id_candidates = ['battery_id', 'id', 'cell', 'battery']
            found_id = None
            for candidate in id_candidates:
                if candidate in df.columns:
                    df['cell_id'] = df[candidate]
                    found_id = candidate
                    break
            
            if found_id is None:
                # If none found, assume all data is from one battery
                df['cell_id'] = 'battery_1'
        
        unique_cells = sorted(df['cell_id'].unique())
        
        voltage_sequences = []
        time_sequences = []
        battery_info = []
        
        for cell_id in unique_cells:
            cell_data = df[df['cell_id'] == cell_id].copy()
            cell_data = cell_data.sort_values('time_minutes').reset_index(drop=True)
            
            if len(cell_data) < 5:  # Skip batteries with too little data
                continue
            
            # Extract voltage and time
            voltage = cell_data['voltage_V'].values
            time_minutes = cell_data['time_minutes'].values
            
            # Calculate time differences (intervals between adjacent time points)
            time_diff = np.diff(time_minutes, prepend=0)
            
            # Handle time reset cases (when time difference is negative, indicating new cycle)
            for i in range(1, len(time_diff)):
                if time_diff[i] < 0:
                    time_diff[i] = time_minutes[i]  # Use absolute time
            
            # Ensure time differences are positive, avoid zero values
            time_diff = np.maximum(time_diff, 0.001)
            
            # Handle sequence length - random sampling
            if len(voltage) >= sequence_length:
                # Random sampling of sequence_length points (maintain time order)
                indices = np.sort(np.random.choice(len(voltage), sequence_length, replace=False))
                voltage_seq = voltage[indices]
                
                # For time_seq: first element is 0, others are time differences between adjacent selected points
                time_seq = np.zeros(sequence_length)
                time_seq[0] = 0  # First time difference is 0
                for i in range(1, sequence_length):
                    # Calculate time difference between current and previous selected points
                    time_seq[i] = time_minutes[indices[i]] - time_minutes[indices[i-1]]
                
                used_length = sequence_length
                is_truncated = len(voltage) > sequence_length
                sampling_info = "random_sampling"
            else:
                # Insufficient data, need padding
                voltage_seq = voltage
                time_seq = time_diff
                
                # Pad to sequence_length
                pad_length = sequence_length - len(voltage)
                voltage_seq = np.pad(voltage_seq, (0, pad_length), 'constant', 
                                   constant_values=voltage.mean())
                time_seq = np.pad(time_seq, (0, pad_length), 'constant', 
                                constant_values=time_diff.mean())
                used_length = len(voltage)
                is_truncated = False
                sampling_info = "padded"
            
            voltage_sequences.append(voltage_seq)
            time_sequences.append(time_seq)
            
            # Collect battery information
            info = {
                'cell_id': cell_id,
                'original_length': len(voltage),
                'used_length': used_length,
                'is_truncated': is_truncated,
                'sampling_method': sampling_info,
                'voltage_mean': np.mean(voltage),
                'voltage_std': np.std(voltage),
                'voltage_min': np.min(voltage),
                'voltage_max': np.max(voltage),
                'time_diff_mean': np.mean(time_diff),
                'time_diff_std': np.std(time_diff),
                'time_diff_min': np.min(time_diff),
                'time_diff_max': np.max(time_diff)
            }
            
            # Add other available information
            for col in ['label', 'value', 'zero_time_count', 'batch_name']:
                if col in cell_data.columns:
                    info[col] = cell_data[col].iloc[0]
            
            battery_info.append(info)
        
        if not voltage_sequences:
            print("Error: No valid battery data found")
            return None, None, None
        
        # Convert to numpy arrays
        voltage_data = np.array(voltage_sequences).reshape(-1, sequence_length, 1)
        time_data = np.array(time_sequences).reshape(-1, sequence_length, 1)
        print("w", voltage_data)
        print("l", time_data)
        
        return voltage_data, time_data, battery_info
    
    def train_tlstm_ae(self, voltage_data, time_data, epochs=200, batch_size=8, 
                       lr=0.001, model_save_path="tlstm_ae_model.pth"):
        """
        Train TLSTM_AE model
        
        Args:
            voltage_data: Voltage sequence data
            time_data: Time difference sequence data
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            model_save_path: Model save path
        """
        print(f"Training TLSTM_AE model with {len(voltage_data)} samples")
        
        # Data normalization
        voltage_flat = voltage_data.reshape(-1, 1)
        time_flat = time_data.reshape(-1, 1)
        
        voltage_scaled = self.voltage_scaler.fit_transform(voltage_flat).reshape(voltage_data.shape)
        time_scaled = self.time_scaler.fit_transform(time_flat).reshape(time_data.shape)
        
        # Convert to tensors
        voltage_tensor = torch.FloatTensor(voltage_scaled)
        time_tensor = torch.FloatTensor(time_scaled)
        
        # Optimizer and learning rate scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=15, factor=0.5, verbose=True
        )
        
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 30
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Randomly shuffle data
            indices = torch.randperm(len(voltage_tensor))
            
            self.model.train()
            for i in range(0, len(voltage_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_voltage = voltage_tensor[batch_indices]
                batch_time = time_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                try:
                    # Calculate reconstruction loss
                    recon_loss = self.model.get_reconstruction_loss(batch_voltage, batch_time)
                    
                    # Backpropagation
                    recon_loss.backward()
                    
                    # Gradient clipping to prevent gradient explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += recon_loss.item()
                    batch_count += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            if batch_count > 0:
                epoch_loss /= batch_count
                losses.append(epoch_loss)
                
                # Learning rate scheduling
                scheduler.step(epoch_loss)
                
                # Print training progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f} - Best: {best_loss:.6f} - LR: {current_lr:.2e} - Patience: {patience_counter}/{early_stop_patience}")
                
                # Early stopping check
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'voltage_scaler': self.voltage_scaler,
                        'time_scaler': self.time_scaler,
                        'config': self.config,
                        'epoch': epoch,
                        'loss': epoch_loss,
                        'losses': losses
                    }, model_save_path)
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        self.is_trained = True
        print(f"Training completed with best loss: {best_loss:.6f}")
        
        return losses
    
    def extract_features(self, voltage_data, time_data, battery_info):
        """
        Extract features using trained TLSTM_AE
        
        Args:
            voltage_data: Voltage sequence data
            time_data: Time difference sequence data
            battery_info: Battery information
            
        Returns:
            dict: Dictionary containing various features
        """
        if not self.is_trained:
            print("Error: Model not trained! Please train the model first.")
            return None
        
        # Data normalization
        voltage_flat = voltage_data.reshape(-1, 1)
        time_flat = time_data.reshape(-1, 1)
        
        voltage_scaled = self.voltage_scaler.transform(voltage_flat).reshape(voltage_data.shape)
        time_scaled = self.time_scaler.transform(time_flat).reshape(time_data.shape)
        
        voltage_tensor = torch.FloatTensor(voltage_scaled)
        time_tensor = torch.FloatTensor(time_scaled)
        
        self.model.eval()
        with torch.no_grad():
            # 1. Get encoder representation (main features)
            representation, decoder_initial = self.model.get_representation(voltage_tensor, time_tensor)
            encoder_features = representation.numpy()  # [batch, hidden_dim2]
            
            # 2. Get encoder layer states
            encoder1_states = self.model.get_encoder_states(voltage_tensor, time_tensor)
            encoder1_features = encoder1_states[:, -1, :].numpy()  # First layer last state
            
            encoder2_states, encoder2_cells = self.model.get_encoder2_states(voltage_tensor, time_tensor)
            encoder2_features = encoder2_states[:, -1, :].numpy()  # Second layer last state
            
            # 3. Get reconstruction results
            reconstructed = self.model(voltage_tensor, time_tensor)
            recon_data = reconstructed.numpy()
            
            # 4. Calculate reconstruction error
            recon_error = np.mean((voltage_scaled - recon_data) ** 2, axis=(1, 2))
            
            # 5. Calculate sequence-level statistical features
            seq_mean = np.mean(voltage_scaled, axis=1).flatten()
            seq_std = np.std(voltage_scaled, axis=1).flatten()
            seq_min = np.min(voltage_scaled, axis=1).flatten()
            seq_max = np.max(voltage_scaled, axis=1).flatten()
        
        print(f"Feature extraction completed. Main features shape: {encoder_features.shape}")
        
        return {
            'main_features': encoder_features,           # Main representation features
            'encoder1_features': encoder1_features,     # First encoder layer features
            'encoder2_features': encoder2_features,     # Second encoder layer features
            'reconstruction_error': recon_error,        # Reconstruction error
            'sequence_stats': {                         # Sequence statistical features
                'mean': seq_mean,
                'std': seq_std,
                'min': seq_min,
                'max': seq_max
            },
            'reconstructed_data': recon_data            # Reconstructed data
        }
    
    def save_features_to_csv(self, features, battery_info, output_file="tlstm_ae_features.csv"):
        """
        Save extracted features to CSV file - only keep main_features
        
        Args:
            features: Extracted features dictionary
            battery_info: Battery information
            output_file: Output file path
        """
        print(f"Saving main features to: {output_file}")
        
        results = []
        main_features = features['main_features']
        
        for i, info in enumerate(battery_info):
            row = info.copy()
            
            # Only keep main representation features (hidden_dim2 dimensions)
            for j in range(main_features.shape[1]):
                row[f'tlstm_ae_main_{j}'] = main_features[i, j]
            
            results.append(row)
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        return df
    
    def load_model(self, model_path="tlstm_ae_model.pth"):
        """Load trained model"""
        if not os.path.exists(model_path):
            return False
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.voltage_scaler = checkpoint['voltage_scaler']
        self.time_scaler = checkpoint['time_scaler']
        
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        self.is_trained = True
        
        return True

def main():
    """Main function - TLSTM_AE battery feature extraction"""
    # Check data files
    data_files = [
        "./OriginalData/charge.csv",
        # "./OriginalData/discharge.csv"
    ]
    
    available_files = [f for f in data_files if os.path.exists(f)]
    if not available_files:
        return
    
    # Use first available file
    input_file = available_files[0]
    
    # Create TLSTM_AE feature extractor
    extractor = BatteryTLSTMAEFeatureExtractor(
        input_dim=1,           # Voltage input
        output_dim=32,         # First encoder layer output
        output_dim2=16,        # Second encoder layer output
        output_dim3=1,         # Final decoder output
        hidden_dim=64,         # First layer hidden dimension
        hidden_dim2=32,        # Second layer hidden dimension (main features)
        hidden_dim3=16,        # Third layer hidden dimension
        dropout_rate=0.1
    )
    
    voltage_data, time_data, battery_info = extractor.load_battery_data(
        csv_file=input_file, 
        sequence_length=10
    )
    
    if voltage_data is None:
        return
    
    # Train TLSTM_AE model
    losses = extractor.train_tlstm_ae(
        voltage_data, time_data,
        epochs=50,
        batch_size=8,
        lr=0.001,
        model_save_path="output/tlstm_ae_battery_model.pth"
    )
    
    # Extract features
    features = extractor.extract_features(voltage_data, time_data, battery_info)
    
    if features is not None:
        # Save features
        df = extractor.save_features_to_csv(
            features, battery_info,
            "ExtractedData/charge.csv"
        )
        
        print("TLSTM_AE battery feature extraction completed successfully!")

if __name__ == "__main__":
    main()