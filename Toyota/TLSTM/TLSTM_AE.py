import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class T_LSTM_AE(nn.Module):
    def __init__(self, input_dim, output_dim, output_dim2, output_dim3, 
                 hidden_dim, hidden_dim2, hidden_dim3, dropout_rate=0.0):
        super(T_LSTM_AE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim2 = output_dim2
        self.output_dim3 = output_dim3
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.dropout_rate = dropout_rate
        
        # Encoder parameters (First layer)
        self.Wi_enc = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Ui_enc = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bi_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wf_enc = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uf_enc = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bf_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wog_enc = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uog_enc = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bog_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        self.Wc_enc = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uc_enc = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bc_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_decomp_enc = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_decomp_enc = nn.Parameter(torch.zeros(hidden_dim))

        # Encoder parameters (Second layer)
        self.Wi_enc2 = nn.Parameter(torch.Tensor(output_dim, hidden_dim2))
        self.Ui_enc2 = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bi_enc2 = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.Wf_enc2 = nn.Parameter(torch.Tensor(output_dim, hidden_dim2))
        self.Uf_enc2 = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bf_enc2 = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.Wog_enc2 = nn.Parameter(torch.Tensor(output_dim, hidden_dim2))
        self.Uog_enc2 = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bog_enc2 = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.Wc_enc2 = nn.Parameter(torch.Tensor(output_dim, hidden_dim2))
        self.Uc_enc2 = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bc_enc2 = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.W_decomp_enc2 = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.b_decomp_enc2 = nn.Parameter(torch.zeros(hidden_dim2))

        # Decoder parameters (First layer)
        self.Wi_dec = nn.Parameter(torch.Tensor(input_dim, hidden_dim2))
        self.Ui_dec = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bi_dec = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.Wf_dec = nn.Parameter(torch.Tensor(input_dim, hidden_dim2))
        self.Uf_dec = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bf_dec = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.Wog_dec = nn.Parameter(torch.Tensor(input_dim, hidden_dim2))
        self.Uog_dec = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bog_dec = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.Wc_dec = nn.Parameter(torch.Tensor(input_dim, hidden_dim2))
        self.Uc_dec = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.bc_dec = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.W_decomp_dec = nn.Parameter(torch.Tensor(hidden_dim2, hidden_dim2))
        self.b_decomp_dec = nn.Parameter(torch.zeros(hidden_dim2))

        # Decoder parameters (Second layer)
        self.Wi_dec2 = nn.Parameter(torch.Tensor(output_dim2, hidden_dim3))
        self.Ui_dec2 = nn.Parameter(torch.Tensor(hidden_dim3, hidden_dim3))
        self.bi_dec2 = nn.Parameter(torch.zeros(hidden_dim3))
        
        self.Wf_dec2 = nn.Parameter(torch.Tensor(output_dim2, hidden_dim3))
        self.Uf_dec2 = nn.Parameter(torch.Tensor(hidden_dim3, hidden_dim3))
        self.bf_dec2 = nn.Parameter(torch.zeros(hidden_dim3))
        
        self.Wog_dec2 = nn.Parameter(torch.Tensor(output_dim2, hidden_dim3))
        self.Uog_dec2 = nn.Parameter(torch.Tensor(hidden_dim3, hidden_dim3))
        self.bog_dec2 = nn.Parameter(torch.zeros(hidden_dim3))
        
        self.Wc_dec2 = nn.Parameter(torch.Tensor(output_dim2, hidden_dim3))
        self.Uc_dec2 = nn.Parameter(torch.Tensor(hidden_dim3, hidden_dim3))
        self.bc_dec2 = nn.Parameter(torch.zeros(hidden_dim3))
        
        self.W_decomp_dec2 = nn.Parameter(torch.Tensor(hidden_dim3, hidden_dim3))
        self.b_decomp_dec2 = nn.Parameter(torch.zeros(hidden_dim3))

        # Output layers
        self.Wo = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.bo = nn.Parameter(torch.zeros(output_dim))
        self.Wo2 = nn.Parameter(torch.Tensor(hidden_dim2, output_dim2))
        self.bo2 = nn.Parameter(torch.zeros(output_dim2))
        self.Wo3 = nn.Parameter(torch.Tensor(hidden_dim3, output_dim3))
        self.bo3 = nn.Parameter(torch.zeros(output_dim3))

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize parameters
        self.init_parameters()

    def init_parameters(self):
        """Xavier initialization for all parameters"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def map_elapse_time(self, t, dim):
        """Map elapsed time using logarithmic transformation"""
        c1, c2 = 1.0, 2.7183
        T = c1 / torch.log(t + c2)
        return T.expand(-1, dim)

    def T_LSTM_Encoder_Unit(self, prev_hidden_memory, concat_input):
        """First layer encoder unit with time-aware decay"""
        h_prev, c_prev = prev_hidden_memory
        t = concat_input[:, 0:1]
        x = concat_input[:, 1:]
        
        # Time-aware cell state decay
        T = self.map_elapse_time(t, self.hidden_dim)
        C_ST = torch.sigmoid(c_prev @ self.W_decomp_enc + self.b_decomp_enc)
        c_prev = c_prev - C_ST + T * C_ST
        
        # LSTM gates computation
        i = torch.sigmoid(x @ self.Wi_enc + h_prev @ self.Ui_enc + self.bi_enc)
        f = torch.sigmoid(x @ self.Wf_enc + h_prev @ self.Uf_enc + self.bf_enc)
        o = torch.sigmoid(x @ self.Wog_enc + h_prev @ self.Uog_enc + self.bog_enc)
        C = torch.tanh(x @ self.Wc_enc + h_prev @ self.Uc_enc + self.bc_enc)
        
        Ct = f * c_prev + i * C
        ht = o * torch.tanh(Ct)
        
        return ht, Ct

    def T_LSTM_Encoder_Unit2(self, prev_hidden_memory, concat_input):
        """Second layer encoder unit"""
        h_prev, c_prev = prev_hidden_memory
        t = concat_input[:, 0:1]
        x = concat_input[:, 1:]
        
        T = self.map_elapse_time(t, self.hidden_dim2)
        C_ST = torch.sigmoid(c_prev @ self.W_decomp_enc2 + self.b_decomp_enc2)
        c_prev = c_prev - C_ST + T * C_ST
        
        i = torch.sigmoid(x @ self.Wi_enc2 + h_prev @ self.Ui_enc2 + self.bi_enc2)
        f = torch.sigmoid(x @ self.Wf_enc2 + h_prev @ self.Uf_enc2 + self.bf_enc2)
        o = torch.sigmoid(x @ self.Wog_enc2 + h_prev @ self.Uog_enc2 + self.bog_enc2)
        C = torch.tanh(x @ self.Wc_enc2 + h_prev @ self.Uc_enc2 + self.bc_enc2)
        
        Ct = f * c_prev + i * C
        ht = o * torch.tanh(Ct)
        
        return ht, Ct

    def T_LSTM_Decoder_Unit(self, prev_hidden_memory, concat_input):
        """First layer decoder unit"""
        h_prev, c_prev = prev_hidden_memory
        t = concat_input[:, 0:1]
        x = concat_input[:, 1:]
        
        T = self.map_elapse_time(t, self.hidden_dim2)
        C_ST = torch.sigmoid(c_prev @ self.W_decomp_dec + self.b_decomp_dec)
        c_prev = c_prev - C_ST + T * C_ST
        
        i = torch.sigmoid(x @ self.Wi_dec + h_prev @ self.Ui_dec + self.bi_dec)
        f = torch.sigmoid(x @ self.Wf_dec + h_prev @ self.Uf_dec + self.bf_dec)
        o = torch.sigmoid(x @ self.Wog_dec + h_prev @ self.Uog_dec + self.bog_dec)
        C = torch.tanh(x @ self.Wc_dec + h_prev @ self.Uc_dec + self.bc_dec)
        
        Ct = f * c_prev + i * C
        ht = o * torch.tanh(Ct)
        
        return ht, Ct

    def T_LSTM_Decoder_Unit2(self, prev_hidden_memory, concat_input):
        """Second layer decoder unit"""
        h_prev, c_prev = prev_hidden_memory
        t = concat_input[:, 0:1]
        x = concat_input[:, 1:]
        
        T = self.map_elapse_time(t, self.hidden_dim3)
        C_ST = torch.sigmoid(c_prev @ self.W_decomp_dec2 + self.b_decomp_dec2)
        c_prev = c_prev - C_ST + T * C_ST
        
        i = torch.sigmoid(x @ self.Wi_dec2 + h_prev @ self.Ui_dec2 + self.bi_dec2)
        f = torch.sigmoid(x @ self.Wf_dec2 + h_prev @ self.Uf_dec2 + self.bf_dec2)
        o = torch.sigmoid(x @ self.Wog_dec2 + h_prev @ self.Uog_dec2 + self.bog_dec2)
        C = torch.tanh(x @ self.Wc_dec2 + h_prev @ self.Uc_dec2 + self.bc_dec2)
        
        Ct = f * c_prev + i * C
        ht = o * torch.tanh(Ct)
        
        return ht, Ct

    def get_encoder_states(self, input_seq, time_seq):
        """Get all encoder states from the first layer"""
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device
        
        h_enc = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_enc = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        all_encoder_states = []
        for t in range(seq_len):
            concat_input = torch.cat([time_seq[:, t, :], input_seq[:, t]], dim=1)
            h_enc, c_enc = self.T_LSTM_Encoder_Unit((h_enc, c_enc), concat_input)
            all_encoder_states.append(h_enc)
        
        return torch.stack(all_encoder_states, dim=1)  # [batch, seq, hidden_dim]

    def get_encoder2_states(self, input_seq, time_seq):
        """Get encoder states from both layers"""
        # First layer
        encoder1_states = self.get_encoder_states(input_seq, time_seq)
        
        # Second layer
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device
        
        h_enc2 = torch.zeros(batch_size, self.hidden_dim2, device=device)
        c_enc2 = torch.zeros(batch_size, self.hidden_dim2, device=device)
        
        all_encoder2_states = []
        all_encoder2_cells = []
        
        for t in range(seq_len):
            # Get output from first layer
            encoder1_output = encoder1_states[:, t] @ self.Wo + self.bo
            time_part = time_seq[:, t, :]  # shape: [batch, 1]
            concat_input = torch.cat([time_part, encoder1_output], dim=1)  # shape: [batch, 1+output_dim]

            h_enc2, c_enc2 = self.T_LSTM_Encoder_Unit2((h_enc2, c_enc2), concat_input)
            all_encoder2_states.append(h_enc2)
            all_encoder2_cells.append(c_enc2)
        
        return (torch.stack(all_encoder2_states, dim=1), 
                torch.stack(all_encoder2_cells, dim=1))

    def get_representation(self, input_seq, time_seq):
        """Get the final representation from encoder"""
        all_encoder2_states, all_encoder2_cells = self.get_encoder2_states(input_seq, time_seq)
        
        # Use the last hidden state as representation
        representation = all_encoder2_states[:, -1, :]  # [batch, hidden_dim2]
        decoder_initial_cell = all_encoder2_cells[:, -1, :]
        
        return representation, decoder_initial_cell

    def forward(self, input_seq, time_seq, return_representation=False):
        """Forward pass through encoder-decoder"""
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device
        
        # Get representation from encoder
        representation, decoder_initial_cell = self.get_representation(input_seq, time_seq)
        
        if return_representation:
            return representation
        
        # Decoder
        h_dec = representation
        c_dec = decoder_initial_cell
        
        # First layer decoder
        decoder1_outputs = []
        for t in reversed(range(seq_len)):
            # Use zero input for autoencoder training
            dummy_input = torch.zeros(batch_size, self.input_dim, device=device)
            concat_input = torch.cat([time_seq[:, t, :], dummy_input], dim=1)
            h_dec, c_dec = self.T_LSTM_Decoder_Unit((h_dec, c_dec), concat_input)
            decoder1_outputs.append(h_dec @ self.Wo2 + self.bo2)
        
        # Second layer decoder
        h_dec2 = torch.zeros(batch_size, self.hidden_dim3, device=device)
        c_dec2 = torch.zeros(batch_size, self.hidden_dim3, device=device)
        
        final_outputs = []
        for t, dec1_output in enumerate(reversed(decoder1_outputs)):
            time_idx = seq_len - 1 - t
            time_part = time_seq[:, time_idx, :]  # [batch, 1]
            concat_input = torch.cat([time_part, dec1_output], dim=1)
            h_dec2, c_dec2 = self.T_LSTM_Decoder_Unit2((h_dec2, c_dec2), concat_input)
            final_outputs.append(h_dec2 @ self.Wo3 + self.bo3)
        
        # Reverse to get correct temporal order
        reconstructions = torch.stack(final_outputs[::-1], dim=1)
        
        return reconstructions

    def get_reconstruction_loss(self, input_seq, time_seq):
        """Compute reconstruction loss"""
        reconstructions = self.forward(input_seq, time_seq)
        return F.mse_loss(reconstructions, input_seq)