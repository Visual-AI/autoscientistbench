# Toggle INR mode
USE_INR = True

# Random seed
SEED = 42

# Model architecture
HIDDEN_SIZE = 32
DECODER_LAYERS = [256, 256]
NUM_LSTM_LAYERS = 2
DROPOUT = 0.1

# Sensor configuration
N_SENSORS = 64
LAGS = 5
SENSOR_STRATEGY = 'random'

# Training hyperparameters
SHRED_EPOCHS = 800
SHRED_PATIENCE = 30
DASHRED_EPOCHS = 1500
DASHRED_PATIENCE = 50
GAN_EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-4

# Frequency sparsity
LAMBDA_SPARSE = 0.05
MAX_TARGET_FREQ = 16
HF_WARMUP = 100

# Hierarchical peeling
N_PEEL_LAYERS = 2
PEEL_CONFIG = [
    {'name': 'HF_Peel1', 'target_k': 3, 'epochs': 500,
     'lambda_sparse': 0.05, 'finetune_epochs': 200, 'finetune_lambda': 0.005},
    {'name': 'HF_Peel2', 'target_k': None, 'epochs': 500,
     'lambda_sparse': 0.05, 'finetune_epochs': 200, 'finetune_lambda': 0.005},
]

# INR configuration
INR_CONFIG = {
    'pe_num_frequencies': 16, 'pe_max_frequency': 8.0, 'pe_include_input': True,
    'encoder_hidden': [128, 128], 'latent_dim': 64,
    'decoder_hidden': [256, 256, 128], 'activation': 'relu', 'omega_0': 30.0,
    'lambda_smooth': 0.1, 'smooth_type': 'laplacian',
}


