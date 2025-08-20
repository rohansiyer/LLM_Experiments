"""Configuration parameters for RCG-Neuro experiment."""

# --- Model Settings ---
MODEL_ID = "unsloth/llama-3.1-8b-Instruct-bnb-4bit"
HF_HOME_CACHE = "D:/hf_cache"

# --- Experiment Parameters ---
NUM_ITERATIONS = 50
LOG_FILE_PREFIX = "v0.9_rcg_neuro"

# --- RCG Framework Parameters ---
# Incoherence weights (will be replaced by softmax in v1.0)
ALPHA_ENTROPY = 1.0
BETA_REPETITION = 1.0
GAMMA_ATTENTION = 1.0
DELTA_DYNAMISM = 1.0

# Collapse thresholds and parameters
COLLAPSE_THRESHOLD = 1.0    # Stress level to trigger a "reroute"
SNAP_THRESHOLD = 5.0        # Avg stress in a region to trigger a "snap"
SNAP_CANDIDATE_COUNT_THRESHOLD = 20  # Number of stress points to trigger snap instead of reroute
DELTA_PERCENTAGE = 0.01     # % of weight to transfer during a reroute
