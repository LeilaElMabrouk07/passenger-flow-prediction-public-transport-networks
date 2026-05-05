from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

YEAR_DIR_GLOB = "NUMBAT_20??"

PROCESSED_PARQUET = CACHE_DIR / "link_loads_long.parquet"
MODEL_PATH = CACHE_DIR / "stgcn_model.pt"
NORM_PATH = CACHE_DIR / "stgcn_norm.npz"

TRAIN_YEARS = list(range(2016, 2023))  
VAL_YEAR = 2023
TEST_YEAR = 2024

# Excel sheets
SHEET_CANDIDATES = ["Link Loads", "Link_Loads"]
SKIPROWS = 2

# Forecasting setup
N_HISTORY = 8       # fenêtre d'entrée (ex: 8 = 2h pour 15min)
HORIZON = 4        # prédire t+1 
STRIDE = 1          # pas de glissement

# STGCN model hyperparams
IN_CHANNELS = 1
HIDDEN_CHANNELS = 32
KT = 3              # kernel temporel
DROPOUT = 0.1

# Training
BATCH_SIZE = 16    
EPOCHS = 30
LR = 5e-4


RESULTS_CSV = CACHE_DIR / "training_metrics_stgcn.csv"


