from pathlib import Path
import matplotlib.pyplot as plt
# Path configurations
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Dataset paths
MADAR_PATH = DATA_DIR / "MADAR"
QADI_PATH = DATA_DIR / "QADI"

# Preprocessing constants
ARABIC_STOPWORDS = set([
    'في', 'من', 'إلى', 'على', 'أن', 'ما', 'هذا', 'هذه', 'ذلك', 'كان',
    'يكون', 'مع', 'هو', 'هي', 'هم', 'التي', 'الذي', 'عن', 'ليس', 'إذا'
])

# Visualization constants
PLOT_STYLE = 'whitegrid'
COLOR_PALETTE = "husl"