"""
FinSecure - Application Root Entrypoint.
Enables launching via 'streamlit run app.py'.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run dashboard module directly
if __name__ == "__main__" or True:
    import app.dashboard
