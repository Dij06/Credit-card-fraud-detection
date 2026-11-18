"""
FinSecure - Application Root Entrypoint.
Enables launching via 'streamlit run app.py'.
"""
import runpy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute the dashboard script on every Streamlit rerun
dashboard_path = PROJECT_ROOT / "app" / "dashboard.py"
runpy.run_path(str(dashboard_path), run_name="__main__")
