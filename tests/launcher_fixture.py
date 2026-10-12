"""Copy only launch code into a temporary fake-device environment."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def copy_launcher(root):
    shutil.copy2(ROOT/'vehicle',root/'vehicle')
    shutil.copytree(ROOT/'tools',root/'tools',dirs_exist_ok=True)
    for name in ('run_vehicle_dashboard.sh','run_autodrive_panel.sh','run_vehicle_experiment.sh'):
        shutil.copy2(ROOT/name,root/name)
    return root/'vehicle'
