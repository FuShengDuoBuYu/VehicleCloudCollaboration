"""Start existing YOLOPv2 runtime with enforced observation-only configuration."""
import argparse
import os
from pathlib import Path
import sys
import yaml


CONTROL_DIR = Path(__file__).resolve().parents[2] / 'control'
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))
from vehicle_control.profile import load_vehicle_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vehicle-profile', default='rosmaster_jetson_yolopv2',
                        help='Vehicle profile ID or YAML path to preview with motors disabled')
    args = parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    source_path, profile = load_vehicle_profile(args.vehicle_profile, relative_to=Path.cwd())
    profile['preview_source'] = {'id': profile['id'], 'path': str(source_path)}
    profile.pop('profile', None)
    profile.pop('profile_path', None)
    profile['id']='rosmaster_jetson_dashboard_monitor'
    profile.setdefault('status',{})['motion_calibrated']=False
    overrides=profile.setdefault('runtime_overrides',{})
    overrides.setdefault('camera',{}).setdefault('gimbal',{})['initialize_on_startup']=False
    overrides.setdefault('cloud_arbitration',{})['enabled']=False
    overrides.setdefault('runtime',{})['archive_runs']=False
    overrides['runtime']['live_update_hz']=10
    folder=root/'outputs/vehicle_dashboard/monitor';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'monitor-profile.yaml'
    path.write_text(yaml.safe_dump(profile,allow_unicode=True))
    os.chdir(str(root))
    os.execv(sys.executable,[sys.executable,str(root/'car/autodrive/run_onboard.py'),
        '--config',str(root/'car/autodrive/config/onboard_runtime.yaml'),'--vehicle-profile',str(path),
        '--output-dir',str(folder),'--max-runtime-seconds','1800','--no-run-archive'])

if __name__=='__main__':main()
