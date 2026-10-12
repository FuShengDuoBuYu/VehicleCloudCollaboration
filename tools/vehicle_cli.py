"""Unified entry for the vehicle repository; importing it never opens devices."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from tools.process_lifecycle import managed_process

ROOT=Path(__file__).resolve().parents[1]
PROFILE='rosmaster_jetson_yolopv2_visual_feedback'
MOTOR_CONFIRMATION='I_UNDERSTAND_MOTORS_WILL_MOVE'
PERCEPTION='vehicle-perception-monitor.service'
SERIAL='vehicle-serial-monitor.service'
HELP={
    'dashboard':'Start HTTP dashboard (no device capture).',
    'panel':'Start session supervisor; hardware only on an authorized later Start.',
    'drive':'Capture/control run, motors disabled unless explicitly confirmed.',
    'monitor':'Start one collector: perception, serial, or sensors (uses devices).',
    'replay':'Offline replay; defaults to --mode recorded --profile recorded.',
    'trial':'Bounded physical trial then recorded replay; explicit motion confirmation required.',
    'wheel-check':'Wheel calibration; requires the tool\'s wheels-lifted confirmation.',
    'check':'Offline validation: --static, --torch, --loopback, --websocket-loopback.',
    'catalog':'Read-only evidence scan; --output must be a new directory.',
}


def runtime_python(root, environ):
    selected=environ.get('CAR_PYTHON')
    if selected:
        executable=shutil.which(selected)
        if executable is None:raise ValueError('CAR_PYTHON is not an executable interpreter')
        return executable
    project=root/'outputs/field_yolopv2/venv/bin/python'
    return str(project) if project.is_file() and os.access(str(project),os.X_OK) else sys.executable


def environment(root):
    env=dict(os.environ)
    env['PYTHONPATH']=str(root/'car')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def option(arguments, name, default=None):
    """argparse's last explicit value wins; support both option forms."""
    value=default
    for index,item in enumerate(arguments):
        if item==name:
            if index+1>=len(arguments):raise ValueError('missing value for '+name)
            value=arguments[index+1]
        elif item.startswith(name+'='):value=item.split('=',1)[1]
    return value


def drive_command(root, python, arguments):
    if '--enable-motors' in arguments:
        if option(arguments,'--confirm-motor-motion')!=MOTOR_CONFIRMATION:
            raise ValueError('motor operation requires --confirm-motor-motion '+MOTOR_CONFIRMATION)
    return [python,'-B',str(root/'car/autodrive/run_onboard.py'),
            '--config',str(root/'car/autodrive/config/onboard_runtime.yaml'),
            '--vehicle-profile',PROFILE,'--max-runtime-seconds','30',*arguments]


def show_help(command=None):
    if command is None:
        print('Usage: ./vehicle COMMAND [arguments]\n\nNo command starts nothing. Commands:')
        for name,description in HELP.items():print('  {:12s} {}'.format(name,description))
    else:
        print('Usage: ./vehicle {} [arguments]\n{}'.format(command,HELP[command]))
        if command=='drive':
            print('Default profile: '+PROFILE+'; default 30 seconds; motors disabled.\n'
                  'Options: --vehicle-profile ID --video PATH --output-dir PATH --max-samples N\n'
                  '         --max-runtime-seconds N --field-trial --enable-motors\n'
                  '         --confirm-motor-motion '+MOTOR_CONFIRMATION)
        elif command=='trial':
            print('Usage: ./vehicle trial SECONDS --confirm-motor-motion '+MOTOR_CONFIRMATION+'\n'
                  'SECONDS is 1..120; moves the vehicle, stops, then replays frozen inputs.')
        elif command=='wheel-check':
            print('Options: --confirm-wheels-lifted WHEELS_ARE_LIFTED --all-wheels\n'
                  '         --wheel N --pwm N --duration SECONDS --direction {forward,reverse}')
        elif command=='panel':print('Options: --authorize-motion '+MOTOR_CONFIRMATION)
        elif command=='replay':print('Options: --run PATH --output NEW_DIRECTORY --profile recorded --mode recorded')
        elif command=='monitor':print('Usage: ./vehicle monitor {perception|serial|sensors} [collector arguments]')
        elif command=='catalog':print('Usage: ./vehicle catalog --output NEW_DIRECTORY [--root REPO]')
        elif command=='dashboard':print('Options: --host ADDRESS --port PORT --runtime-dir PATH --sensor-dir PATH')
    return 0


def exec_runtime(command, root, env):
    os.chdir(root)
    os.execvpe(command[0],command,env)


def sensor_command(root):
    return ['docker','run','--rm','--name','vehicle-dashboard-sensors','--network','none',
        '--cap-drop','ALL','--cap-add','DAC_OVERRIDE','--device','/dev/rplidar:/dev/rplidar',
        '--device-cgroup-rule','c 189:* rmw','-v','/dev/bus/usb:/dev/bus/usb',
        '-v',str(root/'car')+':/workspace/car:ro',
        '-v',str(root/'outputs/vehicle_dashboard/sensors')+':/telemetry',
        '-e','PYTHONPATH=/workspace/car','-e','ROS_DOMAIN_ID=67',
        'yahboomtechnology/ros-foxy:4.0.5','/bin/bash','-c',
        'source /opt/ros/foxy/setup.bash; '
        'for setup in /root/yahboomcar_ros2_ws/software/library_ws/install/setup.bash; do '
        'if [ -f "$setup" ]; then source "$setup"; fi; done; '
        'exec python3 -B -m autodrive.web.sensor_publishers ros --output-dir /telemetry --start-drivers "$@"',
        'dashboard-sensors']


def run_trial(arguments,root,python,env):
    parser=argparse.ArgumentParser(prog='./vehicle trial',allow_abbrev=False)
    parser.add_argument('seconds',type=int)
    parser.add_argument('--confirm-motor-motion',required=True,choices=[MOTOR_CONFIRMATION])
    args=parser.parse_args(arguments)
    if not 1<=args.seconds<=120:parser.error('seconds must be between 1 and 120')
    started=time.time()
    drive=drive_command(root,python,['--field-trial','--enable-motors',
        '--confirm-motor-motion',args.confirm_motor_motion,'--max-runtime-seconds',str(args.seconds)])
    cancelled=False
    def mark_cancelled():
        nonlocal cancelled
        cancelled=True
    result=managed_process(drive,services=(PERCEPTION,SERIAL),cwd=root,env=env,
                           on_interrupt=mark_cancelled)
    if cancelled or result in (130,143):return result
    # Never hides drive failure. Replay itself checks final state and timestamp.
    replay=[python,'-B',str(root/'car/autodrive/tools/replay_outer_loop.py'),
        '--status',str(root/'outputs/onboard_runtime'/PROFILE/'status.json'),
        '--after',str(started),'--profile','recorded','--mode','recorded']
    replay_result=managed_process(replay,cwd=root,env=env)
    return result or replay_result


def main(argv=None, root=ROOT):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('--help','-h','help'):return show_help()
    command=args.pop(0)
    if command not in HELP:
        print('Unknown command: '+command,file=sys.stderr);return 2
    # Help is handled before interpreter selection, files, ports or services.
    if '--help'in args or '-h'in args or command=='wheel-check' and not args:
        return show_help(command)
    root=Path(root).resolve();env=environment(root)
    try:
        python=runtime_python(root,env)
        if command=='drive':
            child=drive_command(root,python,args)
            services=(PERCEPTION,SERIAL) if '--enable-motors'in args else (PERCEPTION,)
            return managed_process(child,services=services,cwd=root,env=env)
        if command=='trial':return run_trial(args,root,python,env)
        if command=='wheel-check':
            if option(args,'--confirm-wheels-lifted')!='WHEELS_ARE_LIFTED':
                raise ValueError('wheel-check requires --confirm-wheels-lifted WHEELS_ARE_LIFTED')
            child=[python,'-u','-B',str(root/'car/autodrive/tools/check_wheel_directions.py'),
                '--vehicle-profile','rosmaster_jetson_yolopv2',*args]
            base=root/'outputs/manual_wheel_check';base.mkdir(parents=True,exist_ok=True)
            folder=Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'),dir=str(base)))
            (folder/'command.json').write_text(json.dumps({'argv':child},indent=2)+'\n')
            print('Console log: '+str(folder/'console.log'),flush=True)
            with (folder/'console.log').open('x') as output:
                result=managed_process(child,services=(SERIAL,),cwd=root,env=env,output=output)
            (folder/'exit-code.txt').write_text(str(result)+'\n')
            return result
        if command=='dashboard':
            child=[python,'-B','-m','autodrive.web.dashboard',
                '--host',env.get('DASHBOARD_HOST','0.0.0.0'),'--port',env.get('DASHBOARD_PORT','8080'),
                '--runtime-dir',str(root/'outputs/vehicle_dashboard/monitor'),
                '--runtime-dir',str(root/'outputs/onboard_runtime/rosmaster_jetson_yolopv2'),
                '--runtime-dir',str(root/'outputs/onboard_runtime'),
                '--sensor-dir',str(root/'outputs/vehicle_dashboard/sensors'),
                '--session-socket',str(root/'outputs/vehicle_dashboard/control.sock'),*args]
        elif command=='panel':child=[python,'-B','-m','autodrive.web.drive_sessions',*args]
        elif command=='monitor':
            if not args or args[0]not in ('perception','serial','sensors'):
                raise ValueError('monitor requires perception, serial, or sensors')
            collector=args.pop(0)
            if collector=='sensors':
                (root/'outputs/vehicle_dashboard/sensors').mkdir(parents=True,exist_ok=True)
                child=sensor_command(root)+args
            elif collector=='serial':child=[python,'-B','-m','autodrive.web.sensor_publishers','serial',
                '--output-dir',str(root/'outputs/vehicle_dashboard/sensors'),*args]
            else:child=[python,'-B','-m','autodrive.web.perception_monitor',*args]
        elif command=='replay':child=[python,'-B',str(root/'car/autodrive/tools/replay_outer_loop.py'),
            '--mode','recorded','--profile','recorded',*args]
        elif command=='check':child=[python,'-B',str(root/'tools/check.py'),*args]
        else:child=[python,'-B','-m','experiments.catalog',*args]
        exec_runtime(child,root,env)
    except (ValueError,OSError) as error:
        print('vehicle: '+str(error),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
