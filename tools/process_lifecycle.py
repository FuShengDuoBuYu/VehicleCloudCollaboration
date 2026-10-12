"""Own monitor leases and wait for child cleanup before restoring collectors."""
import codecs
import os
import signal
import subprocess
import sys
import threading


def managed_process(command, *, services=(), cwd=None, env=None, output=None,
                    on_interrupt=None):
    """Run an isolated process; restore only monitors active on entry.

    Signal forwarding happens once. A failed release prevents child startup;
    a failed restore is an error even if the child succeeded. No forced kill is
    used here: the existing runtime/supervisor owns actuator cleanup deadlines.
    """
    paused=[]
    child=None
    requested=False
    forwarded=False
    result=1
    pump=None
    output_errors=[]

    def interrupt(signum, frame):
        nonlocal requested, forwarded
        requested=True
        if on_interrupt is not None:on_interrupt()
        if child is not None and not forwarded:
            forwarded=True
            try:child.send_signal(signal.SIGINT)
            except ProcessLookupError:pass

    def forward_output():
        # read() on a buffered text stream can wait for a newline/full buffer.
        # Raw reads preserve input() prompts, including split UTF-8 characters.
        decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
        try:
            while True:
                chunk=os.read(child.stdout.fileno(),4096)
                text=decoder.decode(chunk,final=not chunk)
                output.write(text);output.flush()
                sys.stdout.write(text);sys.stdout.flush()
                if not chunk:break
        except (OSError,ValueError) as error:
            output_errors.append(error)
            # A lost console/log must not leave an interactive child running.
            interrupt(signal.SIGINT,None)
        finally:
            child.stdout.close()

    previous={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        for unit in services:
            if requested:break
            status=subprocess.run(['systemctl','is-active','--quiet',unit],env=env).returncode
            if status==3:continue
            if status!=0:raise RuntimeError('cannot determine monitor state: '+unit)
            # Set ownership before stop: an interrupted or partially failed stop
            # must still be followed by a restore attempt.
            paused.append(unit)
            subprocess.run(['sudo','-n','systemctl','stop',unit],env=env,check=True)
        if requested:
            result=130
        else:
            child=subprocess.Popen(command,cwd=cwd,env=env,start_new_session=True,
                stdout=subprocess.PIPE if output is not None else None,
                stderr=subprocess.STDOUT if output is not None else None)
            if output is not None:
                pump=threading.Thread(target=forward_output)
                pump.start()
            if requested:interrupt(signal.SIGINT,None)
            result=child.wait()
            if pump is not None:pump.join()
            if output_errors:raise RuntimeError('cannot forward console output: '+str(output_errors[0]))
            if result<0:result=128-result
    except (OSError,subprocess.SubprocessError,RuntimeError) as error:
        print('Launcher failed: '+str(error),file=sys.stderr)
        result=1
    finally:
        # Wait/reap is completed above before these services can open devices.
        for unit in paused:
            try:
                subprocess.run(['sudo','-n','systemctl','start',unit],env=env,check=True)
            except (OSError,subprocess.SubprocessError):
                print('FAILED to restore monitor: '+unit,file=sys.stderr)
                if result==0:result=1
        for sig,handler in previous.items():signal.signal(sig,handler)
    return 130 if requested and result==0 else result
