import sys,unittest,tempfile,time,threading,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.control.cloud_arbitration import CloudArbitrationConfig
from autodrive.control.cloud_worker import CloudCoordinator
from autodrive.control.lane_centering import DifferentialDriveCommand
from car.test.test_cloud_arbitration import scene,observation

class Network:
    def __init__(self,fail=False):
        self.started=threading.Event();self.release=threading.Event();self.fail=fail;self.paths=[]
        self.last_request_metadata={'provider':'test-only','request_id':'mock-network','timings_ms':{'http':2},'response_text':'must-not-archive'}
    def request_scene(self,paths,context):
        self.last_request_metadata=dict(self.last_request_metadata)
        self.paths=list(paths);self.started.set();self.release.wait(2)
        if self.fail:raise TimeoutError('injected network timeout')
        return SimpleNamespace(scene=scene(),response_model='test-only',request_id='mock-network',provider='test-only',requested_model='mock',schema_version='road-scene-v1',prompt_version='test-v1',timings_ms={'http':2},input_manifest=[{'sha256':'test-hash'}])

class WorkerTests(unittest.TestCase):
    def exercise(self,fail=False):
        root=tempfile.TemporaryDirectory();self.addCleanup(root.cleanup)
        network=Network(fail);self.addCleanup(network.release.set)
        coordinator=CloudCoordinator(CloudArbitrationConfig(enabled=True,trigger_observations=1,resume_observations=2),network,Path(root.name))
        self.addCleanup(coordinator.close)
        frame=observation(1,0);frame['frame']=np.full((32,32,3),37,np.uint8)
        command=DifferentialDriveCommand('forward',0,.1,.1,.9,'ok')
        t=time.monotonic();result=coordinator.filter(command,frame,False,'obstacle',now=0)
        self.assertLess(time.monotonic()-t,.1)
        self.assertEqual(result.action,'stop');self.assertTrue(network.started.wait(1))
        network.release.set()
        until=time.monotonic()+2
        sequence=2
        while time.monotonic()<until:
            obs=observation(sequence,.1+sequence/1000);obs['frame']=frame['frame']
            result=coordinator.filter(command,obs,True,'ok',now=.1+sequence/1000)
            sequence+=1
            if coordinator.get_state()['phase']!='pending':break
            time.sleep(.01)
        if fail:self.assertEqual(coordinator.get_state()['phase'],'hold')
        else:
            for i in range(sequence,sequence+3):
                obs=observation(i,.1+i/1000);obs['frame']=frame['frame']
                result=coordinator.filter(command,obs,True,'ok',now=.1+i/1000)
            self.assertEqual(result.action,'forward')
        self.assertEqual(len(list(Path(root.name).glob('*/input.png'))),1)
        self.assertEqual(len(list(Path(root.name).glob('*/response.json'))),1)
        result=json.loads(next(Path(root.name).glob('*/response.json')).read_text())
        self.assertEqual(result['metadata']['provider'],'test-only')
        self.assertEqual(result['metadata']['timings_ms']['http'],2)
        self.assertNotIn('response_text',result['metadata'])
        self.assertEqual(len(result['input_sha256']),64)
    def test_slow_network_is_nonblocking_and_binds_image(self):self.exercise()
    def test_worker_error_is_stopped_and_recorded(self):self.exercise(True)
    def test_close_during_request_never_releases_motion(self):
        with tempfile.TemporaryDirectory() as root:
            network=Network()
            c=CloudCoordinator(CloudArbitrationConfig(enabled=True,trigger_observations=1),network,Path(root))
            obs=observation(1,time.monotonic());obs['frame']=np.zeros((32,32,3),np.uint8)
            command=DifferentialDriveCommand('forward',0,.1,.1,.9,'ok')
            c.filter(command,obs,False,'obstacle')
            self.assertTrue(network.started.wait(1));c.close();network.release.set()
            c._thread.join(2)
            self.assertEqual(c.filter(command,obs,True,'ok').action,'stop')
            self.assertFalse(c._thread.is_alive())

    def test_evidence_write_failure_keeps_stop_and_skips_network(self):
        with tempfile.TemporaryDirectory() as root, patch('autodrive.control.cloud_worker.cv2.imwrite',return_value=False):
            network=Network()
            c=CloudCoordinator(CloudArbitrationConfig(enabled=True,trigger_observations=1),network,Path(root))
            try:
                obs=observation(1,time.monotonic());obs['frame']=np.zeros((32,32,3),np.uint8)
                command=DifferentialDriveCommand('forward',0,.1,.1,.9,'ok')
                c.filter(command,obs,False,'obstacle')
                deadline=time.monotonic()+1
                while time.monotonic()<deadline and c.get_state()['phase']=='pending':
                    result=c.filter(command,None,False,'no observation');time.sleep(.01)
                self.assertEqual(result.action,'stop');self.assertEqual(c.get_state()['phase'],'hold')
                self.assertFalse(network.started.is_set())
                response=json.loads(next(Path(root).glob('*/response.json')).read_text())
                self.assertEqual(response['metadata'],{})
            finally:c.close()

    def test_disabled_has_no_network_or_background_thread(self):
        with tempfile.TemporaryDirectory() as root:
            network=Network();c=CloudCoordinator(CloudArbitrationConfig(),network,Path(root))
            try:
                command=DifferentialDriveCommand('forward',0,.1,.1,.9,'ok')
                self.assertEqual(c.filter(command,None,True,'ok'),command)
                self.assertFalse(network.started.is_set())
            finally:c.close()

if __name__=='__main__':unittest.main()
