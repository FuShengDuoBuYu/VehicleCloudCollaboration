import copy
import json
import base64
import hashlib
import unittest
from unittest.mock import patch
from car.test import test_cloud_scene_realtime as realtime_tests
from cloud_client.contracts import get_contract


def recovery():
    return dict(schema_version='road-recovery-v1', assessment='white_marking',
                recommendation='try_candidate', candidate_id='forward-white',
                uncertain=False, hazards=[], reason='当前白色路面标记')


class RecoveryContractTests(unittest.TestCase):
    setUp=realtime_tests.RealtimeTest.setUp
    client=realtime_tests.RealtimeTest.client
    def test_recovery_strict_and_fail_closed(self):
        validate = get_contract('road-recovery-v1')[2]
        self.assertEqual(validate(recovery()), recovery())
        for changes in ({'uncertain':1}, {'pwm':30}, {'hazards':['invented']},
                        {'reason':'x'*81}, {'recommendation':'hold'},
                        {'assessment':'unknown'}, {'hazards':['obstacle']}, {'uncertain':True}):
            value=recovery();value.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): validate(value)

    def test_three_frames_ordered_and_paced(self):
        client=self.client(contract='road-recovery-v1')
        sock=realtime_tests.FakeSocket(realtime_tests.events(recovery()))
        paths=[]
        for index,color in enumerate(('red','green','blue')):
            path=self.root/('%d.png'%index)
            realtime_tests.Image.new('RGB',(320,240),color).save(path);paths.append(path)
        with patch('cloud_client.realtime.websocket.create_connection',return_value=sock), \
             patch('cloud_client.realtime.time.sleep') as sleep:
            result=client.request_scene(paths, {'candidate_version':'v1'})
        self.assertEqual(len(result.input_manifest),3)
        self.assertEqual([e['type'] for e in sock.sent].count('input_image_buffer.append'),3)
        self.assertEqual(sleep.call_count,2)
        self.assertEqual(result.request_config['image_count'],3)
        self.assertEqual(result.request_config['synthetic_silence_ms'],2200)
        hashes=[hashlib.sha256(base64.b64decode(e['image'])).hexdigest()
                for e in sock.sent if e['type']=='input_image_buffer.append']
        self.assertEqual(len(set(hashes)),3)
        self.assertEqual(hashes,[item['sha256'] for item in result.input_manifest])
        self.assertEqual([item['path'] for item in result.input_manifest],[str(p.resolve()) for p in paths])
        with self.assertRaises(ValueError):client.build_payload([self.image]*4)


if __name__=='__main__': unittest.main()
