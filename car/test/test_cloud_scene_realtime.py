"""Realtime contract/fault regression: synthetic credentials, no paid inference."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import websocket

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud_client import CloudClient, CloudConfig, CloudAPIError


def observation():
    return {'features':['straight_arrow','right_arrow'],'uncertain':False,'advice':'observe','reason':'直行或右转'}


def events(scene=None):
    text=json.dumps(scene if scene is not None else observation(),ensure_ascii=False)
    return [
        {'type':'session.created','session':{'id':'s'}}, {'type':'session.updated','session':{'id':'s'}},
        {'type':'input_audio_buffer.committed','item_id':'input1'},
        {'type':'response.created','response':{'id':'r','status':'in_progress'}},
        {'type':'response.text.delta','response_id':'r','item_id':'i','output_index':0,'content_index':0,'delta':text},
        {'type':'response.text.done','response_id':'r','item_id':'i','output_index':0,'content_index':0,'text':text},
        {'type':'response.done','response':{'id':'r','status':'completed','output':[
            {'id':'i','type':'message','role':'assistant','content':[{'type':'text','text':text}]}],
            'usage':{'input_tokens':400,'output_tokens':40,'input_tokens_details':{'audio_tokens':7}}}}]


class FakeSocket:
    def __init__(self, values): self.values=list(values);self.sent=[];self.closed=False
    def settimeout(self,value): self.timeout=value
    def send(self,value): self.sent.append(json.loads(value))
    def recv_data(self,control_frame=True):
        if not self.values: raise websocket.WebSocketTimeoutException('synthetic timeout')
        value=self.values.pop(0)
        if isinstance(value,Exception): raise value
        if isinstance(value,tuple): return value
        return websocket.ABNF.OPCODE_TEXT,json.dumps(value,ensure_ascii=False).encode()
    def close(self,timeout=None): self.closed=True


class RealtimeTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.env=self.root/'empty.env';self.env.write_text('')
        self.image=self.root/'frame.png';Image.new('RGB',(640,480),(80,80,80)).save(self.image)
    def client(self,**overrides):
        settings={'api_key':'synthetic-key','workspace_id':'llm-test'};settings.update(overrides)
        with patch.dict(os.environ,{},clear=True):
            return CloudClient(env_file=self.env,**settings)
    def request(self,values=None,client=None,context=None):
        sock=FakeSocket(events() if values is None else values)
        with patch('cloud_client.realtime.websocket.create_connection',return_value=sock) as connect:
            result=(client or self.client()).request_scene(self.image,context)
        return result,sock,connect
    def test_default_is_realtime_and_no_two_second_cutoff(self):
        with patch.dict(os.environ,{},clear=True): config=CloudConfig.from_env(self.env)
        self.assertEqual(config.provider,'qwen-realtime')
        self.assertEqual(config.model,'qwen3.8-omni-flash-realtime')
        self.assertEqual(config.timeout,30)
        self.assertEqual(config.contract,'road-observation-fast-v1')
    def test_new_session_single_jpeg_text_only_and_result_identity(self):
        result,sock,connect=self.request(context={'event_id':'e','frame_id':'f','candidate_version':'cv'})
        self.assertEqual(result.scene,observation());self.assertEqual(result.schema_version,'road-observation-fast-v1')
        self.assertEqual(result.context['event_id'],'e');self.assertEqual(result.response_id,'r')
        self.assertEqual(result.raw_response['session_id'],'s');self.assertTrue(sock.closed)
        self.assertIn('model=qwen3.8-omni-flash-realtime',connect.call_args.args[0])
        self.assertEqual(connect.call_args.kwargs['redirect_limit'],0)
        kinds=[x['type'] for x in sock.sent]
        self.assertEqual(kinds,['session.update','input_audio_buffer.append','input_image_buffer.append','input_audio_buffer.commit','response.create'])
        self.assertEqual(sock.sent[0]['session']['modalities'],['text'])
        self.assertEqual(result.input_manifest[0]['mime_type'],'image/jpeg')
    def test_valid_response_after_two_seconds_is_accepted(self):
        now=[0.0];values=events();sock=FakeSocket(values);original=sock.recv_data
        def receive(*args,**kwargs):
            now[0]+=.5;return original(*args,**kwargs)
        sock.recv_data=receive
        with patch('cloud_client.realtime.websocket.create_connection',return_value=sock),patch('time.monotonic',side_effect=lambda:now[0]):
            result=self.client().request_scene(self.image)
        self.assertGreater(result.timings_ms['total'],2000)
        self.assertEqual(result.scene['advice'],'observe')
    def test_rejects_wrong_response_id_final_text_incomplete_and_close(self):
        variants=[]
        v=events();v[4]['response_id']='wrong';variants.append(v)
        v=events();v[-1]['response']['output'][0]['content'][0]['text']='{}';variants.append(v)
        v=events();v[-1]['response']['status']='failed';variants.append(v)
        v=events();v[5]['text']='{}';variants.append(v)
        v=events();v[-1]['response']['id']='wrong';variants.append(v)
        variants.append(events()[:-1])
        variants.append([(websocket.ABNF.OPCODE_CLOSE,b'\x03\xe8')])
        for values in variants:
            with self.subTest(values=values),self.assertRaises(CloudAPIError): self.request(values)
    def test_api_error_safe_and_no_retry(self):
        sock=FakeSocket([{'type':'error','error':{'message':'synthetic-key'}}]);client=self.client()
        with patch('cloud_client.realtime.websocket.create_connection',return_value=sock) as connect:
            with self.assertRaises(CloudAPIError) as caught: client.request_scene(self.image)
        self.assertNotIn('synthetic-key',str(caught.exception));self.assertEqual(connect.call_count,1)
        self.assertNotIn('synthetic-key',json.dumps(client.last_request_metadata))
    def test_rejects_extra_fields_duplicate_features_overlong_reason(self):
        for updates in [{'pwm':40},{'features':['blur','blur']},{'reason':'😀'*11},{'uncertain':1},{'advice':'resume'}]:
            value=observation();value.update(updates)
            with self.subTest(updates=updates),self.assertRaises(CloudAPIError): self.request(events(value))
    def test_missing_workspace_and_multiple_images_fail_before_network(self):
        for client,images in [(self.client(workspace_id=''),self.image),(self.client(),[self.image,self.image])]:
            with patch('cloud_client.realtime.websocket.create_connection') as connect:
                with self.assertRaises(ValueError): client.request_scene(images)
                connect.assert_not_called()
    def test_http_observation_same_contract(self):
        client=self.client(provider='qwen',model='qwen3.8-omni-flash',contract='road-observation-fast-v1')
        envelope={'id':'http-r','model':client.model,'usage':{},'choices':[{'finish_reason':'stop','message':{'content':json.dumps(observation())}}]}
        self.assertEqual(client.parse_response(json.dumps(envelope)).scene,observation())
    def test_malformed_protocol_objects_fail_safely(self):
        variants=[]
        value=events();value[3]['response']=None;variants.append(value)
        value=events();value[-1]['response']['output'][0]['content']=['bad'];variants.append(value)
        value=events();value[-1]['response']=[];variants.append(value)
        value=events();value[4]['output_index']=False;variants.append(value)
        value=events();value[0]['session']=None;variants.append(value)
        for values in variants:
            with self.subTest(values=values),self.assertRaises(CloudAPIError): self.request(values)


if __name__=='__main__': unittest.main()
