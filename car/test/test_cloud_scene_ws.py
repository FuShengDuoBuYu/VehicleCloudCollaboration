"""Real RFC6455 loopback handshake + CLI; synthetic responses, no paid API."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from PIL import Image
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud_client import CloudClient
from cloud_client.cli import main
from test_cloud_scene_realtime import events


class WebSocketLoopbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.image=self.root/'f.png';Image.new('RGB',(640,480)).save(self.image)
        self.env=self.root/'empty.env';self.env.write_text('');self.requests=[];self.headers=[];self.errors=[];self.bad=False
        def handler(conn):
            try:
                self.headers.append(conn.request.headers.get('Authorization'))
                values=events();conn.send(json.dumps(values[0]));self.requests.append(json.loads(conn.recv()))
                conn.send(json.dumps(values[1]))
                for _ in range(3): self.requests.append(json.loads(conn.recv()))
                conn.send(json.dumps(values[2]));self.requests.append(json.loads(conn.recv()))
                if self.bad: values[-1]['response']['output'][0]['content'][0]['text']='wrong'
                for event in values[3:]: conn.send(json.dumps(event))
            except ConnectionClosed: pass
            except Exception as error: self.errors.append(type(error).__name__)
        self.server=serve(handler,'127.0.0.1',0,compression=None,ping_interval=None,close_timeout=.2)
        self.url=f'ws://127.0.0.1:{self.server.socket.getsockname()[1]}/api-ws/v1/realtime'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.addCleanup(self.close)
    def close(self):
        self.server.shutdown();self.thread.join(2)
    def test_real_client_handshake_and_jpeg_payload(self):
        with patch.dict(os.environ,{},clear=True):
            client=CloudClient(env_file=self.env,api_key='synthetic-key',realtime_ws_url=self.url)
        result=client.request_scene(self.image,{'event_id':'real-loopback'})
        self.assertEqual(self.headers,['Bearer synthetic-key']);self.assertFalse(self.errors)
        self.assertEqual(result.context['event_id'],'real-loopback');self.assertEqual(len(self.requests),5)
        self.assertEqual(self.requests[2]['type'],'input_image_buffer.append')
        self.assertEqual(result.schema_version,'road-observation-fast-v1')
    def test_default_cli_uses_realtime_and_preserves_raw_record(self):
        output=self.root/'result.json';stream=io.StringIO()
        with patch.dict(os.environ,{'CAR_CLOUD_API_KEY':'synthetic-key'},clear=True),redirect_stdout(stream):
            status=main(['--env-file',str(self.env),'--realtime-url',self.url,'--image',str(self.image),'--output',str(output)])
        self.assertEqual(status,0);record=json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(record['provider'],'qwen-realtime');self.assertEqual(record['raw_response']['response']['status'],'completed')
        self.assertNotIn('synthetic-key',output.read_text(encoding='utf-8'));self.assertNotIn('synthetic-key',stream.getvalue())
    def test_failed_final_text_is_preserved_without_credentials(self):
        self.bad=True;output=self.root/'failure.json'
        with patch.dict(os.environ,{'CAR_CLOUD_API_KEY':'synthetic-key'},clear=True),redirect_stderr(io.StringIO()):
            status=main(['--env-file',str(self.env),'--realtime-url',self.url,'--image',str(self.image),'--output',str(output)])
        self.assertEqual(status,1);record=json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(record['mode'],'failed-request');self.assertTrue(record['request']['raw_events'])
        self.assertNotIn('synthetic-key',output.read_text(encoding='utf-8'))


if __name__=='__main__': unittest.main()
