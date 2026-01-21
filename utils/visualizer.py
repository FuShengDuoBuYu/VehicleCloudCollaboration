import threading
import time
import cv2
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Vehicle Cloud Collaboration Monitor</title>
                    <style>
                        body { background: #1a1a1a; color: #fff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; text-align: center; }
                        h1 { color: #00ffcc; margin-bottom: 20px; }
                        .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
                        .pane { background: #2a2a2a; padding: 10px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); width: 48%; min-width: 400px; }
                        .pane h3 { margin-top: 0; color: #aaa; border-bottom: 1px solid #444; padding-bottom: 8px; font-size: 1em; }
                        .img-container { width: 100%; position: relative; padding-top: 75%; background: #000; border-radius: 4px; overflow: hidden; }
                        img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
                    </style>
                </head>
                <body>
                    <h1>🚗 车云协同 - 实时监控</h1>
                    <div class="container">
                        <div class="pane">
                            <h3>实时流 (Live Stream)</h3>
                            <div class="img-container">
                                <img src="/live" id="live-img">
                            </div>
                        </div>
                        <div class="pane">
                            <h3>分析帧 (Latest Analysis)</h3>
                            <div class="img-container">
                                <img src="/analysis" id="analysis-img">
                            </div>
                        </div>
                    </div>
                </body>
                </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/live':
            self.handle_stream('live')
        elif self.path == '/analysis':
            self.handle_stream('analysis')

    def handle_stream(self, stream_type):
        self.send_response(200)
        self.send_header('Age', 0)
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        
        last_frame_id = -1
        try:
            while True:
                with self.server.visualizer.lock:
                    if stream_type == 'live':
                        frame_id = self.server.visualizer.live_id
                        jpeg = self.server.visualizer.live_jpeg
                    else:
                        frame_id = self.server.visualizer.analysis_id
                        jpeg = self.server.visualizer.analysis_jpeg

                if frame_id != last_frame_id and jpeg is not None:
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpeg))
                    self.end_headers()
                    self.wfile.write(jpeg + b'\r\n')
                    last_frame_id = frame_id
                else:
                    time.sleep(0.01) # Small sleep to prevent CPU spinning
        except Exception:
            pass

class Visualizer:
    def __init__(self, port=8080):
        self.port = port
        self.live_jpeg = None
        self.analysis_jpeg = None
        self.live_id = 0
        self.analysis_id = 0
        self.lock = threading.Lock()
        self.server = None
        self.thread = None

    def start(self):
        self.server = ThreadedHTTPServer(('', self.port), StreamingHandler)
        self.server.visualizer = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"\n[INFO] 可视化面板已启动: http://localhost:{self.port}")

    def update_live(self, frame):
        # Scale down for faster streaming on Pi
        small_frame = cv2.resize(frame, (480, 360))
        ret, jpeg = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ret:
            with self.lock:
                self.live_jpeg = jpeg.tobytes()
                self.live_id += 1

    def update_analysis(self, frame, result):
        display_frame = frame.copy()
        status = "LONG-TAIL" if result['is_long_tail'] else "NORMAL"
        color = (0, 0, 255) if result['is_long_tail'] else (0, 255, 0)
        
        # UI Overlays
        cv2.rectangle(display_frame, (0, 0), (320, 80), (0,0,0), -1)
        cv2.putText(display_frame, f"{status}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"Score: {result['score']:.2f}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        ret, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with self.lock:
                self.analysis_jpeg = jpeg.tobytes()
                self.analysis_id += 1

