import threading
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.fps = 0.0
        self.hands_detected = 0
        self.start_time = time.time()
        self.status = "initializing"

    def update(self, fps, hands_detected):
        with self._lock:
            self.fps = fps
            self.hands_detected = hands_detected
            self.status = "running"
    
    def get_snapshot(self):
        with self._lock:
            return {
                "status": self.status,
                "current_fps": round(self.fps, 2),
                "hands_detected": self.hands_detected,
                "uptime_seconds": round(time.time() - self.start_time, 2)
            }

# Global instance to be shared
state = SharedState()

class TelemetryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            data = state.get_snapshot()
            response = json.dumps(data)
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)
            
    # Suppress default logging to keep console clean
    def log_message(self, format, *args):
        pass

def start_server(port=5000):
    server = HTTPServer(('0.0.0.0', port), TelemetryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"📡 Telemetry server running on http://localhost:{port}/health")
    return server
