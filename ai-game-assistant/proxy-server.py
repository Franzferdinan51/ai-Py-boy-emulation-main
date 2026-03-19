#!/usr/bin/env python3
"""Proxy server that serves the frontend and proxies API requests to the backend."""

import http.server
import urllib.request
import urllib.parse
import os

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__)) + '/dist'
BACKEND_URL = 'http://localhost:5002'
PORT = 5173

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/') or self.path.startswith('/mcp/') or self.path in ['/health', '/openclaw/health']:
            # Proxy API requests to backend
            url = BACKEND_URL + self.path
            print(f"Proxy: {self.path} -> {url}")
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    self.send_response(response.status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.write(response.read())
            except Exception as e:
                print(f"Proxy error: {e}")
                self.send_error(502, str(e))
        else:
            # Serve static files from frontend dist
            super().do_GET()
    
    def do_POST(self):
        if self.path.startswith('/api/') or self.path.startswith('/mcp/') or self.path in ['/health', '/openclaw/health']:
            # Proxy POST requests
            url = BACKEND_URL + self.path
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            print(f"POST Proxy: {self.path} -> {url}")
            try:
                req = urllib.request.Request(url, data=body, method='POST')
                req.add_header('Content-Type', 'application/json')
                with urllib.request.urlopen(req, timeout=30) as response:
                    self.send_response(response.status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(response.read())
            except Exception as e:
                print(f"POST Proxy error: {e}")
                self.send_error(502, str(e))
        else:
            self.send_error(405)
    
    def do_OPTIONS(self):
        # CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def translate_path(self, path):
        # Serve from frontend dist directory
        if path == '/':
            path = '/index.html'
        return FRONTEND_DIR + path

if __name__ == '__main__':
    os.chdir(FRONTEND_DIR)
    with http.server.HTTPServer(('0.0.0.0', PORT), ProxyHandler) as httpd:
        print(f"Proxy server running on http://0.0.0.0:{PORT}")
        print(f"Frontend: {FRONTEND_DIR}")
        print(f"Backend: {BACKEND_URL}")
        httpd.serve_forever()
