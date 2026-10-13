"""Local preview server with HTTP Range support (needed for <video> seeking).
Usage: python scripts/preview_server.py [port]   then open http://localhost:<port>/ (via VS Code port forward or ssh -L).
"""
import os, re, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        rng = self.headers.get('Range')
        if os.path.isfile(path) and rng:
            m = re.match(r'bytes=(\d*)-(\d*)', rng); size = os.path.getsize(path)
            start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            if start > end or start >= size:
                self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return None
            f = open(path, 'rb'); f.seek(start)
            self.send_response(206)
            self.send_header('Content-Type', self.guess_type(path)); self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}'); self.send_header('Content-Length', str(end - start + 1))
            self.end_headers(); self._range_left = end - start + 1; return f
        self._range_left = None; return super().send_head()
    def copyfile(self, src, dst):
        if self._range_left is None: return super().copyfile(src, dst)
        left = self._range_left
        while left > 0:
            buf = src.read(min(65536, left))
            if not buf: break
            dst.write(buf); left -= len(buf)
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache'); super().end_headers()
    def log_message(self, *a): pass

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    print(f'serving project page on http://127.0.0.1:{port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), RangeHandler).serve_forever()
