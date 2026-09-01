from flask import Flask, send_from_directory
import os
import re

app = Flask(__name__)
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_pages')

@app.route('/')
def index():
    return send_from_directory(DEMO_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(path):
        from flask import abort
        abort(404)
        
    from flask import send_file
    return send_file(path, conditional=True)

@app.after_request
def add_header(response):
    response.headers['Accept-Ranges'] = 'bytes'
    return response

if __name__ == '__main__':
    print("Starting Demo Server on http://127.0.0.1:8000")
    print("Use this to open the demo pages so that video playback (Range requests) works properly.")
    app.run(port=8000, debug=False)
