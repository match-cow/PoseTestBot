from flask import Flask, request, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PoseTestBot Control</title>
    </head>
    <body>
        <h1>PoseTestBot Control</h1>
        <button onclick="runCommand('start_iiwa')">Start IIWA</button>
        <button onclick="runCommand('stop_iiwa')">Stop IIWA</button>
        <button onclick="runCommand('realsense_multi')">Run Realsense</button>
        <hr>
        <h2>Output</h2>
        <pre id="output"></pre>

        <script>
            function runCommand(command) {
                fetch('/run-command', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ command: command }),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('output').textContent = data.output;
                });
            }
        </script>
    </body>
    </html>
    """

@app.route('/run-command', methods=['POST'])
def run_command():
    command = request.json['command']
    script_path = ''

    if command == 'start_iiwa':
        script_path = 'start_iiwa.py'
    elif command == 'stop_iiwa':
        script_path = 'stop_iiwa.py'
    elif command == 'realsense_multi':
        script_path = 'realsense_multi.py'
    else:
        return jsonify({'output': 'Unknown command'})

    try:
        result = subprocess.check_output([sys.executable, script_path], stderr=subprocess.STDOUT, text=True)
        return jsonify({'output': result})
    except subprocess.CalledProcessError as e:
        return jsonify({'output': e.output})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
