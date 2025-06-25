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
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-4">
            <h1 class="mb-4">PoseTestBot Control</h1>
            <div class="btn-group mb-3" role="group">
                <button class="btn btn-primary" onclick="runCommand('start_iiwa')">Start IIWA</button>
                <button class="btn btn-danger" onclick="runCommand('stop_iiwa')">Stop IIWA</button>
                <button class="btn btn-success" onclick="runCommand('realsense_multi')">Run Realsense</button>
            </div>
            <hr>
            <h2>Output</h2>
            <div class="card">
                <div class="card-body bg-light">
                    <pre id="output" class="mb-0"></pre>
                </div>
            </div>
        </div>

        <script>
            function runCommand(command) {
                const outputElement = document.getElementById('output');
                outputElement.textContent = 'Running ' + command + '...';

                fetch('/run-command', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ command: command }),
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    outputElement.textContent = data.output;
                })
                .catch(error => {
                    console.error('Error:', error);
                    outputElement.textContent = 'Error executing command: ' + command + '. See console for details.';
                });
            }
        </script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

@app.route('/run-command', methods=['POST'])
def run_command():
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({'output': 'Invalid request: command not found'}), 400
    command = data['command']
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
