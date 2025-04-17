import http.server
import socketserver
import os

# --- Configuration ---
PORT = 8080  # You can change this port number if needed
HOST = "localhost" # Use "0.0.0.0" to allow access from other devices on your network
HTML_FILE = "web.html" # The main HTML file to serve
# -------------------

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
# This ensures SimpleHTTPRequestHandler serves files from the correct place
os.chdir(script_dir)

# Define the handler to use (serves files from the current directory)
Handler = http.server.SimpleHTTPRequestHandler

# Create the server
# Binding to HOST (e.g., "localhost") and PORT
# Using socketserver.TCPServer for the server instance
httpd = socketserver.TCPServer((HOST, PORT), Handler)

print(f"Serving HTTP on http://{HOST}:{PORT}/")
print(f"Serving directory: {script_dir}")
print(f"Main file should be accessible at: http://{HOST}:{PORT}/{HTML_FILE}")
print("Press Ctrl+C to stop the server.")

try:
    # Start the server, it will run forever until interrupted
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
    # Cleanly shut down the server
    httpd.shutdown()
    httpd.server_close()