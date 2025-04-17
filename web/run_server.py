# run_server.py
import time
import subprocess
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
SCRIPT_TO_RUN = "main.py"  # The script that starts the actual server
WATCH_PATH = "."  # Directory to watch ('.' means current directory)
WATCH_EXTENSIONS = [".html", ".css", ".js", ".py"]  # File extensions to monitor
PYTHON_EXECUTABLE = sys.executable  # Use the same python that runs this script
RELOAD_COOLDOWN = 5.
# -------------------

server_process = None
last_reload_time = 0


def start_server():
    """Stops the existing server (if running) and starts a new one."""
    global server_process, last_reload_time
    current_time = time.time()

    if server_process:
        print("Reloading: Stopping server...")
        server_process.terminate()  # Ask the process to stop
        try:
            server_process.wait(timeout=5)  # Wait up to 5 seconds for it to stop
        except subprocess.TimeoutExpired:
            print("Server did not stop gracefully, killing.")
            server_process.kill()  # Force kill if it doesn't stop
        server_process = None

    print(f"Starting server: {PYTHON_EXECUTABLE} {SCRIPT_TO_RUN}")
    # Start the main.py script as a subprocess
    server_process = subprocess.Popen([PYTHON_EXECUTABLE, SCRIPT_TO_RUN])
    last_reload_time = current_time # Update time *after* starting


class ChangeHandler(FileSystemEventHandler):
    """Handles file system events."""

    def on_modified(self, event):
        """Restart server ONLY on modification of relevant files."""
        global last_reload_time
        current_time = time.time()

        if event.is_directory:
            return  # Ignore directory modification events

        _, ext = os.path.splitext(event.src_path)
        if ext.lower() in WATCH_EXTENSIONS:
            # Optional: Add cooldown to prevent rapid restarts
            if current_time - last_reload_time < RELOAD_COOLDOWN:
                # Optionally print a message, or just silently ignore
                # print(f"Change detected in {event.src_path}, but still within cooldown. Skipping.")
                return

            print(f"Detected modification in {event.src_path}, reloading...")
            start_server()  # Restart the server (this updates last_reload_time)


if __name__ == "__main__":
    # Ensure the path to watch exists and is a directory
    watch_dir = os.path.abspath(WATCH_PATH)
    if not os.path.isdir(watch_dir):
        print(f"Error: Watch path '{watch_dir}' is not a valid directory.")
        sys.exit(1)

    # Initial start of the server
    start_server()

    # Set up watchdog observer
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(
        event_handler, watch_dir, recursive=False
    )  # Watch only the top directory
    observer.start()
    print(f"Watching directory '{watch_dir}' for changes...")
    print("Press Ctrl+C to stop the watcher and the server.")

    try:
        while True:
            # Keep the main thread alive to allow observer thread to run
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping observer and server...")
        observer.stop()
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("Server stopped.")
    finally:
        observer.join()  # Wait for the observer thread to finish
        print("Watcher stopped.")

