import tkinter as tk
import numpy as np
from SimpleUI import SimpleUI
import threading
import queue
from time import sleep

# Mock Camera class for example purposes
class MockCamera:
    def __init__(self):
        pass

    def frames(self):
        # Returning a random image as a placeholder
        return np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

def update_ui():
    """
    Updates the UI with the latest frame from the camera and handles UI events.
    """
    try:
        # Get the latest frame from the queue
        color_frame = frame_queue.get_nowait()
        ui.update_image(color_frame)

        # Get the most recent click coordinates
        last_click = ui.get_last_click()
        if last_click:
            print(f"Most recent click at: {last_click}")

    except queue.Empty:
        pass

    # Schedule the next update
    root.after(100, update_ui)

def main_loop():
    """
    Main loop that runs alongside the UI thread.
    """
    while True:
        # Get the latest frame from the camera
        color_frame = camera.frames()

        # Put the frame into the queue
        frame_queue.put(color_frame)

        # Handle button presses
        try:
            button_event = button_queue.get_nowait()
            print(f"Button pressed: {button_event}")
        except queue.Empty:
            pass

        # Perform other tasks here
        # For example, process the frame, run algorithms, etc.

        # Simulate a delay for other tasks
        sleep(0.1)

# Main loop
if __name__ == "__main__":
    # Initialize the camera
    camera = MockCamera()

    # Create queues for communication between threads
    frame_queue = queue.Queue()
    button_queue = queue.Queue()

    class CustomSimpleUI(SimpleUI):
        def on_pickup(self):
            """Handles Pickup button click."""
            button_queue.put("Pickup")
            super().on_pickup()

        def on_start(self):
            """Handles Start button click."""
            button_queue.put("Start")
            super().on_start()

        def on_calibrate(self):
            """Handles Calibrate button click."""
            button_queue.put("Calibrate")
            super().on_calibrate()

    # Capture an initial frame to initialize the UI
    initial_frame = camera.frames()

    # Initialize the root window for the UI
    root = tk.Tk()

    # Initialize the UI with the initial frame
    ui = CustomSimpleUI(root, initial_frame)

    # Start the periodic UI update
    root.after(100, update_ui)

    # Start the main loop in a separate thread
    main_thread = threading.Thread(target=main_loop)
    main_thread.daemon = True
    main_thread.start()

    # Start the Tkinter main loop in the main thread
    root.mainloop()
