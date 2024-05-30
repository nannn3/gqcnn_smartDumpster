import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class SimpleUI:
    def __init__(self, root, image_array):
        """
        Initializes the UI with an image and buttons.

        Args:
            root (tk.Tk): The root window.
            image_array (np.ndarray): The image to display as a numpy array.
        """
        self.root = root
        self.root.title("Simple UI")

        # Convert numpy array to PIL image
        self.image = Image.fromarray(image_array)
        self.photo = ImageTk.PhotoImage(self.image)

        # Create a frame for the image and buttons
        self.frame = ttk.Frame(self.root)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        # Create a label to display the image
        self.image_label = ttk.Label(self.frame, image=self.photo)
        self.image_label.grid(row=0, column=0, rowspan=3)
        self.image_label.bind("<Button-1>", self.on_image_click)

        # Create buttons
        self.pickup_button = ttk.Button(self.frame, text="Pickup", command=self.on_pickup)
        self.pickup_button.grid(row=0, column=1, padx=5, pady=5)

        self.start_button = ttk.Button(self.frame, text="Start", command=self.on_start)
        self.start_button.grid(row=1, column=1, padx=5, pady=5)

        self.calibrate_button = ttk.Button(self.frame, text="Calibrate", command=self.on_calibrate)
        self.calibrate_button.grid(row=2, column=1, padx=5, pady=5)

        # Variables to store click coordinates
        self.clicks = []

        # Bind keyboard events
        self.root.bind("<Key>", self.on_key_press)

    def on_image_click(self, event):
        """
        Handles image click events.

        Args:
            event (tk.Event): The event object containing click coordinates.
        """
        self.clicks.append((event.x, event.y))
        print(f"Image clicked at: {event.x}, {event.y}")

    def on_pickup(self):
        """Handles Pickup button click."""
        print("Pickup button clicked")
        return "Pickup"

    def on_start(self):
        """Handles Start button click."""
        print("Start button clicked")
        return "Start"

    def on_calibrate(self):
        """Handles Calibrate button click."""
        print("Calibrate button clicked")
        return "Calibrate"

    def on_key_press(self, event):
        """
        Handles keyboard events.

        Args:
            event (tk.Event): The event object containing the key pressed.
        """
        print(f"Key pressed: {event.keysym}")

    def get_last_click(self):
        """
        Returns the most recent click coordinates or None if no clicks have occurred.

        Returns:
            tuple: The (x, y) coordinates of the last click or None.
        """
        if self.clicks:
            return self.clicks.pop(-1)
        return None

    def update_image(self, image_array):
        """
        Updates the displayed image.

        Args:
            image_array (np.ndarray): The new image to display as a numpy array.
        """
        self.image = Image.fromarray(image_array)
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo

if __name__ == "__main__":
    # Create a random noise image
    image_array = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

    # Initialize the root window
    root = tk.Tk()

    # Initialize the UI with the random noise image
    ui = SimpleUI(root, image_array)

    # Start the Tkinter main loop
    root.mainloop()
