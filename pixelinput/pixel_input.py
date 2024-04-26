import cv2 as cv
#https://chat.openai.com/share/1030246f-4cc6-4437-a2ff-7d261de5aaa1
class PixelInput:
    def __init__(self):
        self._x = None
        self._y = None
        self._posList = []
        self._runflag = False

        # Assign mouse click to the windows
        cv.namedWindow('color')
        cv.setMouseCallback('color', self.on_mouse_click)
        cv.namedWindow('binary_image')
        cv.setMouseCallback('binary_image', self.on_mouse_click)
        cv.namedWindow('depth')
        cv.setMouseCallback('depth', self.on_mouse_click)


    def on_mouse_click(self, event, x, y, flags, param):
        """
        Mouse click event handler.

        Records mouse click positions for object detection.

        Parameters:
            event: Type of mouse event.
            x (int): x-coordinate of the mouse click.
            y (int): y-coordinate of the mouse click.
            flags: Flags indicating the state of the mouse buttons.
            param: Additional parameters.
        """
        if event == cv.EVENT_LBUTTONDOWN:
            self._posList.append((x, y))
            self._runflag = True

    def is_valid_pixel(self, x, y):
        """Check if the pixel coordinates are valid (integers within a specific range)."""
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        return (0 <= x <= 639) and (0 <= y <= 479)

    def get_pixel_coordinates(self):
        """Get the next available valid pixel coordinates if present."""
        if self._posList:
            x, y = self._posList.pop(0)
            if self.is_valid_pixel(x, y):
                self._x, self._y = int(x), int(y)
                return int(x), int(y)
            else:
                raise ValueError(f"Invalid pixel coordinates: ({x}, {y}). "
                                 f"Coordinates must be integers and within range x=[0, 639], y=[0, 479].")
        else:
            return None,None

    @property
    def x(self):
        """Get the x-coordinate of the selected pixel."""
        return self._x

    @property
    def y(self):
        """Get the y-coordinate of the selected pixel."""
        return self._y

if __name__ == "__main__":
    pixel_input = PixelInput()

    # Simulate event loop (replace with actual event loop logic)
    while True:
        # Process events (e.g., GUI updates, camera frames, etc.)
        # In your main event loop, call get_pixel_coordinates to check for new coordinates
        # and handle them accordingly
        try:
            x, y = pixel_input.get_pixel_coordinates()
            if (x is not None) and (y is not None):
                print(f"Selected pixel coordinates: ({x}, {y})")
        except ValueError as e:
            print(f"Error: {e}")

        # You can insert other logic or break conditions here for your event loop
        # For example:
        # if some_condition:
        #     break

        # Simulate delay to prevent busy-waiting (replace with actual event loop delay)
        cv.waitKey(1)

