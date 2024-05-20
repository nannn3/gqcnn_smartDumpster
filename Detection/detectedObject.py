'''https://chatgpt.com/g/g-yLQXrzfua-software-buddy/c/b3d0f2c3-84e7-4f38-89bb-18967602eafd
Simple detection object class'
'''
class DetectedObject:
    def __init__(self, name, color=None, tolerance=10):
        """
        Initialize a new detected object with a name, and optionally, color and tolerance.

        Args:
            name (str): The name of the detected object.
            color (tuple, optional): The RGB color as a tuple (R, G, B). Defaults to None.
            tolerance (int, optional): The default tolerance level for color comparison. Defaults to 10.
        """
        self.name = name
        self.color = color
        self.tolerance = tolerance
        self.contour = None
        self.properties = {}  # Dictionary to store properties dynamically

    def update_properties(self, **properties):
        """
        Update the properties of the detected object with any number of additional attributes.
        Ensures all property keys are strings.

        Args:
            **properties: Arbitrary keyword arguments representing the properties to update.
        
        Raises:
            TypeError: If any of the keys are not strings.
        """
        for key in properties:
            if not isinstance(key, str):
                raise TypeError("All property keys must be strings.")
        self.properties.update(properties)

    def get_property(self, key):
        """
        Retrieve the value of a property by key.

        Args:
            key (str): The property key to retrieve.
        
        Returns:
            The value of the property.
        
        Raises:
            KeyError: If the key does not exist in the properties.
        """
        if key not in self.properties:
            raise KeyError(f"Property '{key}' not found in object '{self.name}'.")
        return self.properties[key]

    def is_same_color(self, color):
        """
        Check if a given color matches the detected object's color within the object's tolerance.

        Args:
            color (tuple): The color to compare.

        Returns:
            bool: True if colors match within the object's tolerance, otherwise False.

        Raises:
            ValueError: If the detected object does not have a color set for comparison.
        """
        if self.color is None:
            raise ValueError(f"{self.name} has no color set for comparison.")
        foo =list(abs(c1 - c2) <= self.tolerance for c1, c2 in zip(self.color, color))
        #print(self.name,list(foo))
        return foo[0] and foo[1] and foo[2]

    def __lt__(self,other):
        """
        Compare this DetectedObject to another based on the x coordinates of the contour's center
        Args:
            other(DetectedObject) The object to compare to
        Returns: 
            bool: True if this object's x coordinate is lower than other's x cordinate
        Raises:
            KeyError if either object doesn't have a contour
        """
        try:
            return self.get_center()[0] < other.get_center()[0]
        except KeyError as e:
            raise e

    def get_center(self):
        '''
        Gets the center point of the contour associated with the object
        Returns:
            tuple( x,y) of the center of the object
        Raises:
            KeyError if it has no contour
        '''
        try: 
            y,x = self.get_property('contour').bounding_box.center
            return (x,y)
        except KeyError as e:
            raise e

    def __str__(self):
        return f"{self.name} with color {self.color}, properties: {self.properties}"
