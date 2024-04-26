import unittest
from unittest.mock import patch
from pixel_input import PixelInput  # Import your PixelInput class
#https://chat.openai.com/share/5a83b6f1-39fc-4b3e-af9f-446f336d98f4

class TestPixelInput(unittest.TestCase):
    def setUp(self):
        self.pixel_input = PixelInput()

    def test_is_valid_pixel(self):
        # Test with valid and invalid pixel coordinates
        self.assertTrue(self.pixel_input.is_valid_pixel(100, 200))
        self.assertFalse(self.pixel_input.is_valid_pixel(-10, 300))
        self.assertFalse(self.pixel_input.is_valid_pixel(700, 480))
        self.assertFalse(self.pixel_input.is_valid_pixel(10.5, 20))  # Non-integer coordinates

    def test_get_pixel_coordinates(self):
        # Test the behavior of get_pixel_coordinates
        # Mocking the user input for test purposes
        with patch.object(self.pixel_input, '_posList', [(100, 200)]):  # Mocking a list of coordinates
            x, y = self.pixel_input.get_pixel_coordinates()
            self.assertEqual((x, y), (100, 200))

        # Test with invalid coordinates in posList
        with patch.object(self.pixel_input, '_posList', [(700, 480)]):  # Invalid coordinates
            with self.assertRaises(ValueError):
                self.pixel_input.get_pixel_coordinates()
        # Test no cordinates given:
        with patch.object(self.pixel_input,'_posList',[]):
            x,y = self.pixel_input.get_pixel_coordinates()
            assert(x is None)
            assert(y is None)


    def test_x_and_y_properties(self):
        # Test the properties x and y after setting valid pixel coordinates
        self.pixel_input._x = 100
        self.pixel_input._y = 200
        self.assertEqual(self.pixel_input.x, 100)
        self.assertEqual(self.pixel_input.y, 200)

    # Add more test cases as needed for other methods or scenarios

if __name__ == '__main__':
    unittest.main()

