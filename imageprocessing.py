""" Image Processing Module

    Description: This module is designed to perform various image processing tasks such as color transformations,
    geometric transformations, filtering, edge detection, Histogram Equalization & calculation, and Edge detection.
    
    Author: Basel Mohamed Mostafa Sayed
    
    Date of Creation: 9/14/2025
    
    Version: 0.1
    
    Email: baselmohamed802@gmail.com

    Changelog:
    - 0.1: Initial version with basic color transformation functions. (09/14/2025)
"""
# Import necessary libraries
import cv2

# ---- 1. Basic Color Transformations Functions ---- #
def convert_to_grayscale(image):
    """ Function that converts an image to grayscale. """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(image):
    """ Function that converts an image to HSV color space. """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ---- 2. Geometric Transformations Functions ---- #
def flip_image(image, flip_code):
    """ Function that flips an image. 
        flip_code: 0 for vertical, 1 for horizontal, -1 for both axes.
        Add match case error handling for invalid flip_code values.
    """
    match flip_code:
        case 0 | 1 | -1:
            return cv2.flip(image, flip_code)
        case _:
            raise ValueError("Invalid flip code. Must be 0, 1, or -1.")
        
def rotate_image(image, angle):
    