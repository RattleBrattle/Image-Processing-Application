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
    """Function that Rotates an image by a given angle.
       angle: Angle in degrees. Positive values mean counter-clockwise rotation.
       Add match case error handling for invalid angle values.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    match angle:
        case int() | float():
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))
        case _:
            raise ValueError("Invalid angle. Must be a number.")
        
def resize_image(image, width=None, height=None, fx=None, fy=None, interpolation=cv2.INTER_AREA):
    """ Function that resizes an image.
    Inputs:
        width: Desired width of the output image.
        height: Desired height of the output image.
        fx: Scale factor along the horizontal axis.
        fy: Scale factor along the vertical axis.
        interpolation: Interpolation method. Default is cv2.INTER_AREA.
        Note: If both width and height are provided, they take precedence over fx and fy.
    
    Add match case error handling for invalid input values.
    """
    match (width, height, fx, fy):
        case (int() | None, int() | None, None, None) if width is not None or height is not None:
            return cv2.resize(image, (width, height), interpolation=interpolation)
        case (None, None, float() | None, float() | None) if fx is not None or fy is not None:
            return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)
        case _:
            raise ValueError("Invalid input. Provide either width and/or height as integers or fx and/or fy as floats.")
        
# ---- 3. Filtering & Smoothing Functions ---- #
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """ Function that applies Gaussian Blur to an image. 
    Inputs:
        kernel_size: Size of the Gaussian kernel. Must be a tuple of two odd integers.
        sigma: Standard deviation in X and Y direction. Default is 0, which means it is calculated from kernel size.

    Add match case error handling for invalid kernel_size values.
    """
    match kernel_size:
        case (int() | None, int() | None) if kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1:
            return cv2.GaussianBlur(image, kernel_size, sigma)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two odd integers or a single odd integer.")

def apply_median_blur(image, kernel_size=(5, 5)):
    """ Function that applies Median blur on input image.
    Inputs:
        image: as the source image to apply the blur to.
        kernsel_size: The size of the kernel to apply.

    Add match case error handling for invalid kernel_size input.
    """
    match kernel_size:
        case (int() | None, int() | None) if kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1:
            return cv2.medianBlur(image, kernel_size)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two odd integers or a single odd integer.")

def simple_filter(image, kernel_size, ddepth_val):
    """ Function that applies simple filter to remove noise from an image.
    Inputs:
        Kernel_size: is the size of the kernel to apply.
        image: input image with noise to be filtered.
        
    """