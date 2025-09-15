""" Image Processing Module

    Description: This module is designed to perform various image processing tasks such as color transformations,
    geometric transformations, filtering, edge detection, Histogram Equalization & calculation, and Edge detection.
    
    Author: Basel Mohamed Mostafa Sayed
    
    Date of Creation: 9/14/2025
    
    Version: 0.1
    
    Email: baselmohamed802@gmail.com

    Changelog:
    - 0.1: Initial version with basic color transformation functions. (09/14/2025)
    - 0.2: Finished all functions (09/15/2025)
"""
# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    """
    match flip_code:
        case 0 | 1 | -1:
            return cv2.flip(image, flip_code)
        case _:
            raise ValueError("Invalid flip code. Must be 0, 1, or -1.")
        
def rotate_image(image, angle):
    """Function that Rotates an image by a given angle.
       angle: Angle in degrees. Positive values mean counter-clockwise rotation.
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
    """ 
    Function that resizes an image.
    Inputs:
        width: Desired width of the output image.
        height: Desired height of the output image.
        fx: Scale factor along the horizontal axis.
        fy: Scale factor along the vertical axis.
        interpolation: Interpolation method. Default is cv2.INTER_AREA.
        Note: If both width and height are provided, they take precedence over fx and fy.
    """
    match (width, height, fx, fy):
        case (int() | None, int() | None, None, None) if width is not None or height is not None:
            return cv2.resize(image, (width, height), interpolation=interpolation)
        case (None, None, float() | None, float() | None) if fx is not None or fy is not None:
            return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)
        case _:
            raise ValueError("Invalid input. Provide either width and/or height as integers or fx and/or fy as floats.")
        
# ---- 3. Filtering & Smoothing Functions ---- #
def average_blur(image, kernel_size=(5, 5)):
    """
    Function that applies normal (average) blur to an image.
    inputs:
        image: input image to apply the blur on.
        kernel_size: Size of the kernel. Must be a tuple of two positive integers.
    """
    match kernel_size:
        case (int(k1), int(k2)) if k1 > 0 and k2 > 0:
            return cv2.blur(image, kernel_size)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two positive integers (e.g., (5, 5)).")
        
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """ 
    Function that applies Gaussian Blur to an image. 
    Inputs:
        kernel_size: Size of the Gaussian kernel. Must be a tuple of two odd integers.
        sigma: Standard deviation in X and Y direction. Default is 0.
    """
    match kernel_size:
        case (int(k1), int(k2)) if k1 % 2 == 1 and k2 % 2 == 1:
            return cv2.GaussianBlur(image, kernel_size, sigma)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two odd integers (e.g., (5, 5)).")

def apply_median_blur(image, kernel_size=5):
    """ 
    Function that applies Median blur on input image.
    Inputs:
        image: the source image.
        kernel_size: The size of the kernel (a single odd integer).
    """
    match kernel_size:
        case int(k) if k % 2 == 1 and k > 1:
            return cv2.medianBlur(image, kernel_size)
        case _:
            raise ValueError("Invalid kernel size. Must be a single, positive odd integer greater than 1 (e.g., 3, 5, 7).")

def simple_filter(image, kernel_size=(5, 5), ddepth_val=-1):
    """ 
    Function that applies a simple averaging filter to remove noise.
    Inputs:
        kernel_size: size of the kernel (tuple of two integers).
        image: input image.
    """
    match kernel_size:
        case (int(k1), int(k2)) if k1 > 0 and k2 > 0:
            kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
            return cv2.filter2D(image, ddepth_val, kernel)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two positive integers (e.g., (5, 5)).")
        
def bilateral_filter(image, d_val=6 , sigma_color=75, sigma_space=75):
    """ 
    Function that applies bilateral filtering to remove noise from input image.
    inputs:
        image: input image with noise.
        d_val: number of neigboring pixels used during filtering.
        sigma_color: color space of the filter.
        sigma_space: coordinate space of the filter.
    """
    match (d_val, sigma_color, sigma_space):
        case (int() | None, int() | None, int() | None) if d_val is not None:
            return cv2.bilateralFilter(image, d_val, sigma_color, sigma_space)
        case _:
            raise ValueError("Invalid d_val or sigma_color or sigma_space, make sure the input values are correct.")
        
def box_filter(image, ddepth_val=-1, k_size=(5, 5), anchor=(-1, -1), normalize=True):
    """ 
    Function that applies a Box Filter to the image.
    inputs:
        image: input image.
        ddepth_val: desired depth of output image (-1 same as input).
        k_size: size of the kernel (tuple of two integers).
        anchor: Anchor point of the kernel; default (-1, -1) is the center.
        normalize: If True, the kernel is normalized (averaging filter).
    """
    match k_size:
        case (int(k1), int(k2)) if k1 > 0 and k2 > 0:
            return cv2.boxFilter(image, ddepth_val, k_size, anchor=anchor, normalize=normalize)
        case _:
            raise ValueError("Invalid kernel size. Must be a tuple of two positive integers (e.g., (5, 5)).")
        
# ---- 4. Edge Detection Functions ---- #
def sobel_edge_detection(image, k_size=3):
    """ Function to apply Sobel Edge detection on input image or video frame.
    inputs:
        image: input image or video frame.
        k_size: size of the extended Sobel kernel; it must be 1, 3, 5, or 7 (odd number).
    """
    match k_size:
        case (int() | None) if k_size % 2 == 1: 
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, k_size)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, k_size)
            return cv2.magnitude(sobel_x, sobel_y)
        case _:
            raise ValueError("Invalid Ksize, it must be an odd value up to 7")
        
def canny_edge_detection(image, threshold1=100, threshold2=200):
    """ 
    Function that applies Canny Edge Detection on input image or video frame.
    inputs:
        image: input image or video frame.
        threshold1: first threshold for the hysteresis procedure.
        threshold2: second threshold for the hysteresis procedure.
    """
    match (threshold1, threshold2):
        case (int() | float(), int() | float()) if threshold1 >= 0 and threshold2 >= 0:
            return cv2.Canny(image, threshold1, threshold2)
        case _:
            raise ValueError("Invalid thresholds. Must be non-negative numbers.")

def laplacian_edge_detection(image):
    """ Function that applies Laplacian edge detection to the input image.
    inputs:
        image: input image or video frame.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

# ---- 5. Thresholding Techniques ---- #
def binary_thresholding(image, min_threshold, max_threshold, threshold_type=cv2.THRESH_BINARY):
    """
    Function to perform binary thresholding on input image or video frame.
    inputs:
        image: input image or video frame.
        min_threshold: minimum threshold value.
        max_threshold: maximum threshold value.
        threshold_type: threshold type (cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV).
    """
    match (min_threshold, max_threshold, threshold_type):
        case (int() | float(), int() | float(), int()) if min_threshold >= 0 and max_threshold >= 0 and threshold_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
            _, thresholded = cv2.threshold(image, min_threshold, max_threshold, threshold_type)
            return thresholded
        case _:
            raise ValueError("Invalid threshold values or threshold type. Threshold values must be non-negative numbers and threshold type must be cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV.")

def adaptive_thresholding(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    """
    Function to perform adaptive thresholding on input image or video frame.
    inputs:
        image: input image or video frame.
        max_value: maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
        adaptive_method: adaptive method to use (cv2.ADAPTIVE_THRESH_GAUSSIAN_C or cv2.ADAPTIVE_THRESH_MEAN_C).
        threshold_type: threshold type (cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV).
        block_size: size of the neighborhood area (must be odd and greater than 1).
        C: constant subtracted from the mean or weighted mean.
    """
    match (max_value, adaptive_method, threshold_type, block_size, C):
        case (int() | float(), int(), int(), int() | None, int() | None) if block_size > 1 and block_size % 2 == 1:
            return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)
        case _:
            raise ValueError("Invalid parameters for adaptive thresholding.")
        
def color_thresholding(image, lower_bound, upper_bound):
    """
    Function to perform color thresholding on an input image.
    inputs:
        image: input image.
        lower_bound: Lower HSV bound (e.g., np.array([20, 100, 100])).
        upper_bound: Upper HSV bound (e.g., np.array([30, 255, 255])).
    """
    hsv_img = convert_to_hsv(image)
    return cv2.inRange(hsv_img, lower_bound, upper_bound)
        
# ---- 6. Histogram Operations ---- #
def plot_histogram(image, image_color, plot_title='Image Histogram', x_label='Pixel Value', y_label='Frequency'):
    """ 
    Function to plot the image histogram for all color values.
    inputs:
        image: input image to plot the histogram for.
        image_color: determine what type of image color to plot for (BGR, RGB, etc..)
        for RGB = 1 and BGR = 0 and Grayscale = 2.
    """
    match image_color:
        case 0:
            bgr_channels = ['b', 'g', 'r']
            for i, col in enumerate(bgr_channels):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
                plt.title(plot_title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            plt.show()
        case 1:
            rgb_channels = ['r', 'g', 'b']
            for i, col in enumerate(rgb_channels):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
                plt.title(plot_title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            plt.show()
        case 2:
            gray_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(gray_hist, color='k')
            plt.xlim([0, 256])
            plt.title(plot_title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()
        case _:
            raise ValueError("Invalid image color. It must be 0 for BGR or 1 for RGB.")

def apply_histogram_equalization(image):
    """ 
    Function that applies Histogram Equalization to enhance the contrast of the input grayscale image.
    inputs:
        image: input grayscale image.
    """
    if len(image.shape) == 3:
        gray_img = convert_to_grayscale(image)
    else:
        gray = image
    image_eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(image_eq, cv2.COLOR_GRAY2BGR)
    
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """ 
    Function that applies CLAHE Histogram Equalization to enhance the contrast of the input grayscale image.
    inputs:
        image: input image.
        clip_limit: threshold for contrast limiting.
        tile_grid_size: size of grid for histogram equalization (tuple of two integers).
    """
    match tile_grid_size:
        case (int(t1), int(t2)) if t1 > 0 and t2 > 0:
            gray_image = convert_to_grayscale(image)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            image_clahe = clahe.apply(gray_image)
            return cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2BGR)
        case _:
            raise ValueError("Invalid tile grid size. Must be a tuple of two positive integers (e.g., (8, 8)).")
        
# ---- 7. Contours and Drawing ---- #
def find_and_draw_contours(image, method, rtr_type=cv2.RETR_EXTERNAL, rtr_approximation=cv2.CHAIN_APPROX_SIMPLE, border_color=(0, 255, 0), line_thickness=3):
    """
    Function that finds and draws the contours on an image.
    inputs:
        image: input image.
        method: determine what type of contour to draw (Rectangle, Circle, Ellipse, Polygon) or Normal.
        rtr_type: The type of return to be used with the find contours function.
        rtr_approximation: Type of approximation method.
        border_color: Tuple of border color for the contour drawing.
        line_thickness: Thickness of the border contour line.
    """
    image_copy = image.copy()
    gray_image = convert_to_grayscale(image)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, rtr_type, rtr_approximation)
    
    match method:
        case 'Rectangle':
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), border_color, line_thickness)
        case 'Circle':
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(image_copy, center, radius, border_color, line_thickness)
        case 'Ellipse':
            for contour in contours:
                if len(contour) >= 5: 
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(image_copy, ellipse, border_color, line_thickness)
        case 'Polygon':
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(image_copy, [approx], -1, border_color, line_thickness)
        case 'Normal':
            cv2.drawContours(image_copy, contours, -1, border_color, line_thickness)
        case _:
            raise ValueError("Invalid method. Must be 'Rectangle', 'Circle', 'Ellipse', 'Polygon', or 'Normal'.")
    return image_copy