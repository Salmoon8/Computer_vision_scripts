from typing import Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float
from skimage.filters import sobel

def find_edges(img: np.ndarray) -> np.ndarray:
    """Finds edges in the image using the Sobel operator."""
    return sobel(img) if img.ndim == 2 else np.dstack([sobel(c) for c in img.T])

def calculate_energy(img: np.ndarray, w_line: float, w_edge: float, edge: np.ndarray) -> np.ndarray:
    """Calculates the energy of the snake based on image intensities and edges."""
    return w_line * img + w_edge * edge

def calculate_forces(snake: np.ndarray, alpha: float, beta: float, gamma: float,
                     energy: np.ndarray, spline: RectBivariateSpline) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the internal and external forces acting on the snake."""
    coeffs = spline.get_coeffs().reshape(2, -1)
    coeffs_resized = np.resize(coeffs[0][1:-1], snake[:, 0].shape)
    fx = beta * (snake[:, 0] - 2 * coeffs_resized + np.roll(snake[:, 0], 1) + np.roll(snake[:, 0], -1)) + \
     gamma * coeffs_resized

    fy = beta * (snake[:, 1] - 2 * coeffs_resized + np.roll(snake[:, 1], 1) + np.roll(snake[:, 1], -1)) + \
         gamma * coeffs_resized
    fx[1:-1] -= alpha * (coeffs[1, 2:(len(snake)+2)][1:-1] - coeffs[1, :len(snake)][1:-1])
    fy[1:-1] += alpha * (coeffs[0, 2:len(snake)+2][1:-1] - coeffs[0, :len(snake)][1:-1])

    return fx, fy

def move_snake(snake: np.ndarray, Fx: np.ndarray, Fy: np.ndarray, max_px_move: float = 0.01) -> np.ndarray:
    """Moves the snake based on the forces acting on it."""
    dx = np.clip(Fx, -max_px_move, max_px_move)
    dy = np.clip(Fy, -max_px_move, max_px_move)
    snake[:, 0] = np.round(snake[:, 0] + dx)
    snake[:, 1] = np.round(snake[:, 1] + dy)
    return snake


def active_contour(image: np.ndarray, snake: np.ndarray,
                   alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.01,
                   max_num_iter: int = 2500, convergence: float = 0.1,) -> np.ndarray:
    # Convert image to float
    img = img_as_float(image)
    
    # Create spline interpolation of image intensities
    spline = RectBivariateSpline(np.arange(img.shape[0]), np.arange(img.shape[1]), img.T, kx=2, ky=2, s=0, bbox=[0, img.shape[0]-1, 0, img.shape[1]-1])
    
    # Find edges in image
    edge = find_edges(img)
    
    # Calculate initial energy
    energy = calculate_energy(img, 0, gamma, edge)
    
    # Iterate until convergence or max_num_iter is reached
    for i in range(max_num_iter):
        # Calculate internal and external forces
        Fx, Fy = calculate_forces(snake, alpha, beta, gamma, energy, spline)
        
        # Move snake based on forces
        snake = move_snake(snake, Fx, Fy, 1)
        
        # Recalculate energy
        energy = calculate_energy(img, 0, gamma, edge)
        
        # Check for convergence
        if np.max(np.abs(Fx)) < convergence and np.max(np.abs(Fy)) < convergence:
            break
    
    return snake

def init_circle_snake(center, radius, num_points):
    """Initialize a circle snake with given center and radius."""
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    snake = np.array([x, y]).T
    return snake

def init_square_snake(center, side_length):
    """Initialize a square snake with given center and side length."""
    x0, y0 = center
    half_side = side_length / 2
    x = np.array([x0 - half_side, x0 + half_side, x0 + half_side, x0 - half_side])
    y = np.array([y0 - half_side, y0 - half_side, y0 + half_side, y0 + half_side])
    snake = np.array([x, y]).T
    return snake
