import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from filters import read_image_grayscale
import numpy as np


def globalthresholding(img, threshold=127):
    # compare pixels with the threshold
    binary_image = (img >= threshold)
    return binary_image.astype(np.float64)


def localthresholding(img, block_size=15):
    rows, cols = img.shape
    binary_image = np.zeros((rows, cols))
    #img = cv.medianBlur(img,5)
    # go over blocks and apply a threshold over each block
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = img[i:i+block_size, j:j+block_size]
            block_mean = np.mean(block)
            binary_block = (block >= block_mean)
            binary_image[i:i+block_size, j:j+block_size] = binary_block
    return binary_image


def fourier(path):
    img = read_image_grayscale(path)
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    return Fshift, img


def lowpass(Fshift, img, D0):
    M, N = img.shape
    H = np.zeros((M, N), dtype=np.float32)

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0

    Gshift = Fshift * H
    return Gshift, H


def highpass(Fshift, H):
    H = 1 - H
    Gshift = Fshift * H
    return Gshift


def inversefourier(Gshift):
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))

    return g


def hybrid(path, path2, threshold):
    Fshift1, img1 = fourier(path)
    Fshift2, img2 = fourier(path2)
    Gshift, H = lowpass(Fshift1, img1, threshold)

    sum = highpass(Fshift2, H)+Gshift
    hybrid = inversefourier(sum)

    return hybrid


def convertcolortograyscale(colorimg):
    # input image must be a color image and the channels are in form BGR (read with open cv)
    b = colorimg[:, :, 0]
    g = colorimg[:, :, 1]
    r = colorimg[:, :, 2]
    # Convert image to grayscale using the formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = np.dot(colorimg[..., :3], [0.1140, 0.5870, 0.2989]).astype(np.uint8)
    return gray, b, g, r
