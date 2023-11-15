import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.image as mpimg

# read image and return numpy array of it
def read_image(path):
    return cv2.imread(path)

# read image in grayscale
def read_image_grayscale(path):
    return cv2.imread(path, 0)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def read_image_for_edge(path):
    img = mpimg.imread(path)
    return rgb2gray(img)

# takes numpy array of image and stds value and return the image after applying gaussian noise to it
def gaussian_noise(img, sigma=0.7):
    gauss = np.random.normal(0, sigma, img.size)
    gauss = gauss.reshape(
        img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    noisy = cv2.add(img, gauss)
    return noisy

# takes grayscale image as input and returning the image after adding salt and pepper noise to it
def salt_n_pepper_noise(img):
    # Getting the dimensions of the image
    row, col = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = np.random.randint(300, 5000)
    for i in range(number_of_pixels):

        # Pick a random y coordinate
        y_coord = np.random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = np.random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = np.random.randint(300, 5000)
    for i in range(number_of_pixels):

        # Pick a random y coordinate
        y_coord = np.random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = np.random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img

# takes numpy array of image and return the image after applying uniform noise to it
def uniform_noise(img,max = 100):
    uniform = np.random.uniform(0, max, img.size)
    uniform = uniform.reshape(
        img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    noisy = cv2.add(img, uniform)
    return noisy

# takes numpy array of image as input and returns the image after applying averaging filter to it
def average_filter(img, kernel_size=3):
    row,col,ch = img.shape
    
    mask = np.ones([kernel_size,kernel_size], dtype=int)
    mask = mask/ kernel_size**2
    
    filtered_img = np.zeros([row+kernel_size-1,col+kernel_size-1,ch])
    
    for i in range(ch):
        filtered_img[:,:,i] = convolve2d(img[:,:,i],mask)        
            
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

# takes numpy array of image as input and returns the image after applying median filter to it
def median_filter(img, kernel_size):
    temp = []
    indexer = kernel_size // 2
    for i in range(len(img)):

        for j in range(len(img[0])):

            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > len(img) - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(img[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(img[i + z - indexer][j + k - indexer])

            temp.sort()
            img[i][j] = temp[len(temp) // 2]
            temp = []
    return img

# takes numpy array of image as input and returns the image after applying median filter to it
def gaussian_filter(img, kernel_size,sigma=0.7):
    row,col,ch = img.shape
    filtered_img = np.zeros([row+kernel_size-1,col+kernel_size-1,ch])
    # Create a Gaussian kernel using NumPy
    kernel = np.zeros((kernel_size,kernel_size))
    m = kernel_size//2
    for x in range(-m, m+1):
        for y in range(-m, m+1):
            kernel[x+m, y+m] = np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    # apply gaussian filter on the image
    for i in range(ch):
        filtered_img[:,:,i] = convolve2d(img[:,:,i],kernel)   
        
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

# takes numpy array of image as input and returns the image after applying sobel filter to it
def sobel_filter(img):
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], dtype=np.float32)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)

    return (G,theta)

# takes outputs of sobel filter and returns image with edge detection
def non_max_suppression(img,D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M):
        for j in range(1, N):
            try:
                q = 255
                r = 255

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z    

def threshold(img):
    LOW_THRESHOLD_RATIO = 0.09
    HIGH_THRESHOLD_RATIO = 0.17
    WEAK_PIXEL = 100
    STRONG_PIXEL = 255
    hiThresh = img.max()*HIGH_THRESHOLD_RATIO
    loThresh = hiThresh * LOW_THRESHOLD_RATIO

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    strong_i, strong_j = np.where(img >= hiThresh)

    weak_i, weak_j = np.where((img >= loThresh) & (img <= hiThresh))

    res[strong_i, strong_j] = STRONG_PIXEL
    res[weak_i, weak_j] = WEAK_PIXEL

    return res

def hysteresis(img):
    WEAK_PIXEL = 100
    STRONG_PIXEL = 255
    M, N = img.shape

    res = np.copy(img)

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == WEAK_PIXEL):
                try:
                    if (img[i+1, j-1] == STRONG_PIXEL) or (img[i+1, j] == STRONG_PIXEL) or (img[i+1, j+1] == STRONG_PIXEL) or (img[i, j-1] == STRONG_PIXEL) or (img[i, j+1] == STRONG_PIXEL) or (img[i-1, j-1] == STRONG_PIXEL) or (img[i-1, j] == STRONG_PIXEL) or (img[i-1, j+1] == STRONG_PIXEL):
                        res[i, j] = STRONG_PIXEL
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass
    return res

def canny_filter(img,D):
    result= non_max_suppression(img,D)
    result = threshold(result)
    result = hysteresis(result)
    return result


# takes image as input and returns image with edge detection
def robert_filter(img):
    r_v = np.array( [[ 0, 0, 0 ],
                                [ 0, 1, 0 ],
                                [ 0, 0,-1 ]] )

    r_h = np.array( [[ 0, 0, 0 ],
                                [ 0, 0, 1 ],
                                [ 0,-1, 0 ]] )

    r_y = ndimage.convolve( img, r_v )
    r_x = ndimage.convolve( img, r_h )

    R = np.hypot(r_x, r_y)
    theta_r = np.arctan2(r_x, r_y)
    return R

# takes image as input and returns image with edge detection
def prewitt_filter(img):
    p_v = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

    p_h = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])

    p_y = ndimage.convolve(img, p_v)
    p_x = ndimage.convolve(img, p_h)

    P = np.hypot(p_x, p_y)
    theta_p = np.arctan2(p_x, p_y)
    return P

# takes numpy array of image and return image
def write_image(image, mode, type):
    cv2.imwrite(f"Images/img with {type} {mode}.jpg", image)


