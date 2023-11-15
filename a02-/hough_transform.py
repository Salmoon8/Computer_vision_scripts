import numpy as np 
import matplotlib.pyplot as plt
import cv2
from skimage.transform import hough_line_peaks


def houghLine(image):
    ''' Basic Hough line transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space
             thetas : values of theta (-90 : 90)
             rs : values of radius (-max distance : max distance)
    '''
        #Get image dimensions
    # y for rows and x for columns 
    Ny = image.shape[0]
    Nx = image.shape[1]

    #Max diatance is diagonal one 
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
    # initialize parameter space rs, thetas
 # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90))
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    for y in range(Ny):
        for x in range(Nx):
            # Check if it is an edge pixel
            #  NB: y -> rows , x -> columns
            if image[y,x] > 0:
                for k in range(len(thetas)):
                            # Calculate space parameter
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                       
                 # Update the accumulator
                 # N.B: r has value -max to max
                 # map r to its idx 0 : 2*max
                    accumulator[int(r) + Maxdist,k] += 1
    return accumulator, thetas, rs

def read_image(image_path):
    return cv2.imread(image_path)
    
def read_image_grayscale(image_path):
    return cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

def Canny(img):
    return cv2.Canny(img,150,150)

def add_lines(img,accumulator,thetas,rhos):
    fig, axes = plt.subplots(1, 1)
    axes.imshow(img,cmap="gray")
    axes.set_ylim((img.shape[0], 0))
    axes.set_xlim((0,img.shape[1]))
    axes.set_axis_off()
    for _, angle, dist in zip(*hough_line_peaks(accumulator, thetas, rhos)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        axes.axline((x0, y0), slope=np.tan(angle + np.pi/2),color="red")
    plt.tight_layout()
    fig.savefig("result.jpg")