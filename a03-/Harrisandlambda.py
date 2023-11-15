import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2 as cv
from scipy import linalg
import time
from werkzeug.utils import secure_filename
import os


def harrisoperator(img,window=5,k=0.05,q=0.999):
    """"" inputs are the image , the blocksize used for detection (neighborhood size), k is a constant used in
                R score calculations (recommended value is between 0.04:0.06 , quantile percent for non maximum supression) """
    
    """""returns a boolean matrix of the same size as the input image, with the points of interest (corners) masked with the true and the rest is False """"" 
    
    
    # first convert to grayscale if RGB
    if len(img)>=3:
        colored=np.copy(img) # keep color so when we show back the image
        img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
         
        
        
    # smooth the image (optional)
    img= cv.GaussianBlur(img,(5,5),0)
        
    
    # find gradients by sobel derivative 
    
    sobelx = np.array([[ -1 , 0 , 1 ] ,
                    [ -2 , 0 , 2 ] ,
                    [ -1 , 0 , 1 ]])
    sobely = sobelx.transpose()
    
    # spatial derivatives
    
    ix=signal.convolve2d( img , sobelx ,'same') 
    iy=signal.convolve2d( img , sobely ,'same') 
    
    # calculate structure tensor elements 
    ixx=np.multiply( ix, ix) # squared ix
    iyy=np.multiply( iy, iy) # squared iy 
    ixy=np.multiply( ix, iy) # ix * iy
    
    #apply a box window over spatial derivatives to be able to detect larger corner in a neighbor hood instead of a single pixel
    ixx=cv.blur(ixx,(window,window))
    iyy=cv.blur(iyy,(window,window))
    ixy=cv.blur(ixy,(window,window))
    
    # matrix calculations and R score 
    
    detm=  np.multiply(ixx,iyy) - np.multiply(ixy,ixy) 
    trace= ixx+iyy 
    r=   detm - k * trace**2
    #print (r.shape)
    
    # non maximum suppresion
    corners = np.abs(r) >  np.quantile( np.abs(r),q)

    # plot output 
    plt.imshow(colored,zorder=1)
    
    corners_pos = np.argwhere(corners)
    
    plt.scatter(corners_pos[:,1],corners_pos[:,0],zorder=2, c = 'b',marker ='x')

    path= "cornersoutput"+str(time.time())+".png" 
    path= secure_filename(path)
    
    plt.savefig(os.path.join("images", path))

    
    return path




def lambdamin(img,window=3,q=0.999):
    
    """"" inputs are the image , the blocksize used for detection (neighborhood size), quantile percent for non maximum supression) """
    
    """""returns a boolean matrix of the same size as the input image, with the points of interest (corners) masked with the true and the rest is False """"" 
    
    
    
    # first convert to grayscale if RGB
    if len(img)>=3:
        colored=np.copy(img) # keep color so when we show back the image
        img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    # smooth the image 
    img= cv.GaussianBlur(img,(5,5),0)
        
    
    # find gradients by sobel derivative 
    
    sobelx = np.array([[ -1 , 0 , 1 ] ,
                    [ -2 , 0 , 2 ] ,
                    [ -1 , 0 , 1 ]])
    sobely = sobelx.transpose()
    
    # spatial derivatives
    
    ix=signal.convolve2d( img , sobelx ,'same') 
    iy=signal.convolve2d( img , sobely ,'same') 
    
    # calculate structure tensor elements 
    ixx=np.multiply( ix, ix) # squared ix
    iyy=np.multiply( iy, iy) # squared iy 
    ixy=np.multiply( ix, iy) # ix . iy
    
    # improve performance by applying a window
    
    ixx=cv.blur(ixx,(window,window))
    iyy=cv.blur(iyy,(window,window))
    ixy=cv.blur(ixy,(window,window))
    
    # get H matrix for each element and find lambda min
    rows,cols= img.shape
    lambdamat=np.zeros_like(img)
    for i in range (rows):
        for j in range (cols):
            H=[[ixx[i,j],ixy[i,j]],[ixy[i,j],iyy[i,j]]]
            #print(H)
            eigvals=linalg.eigvals(H)
            try: 
                lambdamin= np.min(eigvals[np.nonzero(eigvals)]) # to avoid getting 0 as eigen value for all pixels
            except :
                lambdamin=0  # when both are zero we set it to zero
                
                
            lambdamat[i,j]=lambdamin
            
    # apply non maximal supression to get the highest lambda values
    
    lambdamat=np.abs(lambdamat) >  np.quantile( np.abs(lambdamat),q)

    # plot output

    plt.imshow(colored,zorder=1)
    
    corners_pos = np.argwhere(lambdamat)
    
    plt.scatter(corners_pos[:,1],corners_pos[:,0],zorder=2, c = 'r',marker ='o')

    path="/images/cornersoutput"+str(time.time())
    
    path= "cornersoutput"+str(time.time())+".png" 
    path= secure_filename(path)
    
    plt.savefig(os.path.join("images", path))


    
    return path


            
    
    
    
    


# def Harris_Edge_Detector(Img_Path,Window_Size=3,K=0.5):
#     harris_time_start = time.time()
#     """ Compute Harris operator using hessian matrix of the image
#     input : image
#     Return: Harris operator
#     """
#     #Img= Read_Img(Img_Path)
#     src = cv2.imread(Img_Path)
#     img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
#     # 1.calculate Ix , Iy ( Dervatives in X & Y direction)...Sobel
#     Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Window_Size)
#     Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Window_Size)

#     # 2. Hessian Matrix Calculation H=([Ixx , Ixy],[Ixy , Iyy]) where Ixx = Ix^2 ...
#     Ixx=np.multiply(Ix,Ix)
#     Iyy=np.multiply(Iy,Iy)
#     Ixy=np.multiply(Ix,Iy)

#     # 3. Image Smoothing (Gaussian Filter)
#     Ixx = cv2.GaussianBlur(Ixx,(Window_Size,Window_Size),0)
#     Iyy = cv2.GaussianBlur(Iyy,(Window_Size,Window_Size),0)
#     Ixy = cv2.GaussianBlur(Ixy,(Window_Size,Window_Size),0)

#     # 4. Computing Response Function [ R = det(H) - k*(Trace(H))^2 ]
#     det_H = Ixx*Iyy - Ixy**2
#     trace_H = Ixx + Iyy
#     R = det_H - K*(trace_H**2) 
#     harris_time_end = time.time()
#     print(f"Execution time of Harris Algorithm is {harris_time_end - harris_time_start}  sec")
#     return R


    