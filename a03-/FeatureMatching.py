import cv2
import numpy as np 
width = 0
height=0
h=0
w=0


def calculateSSD(desc_image1,desc_image2):
    sum_square = 0
    for m in range(len(desc_image2)-1):
        sum_square += (desc_image1[m] - desc_image2[m]) ** 2
        
    SSD = - (np.sqrt(sum_square))
    return SSD
    


def calculate_NCC(desc_image1, desc_image2):


    normlized_output1 = (desc_image1 - np.mean(desc_image1)) / (np.std(desc_image1))
    normlized_output2 = (desc_image2 - np.mean(desc_image2)) / (np.std(desc_image2))
    correlation_vector = np.multiply(normlized_output1, normlized_output2)
    NCC = float(np.mean(correlation_vector))

    return NCC



def feature_matching_temp (descriptor1,descriptor2,method):


    keyPoints1 = descriptor1.shape[0]
    keyPoints2 = descriptor2.shape[0]

    #Store matching scores
    matched_features = []

    for kp1 in range(keyPoints1):
        # Initial variables (will be updated)
        distance = -np.inf
        y_index = -1
        for kp2 in range(keyPoints2):
            # Choose methode (ssd or normalized correlation)
            if method=="SSD":
               score = calculateSSD(descriptor1[kp1], descriptor2[kp2])
            elif method =="NCC":
                score = calculate_NCC(descriptor1[kp1], descriptor2[kp2])


            if score > distance:
                distance = score
                y_index = kp2

        feature = cv2.DMatch()
        #The index of the feature in the first image
        feature.queryIdx = kp1
        # The index of the feature in the second image
        feature.trainIdx = y_index
        #The distance between the two features
        feature.distance = distance
        matched_features.append(feature)

    return matched_features





