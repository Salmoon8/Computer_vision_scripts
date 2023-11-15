import numpy as np
import streamlit as st
import cv2 
import plotly.express as px

def normalize(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image
def gray_normalize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize(gray)    

def write_image(image, name):
    return cv2.imwrite(f"Normalized Image {name}.jpg", image)

def one_channel_hist(channel):
    hist_channel = [0] * 256
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            intensity = channel[i,j]
            hist_channel[intensity] += 1
    max_count = max(hist_channel)
    for i in range(len(hist_channel)):
        hist_channel[i] = int(hist_channel[i] / max_count * 255)
    return hist_channel              


def rgb_hist(image):
    blue, green, red = cv2.split(image)
    hist_blue = one_channel_hist(blue)
    hist_green = one_channel_hist(green)
    hist_red = one_channel_hist(red)
    return hist_blue , hist_green , hist_red

def gray_hist(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return one_channel_hist(gray)

def plot_curve(gray_histogram , hist_red , hist_green , hist_blue):
    fig = px.line( y= gray_histogram,  color_discrete_sequence=['gray'] , width=550)
    fig.add_trace(px.line( y= hist_red, color_discrete_sequence=['red']).data[0])
    fig.add_trace(px.line( y= hist_green, color_discrete_sequence=['green']).data[0])
    fig.add_trace(px.line( y= hist_blue, color_discrete_sequence=['blue']).data[0])
    fig.update_layout(title="Distribution Curve", xaxis_title="Value", yaxis_title="Frequency" )           
    return fig
    
def plot_hist( gray_histogram , hist_red , hist_green , hist_blue):
    fig = px.histogram(gray_histogram, nbins=255, color_discrete_sequence=['gray'], opacity=0.7 ,width=550 )
    fig.add_trace(px.histogram(hist_red, nbins=255, color_discrete_sequence=['red'], opacity=0.7).data[0])
    fig.add_trace(px.histogram(hist_green, nbins=255, color_discrete_sequence=['green'], opacity=0.7).data[0])
    fig.add_trace(px.histogram(hist_blue, nbins=255, color_discrete_sequence=['blue'], opacity=0.7).data[0])
    fig.update_layout(title="Histogram", xaxis_title="Value", yaxis_title="Frequency"  )           
    return fig

def select_data(selected_data , fig):
    # Update the figure based on the user's selection
    if selected_data == "Gray Scale":
        fig.data[0].visible = True
        fig.data[1].visible = False
        fig.data[2].visible = False
        fig.data[3].visible = False
    elif selected_data == "Red Channel":
        fig.data[0].visible = False
        fig.data[1].visible = True
        fig.data[2].visible = False
        fig.data[3].visible = False  
    elif selected_data == "Green Channel":
        fig.data[0].visible = False
        fig.data[1].visible = False
        fig.data[2].visible = True
        fig.data[3].visible = False
    elif selected_data == "Blue Channel":
        fig.data[0].visible = False
        fig.data[1].visible = False
        fig.data[2].visible = False
        fig.data[3].visible = True    
    elif selected_data == "All Histograms":
        fig.data[0].visible = True
        fig.data[1].visible = True
        fig.data[2].visible = True
        fig.data[3].visible = True
    
    st.plotly_chart(fig)
    return


def one_channel_equalize(channel):    
    # Compute the histogram of the image
    histogram = one_channel_hist(channel) 
    # Compute the cumulative distribution function (CDF) of the histogram
    cdf = np.array([0] * 256)
    sum = 0
    for i in range(len(histogram)):
        sum += histogram[i]
        cdf[i] = sum
    cdf_norm = cdf * 255 / cdf[-1]
    # Create a lookup table that maps each input pixel value to its corresponding output pixel value
    lut = np.round(cdf_norm).astype(np.uint8)
    # Apply the lookup table to the image to obtain the equalized image
    img_eq = cv2.LUT(channel, lut)
    return img_eq

def rgb_equalize(image):
    blue, green, red = cv2.split(image)
    equalized_blue = one_channel_equalize(blue)
    equalized_green = one_channel_equalize(green)
    equalized_red = one_channel_equalize(red)
    img_eq = cv2.merge((equalized_blue,equalized_green, equalized_red))
    return img_eq
    
        
def gray_equalize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return one_channel_equalize(gray)