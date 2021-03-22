# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 23:16:14 2021

@author: evaba
"""
# =============================================================================
# =============================================================================
# # KU Leuven - Master in Artificial Intelligence
# # #Computer Vision - 1st assignment
# # #Name: Evangelia Balini
# # #Student number: r-------
# =============================================================================
# =============================================================================

import cv2
import numpy as np
import imutils
# from matplotlib import pyplot as plt

#Reading the video
cap = cv2.VideoCapture(r'C:\Users\evaba\Desktop\CV files\finalvideocv.mp4')

#Get width, height, frame rate
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))

#Specify fourcc and output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('output.mp4', fourcc, FPS, (w,h), isColor=True)


# =============================================================================
# #FUNCTION DEFINITIONS - 
# #These functions are called later while processing the video.
# =============================================================================

#Function that helps with taking specific second intervals
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

# =============================================================================
# #PART 1 - BASIC IMAGE PROCESSING
# =============================================================================

#Returns grayscale frames
def Gray_Scale(fr):
    #Convert to gray
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    #Convert gray to RGB so it can be written on the output
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out

#RGB grabbing - returns the object in white and the background in black 
def grabbingRGB(fr):
    #Colors definition
    lower = np.array([0, 0, 150])
    upper = np.array([50, 20, 255])
    #Create mask between upper and lower
    mask = cv2.inRange(fr, lower, upper)
    res = cv2.bitwise_and(fr, fr, mask = mask)
    #Where the mask is positive, convert red to white
    #To keep it red, comment out the next line
    res[mask>0] = [255,255,255]
    return res

#HSV grabbing - returns the object in white and the background in black
def grabbingHSV(img):
    #TRansform to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    
    #Add two masks together
    mask = mask1 + mask2
    res = cv2.bitwise_and(img, img, mask = mask)
    #Where the mask is positive, convert red to white.
    #To keep it red, comment out the next line
    res[mask>0] = [255,255,255]
    return res

#Binary morphological operation: dilation - Dilates frame
def dilation(frame):
    kernel = np.ones((7,7),np.uint8)
    res = cv2.dilate(frame, kernel, iterations = 1)
    return res

#Binary morphological operation: closing - Closes frame
def closing(frame):
    kernel = np.ones((7,7),np.uint8)
    res = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    return res

# =============================================================================
# #PART 2 - OBJECT DETECTION
# =============================================================================

#Performs edge detection on both x and y axis, takes as input the frame and kernel size
def edgeDetectionSobel(fr, k):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    #gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    gray = Gray_Scale(fr)
    
    #Sobel for both axis
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=k, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=k, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    res = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return res

#Performs Hough Transform for circle detection
def HoughCirclesDetection(fr, dist, param1, param2, minRadius, maxRadius):
    #Hough circle detection
    #convert to grayscale and blur
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    #Take the height of the frame
    rows = gray.shape[0]
    #Detect circles (rows/dist) is the minimum distance between the centers of the circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows/dist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    #For each circle, plot the outline and the center 
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(fr, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(fr, center, radius, (255, 0, 100), 3)

    return fr

#Matches a template (given by its path) with the image
def templateMatch(image, template_path):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    
    #Take the shape of the template to help build the rectangle later
    height, width = template.shape[::]
    
    #Match the template
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
    
    #Find where the maximum/minimum value is
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    #Take the min value and build the rectangle 
    top_left = min_loc  #min_loc is only for TM_SQDIFF
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2) 
    
    return image

#Returns the grayscale likelihood map where
#the intensity values are proportional to the likelihood of the object
#being in that position
def returnGrayMap(image, template_path):
    template = cv2.imread(template_path, 0)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Take the frame size to help resize later
    height, width = img_gray.shape[::]
    #Match template
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
    
    #Nomralize to 0-255
    normalized_res = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #Resize map to the frame size
    likelihood_res = cv2.resize(normalized_res, (width, height), interpolation=cv2.INTER_NEAREST)
    #Invert likelihood
    likelihood_res = (255-likelihood_res)
    
    #Transform grayscale map to BGR to make it possible to write to output file
    res = cv2.cvtColor(likelihood_res, cv2.COLOR_GRAY2BGR)
    
    return(res)


# =============================================================================
# #PART3 - CARTE BLANCHE
# =============================================================================

#Sharpen/unblurs a frame
def unblur(frame):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharped_img = cv2.filter2D(frame, -1, sharpen_kernel)
    return sharped_img

#Performs line detection with Hough transform
def houghLines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Find edges with Canny transform
    edges = cv2.Canny(gray,50,180,apertureSize = 3)
    #Find lines
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    #Plot each line on the frame
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    return img

#Detects eyes - with hough transform
def detectEyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)
	# Apply hough transform on the image - parameters adjusted to detect eye shapes
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/2, param1=150, param2=10, minRadius=20, maxRadius=40)
	# Draw detected circles
    if circles is not None:
	    circles = np.uint16(np.around(circles))
	    for i in circles[0, :]:
	        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
	        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img 

#Tracks the movement of the red ball
def trackball(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    
    red = mask1 + mask2
    red = cv2.GaussianBlur(red,(3,3),0)
    
    #Find contours
    cnts = cv2.findContours(red.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) > 0:
            cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.circle(frame, (rect[0][0]+(rect[-1][0] - rect[0][0])//2,rect[1][1]+(rect[-1][-1]-rect[1][1])//2), 
                   25, (0, 255, 0), -1)
    
    return frame

#Processing of the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame or stream ended. Exiting ...")
        break    
    
    if between(cap,0,1000) or between(cap,2000,3000):
        #Switching from color to grayscale
        frame = Gray_Scale(frame)
    elif between(cap, 3000, 4000):
        frame = frame
        
    elif between(cap,4000, 6000):
        #Gaussian filter with Kernel 5
        frame = cv2.GaussianBlur(frame, (5,5), 0)
    
    elif between(cap,6000, 8000):
        #Gaussian filter with Kernel 25
        frame = cv2.GaussianBlur(frame, (25,25), 0)
    
    elif between(cap,8000, 10000):
        #Bilateral filter with size 10
        frame = cv2.bilateralFilter(frame,10,75,75)
    
    elif between(cap, 10000, 12000):
        #Bilateral filter with size 30
        frame = cv2.bilateralFilter(frame,30,75,75)
   
    elif between(cap, 12000, 13000):
        frame = frame
    
    elif between(cap, 13000, 15000):
        #Grab object in RGB
        frame = grabbingRGB(frame)
    
    elif between(cap, 15000, 17000):
        #Grabbing object in HSV
        frame = grabbingHSV(frame)
    
    elif between(cap, 17000, 19000):
        #Improving HSV grabbing with dilation
        frame_between = grabbingHSV(frame)
        frame = dilation(frame_between)
    
    elif between(cap, 19000, 20000):
        #Improving HSV grabbing with closing
        frame_between = grabbingHSV(frame)
        frame = closing(frame_between)
    
    elif between(cap, 20000, 21000):
        frame = frame
        
    elif between(cap, 21000, 23000):
        #Sobel edge detector with kernel size 3
        frame = edgeDetectionSobel(frame, 3)
    
    elif between(cap, 23000, 25000):
        #Sobel edge detector with kernel size 5
        frame = edgeDetectionSobel(frame, 5)
    
    elif between(cap, 25000, 27000):
        #Detect circles with Hough Transform
        frame = HoughCirclesDetection(frame, 8, 120, 30, 1, 0)
    elif between(cap, 27000, 30000):
        frame = HoughCirclesDetection(frame, 4, 120, 30, 1, 0)
    elif between(cap, 30000, 32000):
        frame = HoughCirclesDetection(frame, 8, 180, 30, 5, 70)
    elif between(cap, 32000, 35000):
        frame = HoughCirclesDetection(frame, 12, 100, 30, 5, 70) 
    
    elif between(cap, 35000, 37000):
        #Find object with template matching
        frame = templateMatch(frame, r'C:\Users\evaba\Desktop\CV files\templatenew2.png')
    elif between(cap, 37000, 40000):
        #Build the grayscale likelihood map
        frame = returnGrayMap(frame, r'C:\Users\evaba\Desktop\CV files\templatenew2.png')
    
    elif between(cap, 40000, 41000):
        #Show blurred video
        frame = cv2.GaussianBlur(frame, (5,5), 0)
    
    elif between(cap, 41000, 45000):
        #Unblur the blurred video
        temp = cv2.GaussianBlur(frame, (5,5), 0)
        frame = unblur(temp)
        
    elif between(cap, 45000, 50000):
        #Show sudoku lines - Hough line transform
        frame = houghLines(frame)
        
    elif between(cap, 50000, 55000):
        #Detect eyes 
        #Undereye cicles unfortunately not detected :(
        frame = detectEyes(frame)
        
    elif between(cap, 55000, 60000):
        #Track red ball 
        frame = trackball(frame)
    else:
        frame = frame
    
    #Show output frame by frame
    cv2.imshow('Output video', frame)
    
    #Write frame to the output file
    output.write(frame)
    
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

#Release everything
cap.release()
output.release()
cv2.destroyAllWindows()

