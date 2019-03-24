# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:14:07 2019
@author: fro
"""

# import the necessary packages
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
# load the image, convert it to grayscale, and blur it to remove noise
image = cv2.imread("sample1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# threshold the image
ret,thresh= cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)

# dilate the white portions
dilate = cv2.dilate(thresh, None, iterations=2)

list = []
for i in range(len(dilate)):
    if sum(dilate[i])== 0:
        list.append(i)
    else:
        list.append(0)
nlist=[]
try:
    if list[0] < list[1]:
        for i in range(1,len(list)):
            if list[i] > list[i+1] or list[i] == 0 and list[i+1] > list[i] :
                nlist.append(i)
except IndexError:
    pass  
list2 = []
suma = 0
row = 1
rows = {}

for i in range (0,len(nlist),2):
    for x in range(len(dilate[1])):
        for z in range(nlist[i],nlist[i+1]):
            suma += dilate[z][x]
        if suma == 0:
            list2.append(x)
        else:
            list2.append(0)
            rows[row] = list2       
        suma = 0
    row+=1 
    list2 =[]   
    
nlist2=[]
letterpos = {}

for y in range(1,len(rows)+1):
    if rows[y][0] < rows[y][1]:
        for i in range(1,len(rows[1])):
            try:
                if rows[y][i] > rows[y][i+1]:
                    nlist2.append(rows[y][i])
                    letterpos[y] = nlist2
                elif rows[y][i] < rows[y][i+1] and rows[y][i] == 0 :
                    nlist2.append(rows[y][i+1])
                    letterpos[y] = nlist2
            except IndexError:
                pass   
                    
    nlist2 = []



new = np.array(dilate[nlist[2]:nlist[2+1],:])
new = np.array(dilate[22:134,136:222])
plt.imshow(new)


# find contours in the image
cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
orig = image.copy()
i = 0
resized_image3 = np.zeros(shape=(1,28,28))
for cnt in cnts:
    if(cv2.contourArea(cnt) < 100):
        continue
    x,y,w,h = cv2.boundingRect(cnt) # countours are detected
    roi = image[y:y+h, x:x+w]    #  extracting the region of intereset ROI 
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
    new_r= cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(new_r, (28, 28)) 
    resized_image2 = np.expand_dims(resized_image, axis=0)
    resized_image3 = np.append(resized_image3,resized_image2,axis=0)

    #cv2.imwrite("roi" + str(i) + ".png", roi)
    i = i + 1 

#cv2.imshow("Image", orig) 
#cv2.waitKey(0)
resized_image3 = np.delete(resized_image3,[0],axis=0)
#plt.imshow(resized_image3[3])
