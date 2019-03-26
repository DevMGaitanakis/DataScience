# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:14:07 2019
@author: fro
"""
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
# load the image, convert it to grayscale, and blur it to remove noise
image = cv2.imread("sample5.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# threshold the image
ret,thresh= cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)

# dilate the white portions
dilate = cv2.dilate(thresh, None, iterations=2)
def segmentation(arr):
    list = []
    for i in range(len(arr)):
        if sum(arr[i])== 0:
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
    return list,nlist      
list,nlist = segmentation(dilate)
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
    '''
pos=1   
for i in range (0,len(nlist),2):
    try:
        for z in range (0,len(letterpos[pos]),2):
           rowtop,rowbottom = nlist[i],nlist[i+1]
           letterfront,letterend= letterpos[pos][i],letterpos[pos][i+1]
           new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])              
    except IndexError:
                pass   

'''
dilate2= dilate
image = cv2.cvtColor(dilate2,cv2.COLOR_GRAY2BGR)
#dilate2 = cv2.rectangle(dilate2,pt1=(100,100),pt2=(100,100),color=(0,255,0),thickness=100)
all_letters = np.zeros(shape=(1,28,28))
pos=0  
row=1  
notempty=[]
notemptySize=0
inc=0
if len(nlist) < 3 and len(letterpos)==1 :
        for i in range (0,len(letterpos[1]),2):
            print(i)
            try:
                for z in range (0,len(letterpos),2):
                   print("z = ",z)
                   rowtop,rowbottom = nlist[0],nlist[1]
                   letterfront,letterend= letterpos[row][inc],letterpos[row][inc+1]
                   new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])
                   noUse,trimmed = segmentation(new)
                   notempty.append(trimmed)
                   #if len(notempty[notemptySize]) == 0:
                       #print('yes')
                   cv2.rectangle(image,pt1=(letterpos[1][pos],nlist[0]),pt2=(letterpos[1][pos+1],nlist[1]),color=(0,255,0),thickness=2)
                   #else:
                       #print('no')
                       #new = new[trimmed[0]:trimmed[1],:]
                       #cv2.rectangle(image,pt1=(letterpos[1][pos],nlist[0]-300),pt2=(letterpos[1][pos+1],nlist[1]-abs(notempty[inc][1] - len(noUse))),color=(0,255,0),thickness=2)
                   new = np.pad(new, (5, 5), 'minimum')
                   resized_image = cv2.resize(new, (28, 28))
                   resized_image2 = np.expand_dims(resized_image, axis=0)
                   all_letters = np.append(all_letters,resized_image2,axis=0)
                   notemptySize+=1
            except IndexError:
                            pass           
            pos+=2
            inc+=1
            if len(letterpos)> 1:
                row=1 
        all_letters = np.delete(all_letters,[0],axis=0)
else:
        for i in range (0,len(letterpos),2):
            print(i)
            try:
                for z in range (0,len(letterpos[i]),2):
                   print("z = ",z)
                   rowtop,rowbottom = nlist[0],nlist[1]
                   letterfront,letterend= letterpos[row][inc],letterpos[row][inc+1]
                   new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])
                   noUse,trimmed = segmentation(new)
                   notempty.append(trimmed)
                   #if len(notempty[notemptySize]) == 0:
                       #print('yes')
                   cv2.rectangle(image,pt1=(letterpos[1][pos],nlist[0]),pt2=(letterpos[1][pos+1],nlist[1]),color=(0,255,0),thickness=2)
                   #else:
                       #print('no')
                       #new = new[trimmed[0]:trimmed[1],:]
                       #cv2.rectangle(image,pt1=(letterpos[1][pos],nlist[0]-300),pt2=(letterpos[1][pos+1],nlist[1]-abs(notempty[inc][1] - len(noUse))),color=(0,255,0),thickness=2)
                   new = np.pad(new, (5, 5), 'minimum')
                   resized_image = cv2.resize(new, (28, 28))
                   resized_image2 = np.expand_dims(resized_image, axis=0)
                   all_letters = np.append(all_letters,resized_image2,axis=0)
                   notemptySize+=1
            except IndexError:
                            pass           
            pos+=2
            inc+=1
            if len(letterpos)> 1:
                row=1 
        all_letters = np.delete(all_letters,[0],axis=0)    
plt.imshow(image)
#-notempty[1][0]



























if len(nlist) == 2:
    for i in range (0,len(letterpos[1]),2):
        print(i)
        
    
cv2.rectangle(image,pt1=(letterpos[1][2]-notempty[1][0],nlist[0]),pt2=(letterpos[1][3],nlist[1]-notempty[1][1]),color=(0,255,0),thickness=2)
#dilate2 = np.resize(dilate2, (dilate2[0].size, dilate2[1].size, 3))

if notemptySize[0]
'''else:
    pos=1    
    for i in range (0,len(nlist),2):
        try:
            for z in range (0,len(letterpos[pos]),2):
               rowtop,rowbottom = nlist[i],nlist[i+1]
               letterfront,letterend= letterpos[pos][i],letterpos[pos][i+1]
               noUse,trimmed = segmentation(new)
               new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])
               new= new[trimmed[0]:trimmed[1],:]
               new = np.pad(new, (5, 5), 'minimum')
               resized_image = cv2.resize(new, (28, 28))
               resized_image2 = np.expand_dims(resized_image, axis=0)
               all_letters = np.append(all_letters,resized_image2,axis=0)
        except IndexError:
                        pass           
        pos+=1 
    all_letters = np.delete(all_letters,[0],axis=0)

 '''              
charachter_trimmed = new[trimmed[0]:trimmed[1],:]
               
               
rowtop,rowbottom = nlist[0],nlist[0+1]
letterfront,letterend= letterpos[1][2],letterpos[0][0+1]
new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])

rowtop,rowbottom = 6,156
letterfront,letterend= 251,278
new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])

plt.imshow(new[7:33,:])
newtop,newbot = segmentation(new)


new = np.array(dilate[nlist[2]:nlist[2+1],:])
new = np.array(dilate[69:157,162:219])
plt.imshow(all_letters[1])
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
