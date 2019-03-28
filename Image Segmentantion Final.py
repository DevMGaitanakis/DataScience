# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:14:07 2019
@author: fro
"""

# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Bosstest.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

dilate2= dilate
image = cv2.cvtColor(dilate2,cv2.COLOR_GRAY2BGR)
all_letters = np.zeros(shape=(1,28,28))

rows=len(letterpos)
posa=0  
notempty=[]
notemptySize=0
inc=0

for a in range(1,rows+1):
    for b in range (0,len(letterpos[a]),2):
        try:
           print('One',letterpos[a][b], 'two' ,letterpos[a][b+1])
           print('Nlist',nlist[inc],nlist[inc+1])
           rowtop,rowbottom = nlist[inc],nlist[inc+1]
           letterfront,letterend= letterpos[a][b],letterpos[a][b+1]
           new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])
           noUse,trimmed = segmentation(new)
           notempty.append(trimmed)
           if len(notempty[notemptySize]) == 0:
               cv2.rectangle(image,pt1=(letterpos[a][b],nlist[inc]),pt2=(letterpos[a][b+1],nlist[inc+1]),color=(0,255,0),thickness=2)
           elif len(notempty[notemptySize]) > 1:
               new = new[trimmed[0]:trimmed[1],:]
               cv2.rectangle(image,pt1=(letterpos[a][b],nlist[inc]+notempty[posa][0]),pt2=(letterpos[a][b+1],nlist[inc+1]-abs(notempty[posa][1] - len(noUse))),color=(0,255,0),thickness=2)
                         
           elif len(notempty[notemptySize]) ==1 :
               new = new[trimmed[0]:,:]
               cv2.rectangle(image,pt1=(letterpos[a][b],nlist[inc]+notempty[posa][0]),pt2=(letterpos[a][b+1],nlist[inc+1]),color=(0,255,0),thickness=2)
           new = np.pad(new, (5, 5), 'minimum')
           resized_image = cv2.resize(new, (28, 28))
           resized_image2 = np.expand_dims(resized_image, axis=0)
           all_letters = np.append(all_letters,resized_image2,axis=0)
           notemptySize+=1
        except IndexError:
                        pass           
        posa+=1      
    inc+=2

all_letters = np.delete(all_letters,[0],axis=0)
plt.imshow(image)

#rowtop,rowbottom = 137,159
#letterfront,letterend= 325,354
#new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])
#plt.imshow(new)
