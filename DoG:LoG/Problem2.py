import cv2
import numpy as np
from scipy.misc import toimage


originalImage=cv2.imread("/Users/mitalibhiwande/Desktop/MixedVegetables.jpg")
greyImage= cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', originalImage)
cv2.imshow('OriginayGrey', greyImage)
height = np.size(greyImage, 0)
width = np.size(greyImage, 1)
ht=(height*2)+1
wd=(width*2)+1
tempimage=np.zeros((ht,wd),dtype='int32')
print height,width
print ht,wd

r=1
c=0
for x in range (0, height-1):
    c=0
    for y in range(0, width-1):
        c=c+1
        tempimage[x+r][y+c]=greyImage[x,y]
    r=r+1
    
print tempimage

#save the super grid image data structure
        
toimage(tempimage).save("/Users/mitalibhiwande/Desktop/tempimage.jpg")

# compute values for crack edges
for x in range(1, ht-2):
    for y in range(2, wd-3):
        tempimage[x][y]=np.absolute(tempimage[x][y-1] - tempimage[x][y+1])
        y=y+1
    x=x+1

print tempimage   
for x in range(2, ht-2):
    for y in range(1,wd-1):
        tempimage[x][y]=np.absolute(tempimage[x-1][y] - tempimage[x+1][y] )
        y=y+1
    x=x+1

  
   
#insert 0's or 255 according to the decided threshold in the above obtained image
for x in range(1, ht-2):
    for y in range(2, wd-3):
        if(tempimage[x][y] > 27):
           tempimage[x][y]=255
        else:
            tempimage[x][y]=0
        y=y+1
    x=x+1  
    
for x in range(2, ht-2):
    for y in range(1,wd-1):
        if(tempimage[x][y] > 27):
           tempimage[x][y]=255
        else:
            tempimage[x][y]=0
        y=y+1
    x=x+1  

#toimage(tempimage).save("/Users/mitalibhiwande/Desktop/crackedge.jpg")

#print tempimage
#toimage(tempimage).show()
        
label=1      
second=np.zeros((ht,wd),dtype='int32')

#average or merge the pixels with 0 intensity depending upon the value of its neighbouring pixel intesities.

for x in range(1, ht-2):
    for y in range(2, wd-3):
        if(tempimage[x,y]==0):
            tempimage[x][y]=(np.absolute(tempimage[x][y-1] - tempimage[x][y+1]))/2
        y=y+1
    x=x+1
    
            
for x in range(2, ht-3):
    for y in range(1,wd-2):
        if(tempimage[x,y]==0):
            tempimage[x][y]=(np.absolute(tempimage[x-1][y] - tempimage[x+1][y] ))/2
            
        y=y+1
    x=x+1 
    
print tempimage

toimage(tempimage).show()
