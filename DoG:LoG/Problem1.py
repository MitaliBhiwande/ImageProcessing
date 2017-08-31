# -*- coding: utf-8 -*-
import cv2
import numpy as np
import scipy.signal
from scipy.misc import toimage
from scipy import ndimage
from matplotlib import pyplot as plot

originalImage=cv2.imread("/Users/mitalibhiwande/Desktop/UBCampus.jpg")
greyImage= cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
toimage(greyImage).save("/Users/mitalibhiwande/Desktop/greyImagedd.jpeg")
cv2.imshow('Original', originalImage)
cv2.imshow('OriginalGrey', greyImage)
height = np.size(greyImage, 1)
width = np.size(greyImage, 0)

DoGfilter=[[0.0,0.0,-1.0,-1.0,-1.0,0.0,0.0],
        [0.0,-2.0,-3.0,-3.0,-3.0,-2.0,0.0],
        [-1.0,-3.0,5.0,5.0,5.0,-3.0,-1.0],
        [-1.0,-3.0,5.0,16.0,5.0,-3.0,-1.0],
        [-1.0,-3.0,5.0,5.0,5.0,-3.0,-1.0],
        [0.0,-2.0,-3.0,-3.0,-3.0,-2.0,0.0],
        [0.0,0.0,-1.0,-1.0,-1.0,0.0,0.0]]
        
def applyFilterDoG(og,fl):
    DoG=scipy.signal.convolve2d(og,fl)
    return DoG
            
DoGImage=applyFilterDoG(np.asarray(greyImage), DoGfilter)
cv2.imshow('DoGImage',DoGImage)

def ZeroCross(A):
    imagezero = np.zeros(shape=(width,height), dtype=int)
    imagezero=np.asarray(imagezero)
    Image=A
    for k in range(1,width):
        for l in range(1,height):
            sum=0
            if((Image[k-1,l] > 0 and Image[k,l]<0) or (Image[k-1,l] < 0 and Image[k,l] > 0)):
                sum+=1
            elif((Image[k+1,l] > 0 and Image[k,l]<0) or (Image[k+1,l] < 0 and Image[k,l] > 0)):
                sum+=1
            elif((Image[k,l-1] > 0 and Image[k,l]<0) or (Image[k,l-1] < 0 and Image[k,l] > 0)):
                sum+=1
            elif((Image[k,l+1] > 0 and Image[k,l]<0) or (Image[k,l+1] < 0 and Image[k,l] > 0)):
                sum+=1
        
            if(sum>0.0):
                imagezero[k,l]=0
            else:
                imagezero[k,l]=1

    return imagezero
    #toimage(imagezero).save("/Users/mitalibhiwande/Desktop/zercrossed.jpeg")
#zerocrossed=cv2.imread("/Users/mitalibhiwande/Desktop/zc.jpeg")

DoGzerocrossed=ZeroCross(DoGImage)
toimage(DoGzerocrossed).save("/Users/mitalibhiwande/Desktop/DoGzercrossed.jpeg")  

gi=greyImage.astype('int32')
derix = ndimage.sobel(gi, 0)  # horizontal derivative
deriy = ndimage.sobel(gi, 1)  # vertical derivative
magnitude = np.hypot(derix, deriy)  # magnitude
magnitude *= 255.0 / np.max(magnitude)
scipy.misc.imsave('/Users/mitalibhiwande/Desktop/sobel.jpg', magnitude)
magnitude=cv2.imread('/Users/mitalibhiwande/Desktop/sobel.jpg')
ret,threshold = cv2.threshold(magnitude,35,255,cv2.THRESH_BINARY)
plot.imshow(threshold)
scipy.misc.imsave('/Users/mitalibhiwande/Desktop/thresh.jpg', threshold)
thresh=cv2.imread('/Users/mitalibhiwande/Desktop/thresh.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)


#application of sobel
def Sobel(A,B):
    thresh=A
    zerocrossed=B
    sob = [[0.0 for k in range(height)] for l in range(width)]
    for k in range(1,width):
        for l in range(1,height):
            sob[k][l]=np.absolute((thresh[k][l]) - zerocrossed[k][l])

    return sob

DoGFinal=Sobel(thresh,DoGzerocrossed)    
scipy.misc.toimage(DoGFinal).show()
scipy.misc.imsave('/Users/mitalibhiwande/Desktop/DoGFinal.jpg', DoGFinal)

LoGfilter=[[0.0,0.0,1.0,0.0,0.0],
            [0.0,1.0,2.0,1.0,0.0],
            [1.0,2.0,-16.0,2.0,1.0],
            [0.0,1.0,2.0,1.0,0.0],
            [0.0,0.0,1.0,0.0,0.0]]

def applyFilterLoG(og,fl):
    LoG=scipy.signal.convolve2d(og,fl)
    return LoG
    
        
LoGImage=applyFilterLoG(np.asarray(greyImage), LoGfilter)
cv2.imshow('LoGImage',LoGImage)

LoGzerocrossed=ZeroCross(LoGImage)
toimage(LoGzerocrossed).save("/Users/mitalibhiwande/Desktop/LoGzercrossed.jpeg")

LoGFinal=Sobel(thresh,LoGzerocrossed)
scipy.misc.toimage(LoGFinal).show()
scipy.misc.imsave('/Users/mitalibhiwande/Desktop/LoGFinal.jpg', LoGFinal)
