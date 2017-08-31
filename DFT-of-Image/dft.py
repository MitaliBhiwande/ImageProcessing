import cv2
import numpy as np
import cmath as cm
pi2=cm.pi*2

temp = cv2.imread('/Users/mitalibhiwande/Desktop/123.jpg')
image=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
cv2.imshow('Originalimage',image)

def dftshift(src, dst=None):
    if dst is None:
        dst = np.empty(src.shape, src.dtype)
    elif src.shape != dst.shape:
        raise ValueError("src and dst must have equal sizes")
    elif src.dtype != dst.dtype:
        raise TypeError("src and dst must have equal types")
    if src is dst:
        ret = np.empty(src.shape, src.dtype)
    else:
        ret = dst

    h, w = src.shape[:2]

    cx1 = cx2 = w/2
    cy1 = cy2 = h/2
    if w % 2 != 0:
        cx2 += 1
        if h % 2 != 0:
            cy2 += 1

    ret[h-cy1:, w-cx1:] = src[0:cy1 , 0:cx1 ]   # q1 -> q3
    ret[0:cy2 , 0:cx2 ] = src[h-cy2:, w-cx2:]   # q3 -> q1

    ret[0:cy2 , w-cx2:] = src[h-cy2:, 0:cx2 ]   # q2 -> q4
    ret[h-cy1:, 0:cx1 ] = src[0:cy1 , w-cx1:]   # q4 -> q2

    if src is dst:
        dst[:,:] = ret

    return dst

def DFT2D(image):
    dft2d_sum = image
    height = np.size(image, 0)
    width = np.size(image, 1)
    dft2d_sum = [[0.0 for k in range(width)] for l in range(height)] 
    for k in range(width):
        for l in range(height):
            sum=0.0 + 0.0j
            for m in range(width):
                for n in range(height):
                    e = cm.exp(- 1j * pi2 * ((float(k * m) / width) + (float(l * n) / height)))
                    f= image[m][n] * e
                    sum+=f
            dft2d_sum[l][k] = sum
    return (dft2d_sum)
imagedft = DFT2D(np.asarray(image))

arrdft = np.asarray(imagedft)

def IDFT2D(image):
    idft2d_sum = image
    height = np.size(image, 0)
    width = np.size(image, 1)
    idft2d_sum = [[0.0 for k in range(width)] for l in range(height)] 
    for k in range(width):
        for l in range(height):
            sum=0.0 + 0.0j
            for m in range(width):
                for n in range(height):
                    e = cm.exp( 1j * pi2 * ((float(k * m) / width) + (float(l * n) / height)))
                    f= image[m][n] * e
                    sum+=f
            idft2d_sum[l][k] = sum /width/height
    return (idft2d_sum)
imageidft = IDFT2D(arrdft)
arridft = np.asarray(imageidft)
cv2.imshow("idft", np.uint8(arridft))

image_Re=arrdft.real
image_Im=arrdft.imag
magnitude = cv2.sqrt(image_Re**2.0 + image_Im**2.0)
log_spectrum = cv2.log(1.0 + magnitude)
dftshift(log_spectrum, log_spectrum)
cv2.normalize(log_spectrum, log_spectrum, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow("magnitude", log_spectrum)

def MSE(original, recovered):
    height = np.size(image, 0)
    width = np.size(image, 1)
   # original = [[original[m][n] for m in range(width)] for n in range(height)] 
   # recovered= [[recovered[m][n] for m in range(width)] for n in range(height)]
    sum=0.0 
    for m in range(width):
        for n in range(height):
            sum+=(( (float(original[m][n])) - (float(recovered[m][n])))*((float(original[m][n])-(float(recovered[m][n])))))
    
    print("MSE:", sum/(width*height))
                   
MSE(arrdft,arridft)  

def MSE2(image1,image2):  
	err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
	err /= float(image1.shape[0] * image2.shape[1])
	print("MSE:", err)
	

MSE2(arrdft,arridft)  

      

