import cv2
import numpy as np

#A: input matrix
#B: filter

    
def applyKernel(A, B):
    A = cv2.copyMakeBorder(np.asarray(A), 1, 1, 1, 1, cv2.BORDER_REFLECT)
    wk, hk = np.asarray(A).shape
    
    G = [[0.0 for x in range(hk-2)] for y in range(wk-2)] 
    gi = 0
    gj = 0
    for i in range(1, wk-1):
        gj = 0
        for j in range(1, hk-1):
            G[gi][gj] = (A[i-1][j-1] * B[0][0]) + (A[i-1][j] * B[0][1]) + (A[i-1][j+1] * B[0][2]) + (A[i][j-1] * B[1][0]) + (A[i][j] * B[1][1]) + (A[i][j+1] * B[1][2]) + (A[i+1][j-1] * B[2][0]) + (A[i+1][j] * B[2][1]) + (A[i+1][j+1] * B[2][2])
            gj += 1
        gi += 1 
    return G
 
def downsample(G):
    wg, hg = np.asarray(G).shape
   
    wg = wg/2
    hg = hg/2
    D = [[0.0 for x in range(hg)] for y in range(wg)]
    di = 0
    dj = 0
    for i in range(0, 2*wg , 2):
        dj = 0
        for j in range(0, 2*hg , 2):
            D[di][dj] = G[i][j]
            dj += 1
        di += 1
  
    #print('D: ', D)  
    return D
    
def upsample(D):
    wd, hd = np.asarray(D).shape
    R = [[0.0 for x in range(2*hd)] for y in range(2*wd)]
    di = 0
    dj = 0
    
    for i in range (0, wd):
        dj = 0
        for j in range (0, hd):
            R[di][dj] = D[i][j]
            R[di+1][dj] = D[i][j]
            R[di][dj+1] = D[i][j]
            R[di+1][dj+1] = D[i][j]
            dj += 2
        di += 2
    return R
   
def addition(U, L):
    
     wj, hj = np.asarray(U).shape
     Sum = [[0.0 for x in range(hj)] for y in range(wj)]

     for i in range (0, wj):
         for j in range (0, hj):
             Sum[i][j] = U[i][j] + L[i][j]
    
     return Sum
  
def subtraction(U, G):
    
     wj, hj = np.asarray(U).shape
   
     Difference = [[0.0 for x in range(hj)] for y in range(wj)]

     for i in range (0, wj):
         for j in range (0, hj):
             Difference[i][j] =  G[i][j]-U[i][j]
    
     return Difference                          
          
img = cv2.imread('/Users/mitalibhiwande/Desktop/123.jpg')
greyImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
w,h = greyImage.shape
cv2.imshow('Original', greyImage)

print(w, h)
print(greyImage)
print('border:---------')

appendImage = [[0.0 for x in range(h)] for y in range(w)] 

#appendImage = cv2.copyMakeBorder(greyImage, 1, 1, 1, 1, cv2.BORDER_REFLECT)

filter = [[1.0/16, 1.0/8, 1.0/16], [1.0/8, 1.0/4, 1.0/8], [1.0/16, 1.0/8, 1.0/16]]
print(filter)

#gaussian- downscale- take gaussian- upscale - subtract from original gaussian
#G0 = applyKernel(greyImage, filter)
G0 = greyImage
cv2.imshow('G0', np.uint8(G0))
D1 = downsample(np.asarray(G0))
G1 = applyKernel(D1, filter)
UP0 = upsample(np.asarray(G1))
L1 = subtraction(UP0, G0)
cv2.imshow('L1', np.uint8(L1))

#D1 = downsample(np.asarray(G0))
#G1 = applyKernel(D1, filter)
cv2.imshow('G1', np.uint8(G1))
#D2 = cv2.pyrDown(np.asarray(G1))
D2 = downsample(np.asarray(G1))
G2 = applyKernel(D2, filter)
#UP1 =  cv2.pyrUp(np.asarray(G2))
UP1 = upsample(np.asarray(G2))
L2 = subtraction(UP1, G1)
cv2.imshow('L2', np.uint8(L2))
#
#D2 = downsample(np.asarray(G1))
#G2 = applyKernel(D2, filter)
cv2.imshow('G2', np.uint8(G2))
#D3 = cv2.pyrDown(np.asarray(G2))
D3 = downsample(np.asarray(G2))
G3 = applyKernel(D3, filter)
#UP2 =  cv2.pyrUp(np.asarray(G3))
UP2 = upsample(np.asarray(G3))
L3 = subtraction(UP2, G2)
cv2.imshow('L3', np.uint8(L3))

#D3 = downsample(np.asarray(G2))
#G3 = applyKernel(D3, filter)
cv2.imshow('G3', np.uint8(G3))
#D4 = cv2.pyrDown(np.asarray(G3))
D4 = downsample(np.asarray(G3))
G4 = applyKernel(D4, filter)
#UP3 =  cv2.pyrUp(np.asarray(G4))
UP3 = upsample(np.asarray(G4))
L4 = subtraction(UP3, G3)
cv2.imshow('L4', np.uint8(L4))
cv2.imshow('G4', np.uint8(G4))

#cv2.imshow('Upsample 1', np.uint8(UP3))
C4 = addition(UP3, L4)
cv2.imshow('RC4', np.uint8(C4))

#U3 = cv2.pyrUp(np.asarray(C4))
U3 = upsample(np.asarray(C4))
C3 = addition(U3, L3)
cv2.imshow('RC3', np.uint8(C3))

#U2 =  cv2.pyrUp(np.asarray(C3))
U2 = upsample(np.asarray(C3))
C2 = addition(U2, L2)
cv2.imshow('RC2', np.uint8(C2))

#U1 = U2 =  cv2.pyrUp(np.asarray(C2))
U1 = upsample(np.asarray(C2))
C1 = addition(U1, L1)
cv2.imshow('RC1', np.uint8(C1))


def MSE(image1, image2):
    original=image1
    recovered=image2
    
    height = np.size(image2, 0)
    width = np.size(image2, 1)
   #original = [[0.0 for k in range(width)] for l in range(height)]
    #recovered= [[0.0 for k in range(width)] for l in range(height)]
    sum=0.0
    for m in range(width):
        for n in range(height):
            f=((original[m][n].real)-(recovered[m][n].real))**2
            sum+=f
    
    #err = np.sum((image1.astype("float") - image2.astype("float"))**2)
    #err /= float(image1.shape[0]*image2.shape[1])
    print("MSE:", sum)
    
MSE(np.asarray(greyImage),np.asarray(C1))

