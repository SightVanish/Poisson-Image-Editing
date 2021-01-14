import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


'''helping function'''
def showImag(img):
    '''
    plt.imshow(img)
    plt.axis('off') # turn off axis
    plt.show()
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)

def add(array, element):
    result = np.concatenate((array,[element]))
    return result


# Read images : obj image will be cloned into im
obj = cv2.imread("obj.png")
# denoising to obj
obj = cv2.GaussianBlur(obj,(5,5),0)
im = cv2.imread("bg.png")
# resie picture to ideal isze
obj = cv2.resize(obj, (20,40))
im = cv2.resize(im, (200, 130))


'''implemented in cv2'''
H,W = obj.shape[:2]
# resize the object
# obj = cv2.resize(obj, (W, H-150), cv2.INTER_CUBIC)

# Create an all white mask--no influence
mask = 255 * np.ones(obj.shape, obj.dtype)

# The location of the center of the obj in the im
width, height, channels = im.shape

center = (int(4*height/5), int(width/4))

# Seamlessly clone obj into im and put the results in output
# result = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
# result = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
# showImag(result)


'''Possion Editing Implement'''
def computeGradient(img):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return [dx, dy]

# compute gradient
obj_Gradient = computeGradient(obj)
bg_Gradient = computeGradient(im)
# mask gradient
for m in range(2):
    for i in range(obj_Gradient[0].shape[0]):
        for j in range(obj_Gradient[0].shape[1]):
            for k in range(3):
                bg_Gradient[m][i+3][j+1][k] = obj_Gradient[m][i][j][k]
# compute divergence
lap = computeGradient(bg_Gradient[0])[0] + computeGradient(bg_Gradient[1])[1]
#  showImag(lap)

# compute A matrix
# init the size of A
l = lap.shape[0]
w = lap.shape[1]
# init row, col, data to discribe a sparse matrix
row = np.array([])
col = np.array([])
data = np.array([])

for i in range(w):
    for j in range(l):
        # go through all l*w points for onyl once
        if i == 0 or i == w-1 or j == 0 or j == l-1:
            row = np.concatenate((row,[l*i+j]))
            col = np.concatenate((col,[l*i+j]))
            data = np.concatenate((data,[1]))
        else:
            row = np.concatenate((row,[l*i+j]))
            col = np.concatenate((col,[l*i+j]))
            data = np.concatenate((data,[-4]))

            row = np.concatenate((row,[l*i+j]))
            col = np.concatenate((col,[l*i+j-1]))
            data = np.concatenate((data,[1]))

            row = np.concatenate((row,[l*i+j]))
            col = np.concatenate((col,[l*i+j+1]))
            data = np.concatenate((data,[1]))

            row = np.concatenate((row,[l*i+j]))
            col = np.concatenate((col,[l*(i-1)+j]))
            data = np.concatenate((data,[1]))

            row = np.concatenate((row,[l*i+j]))
            col = np.concatenate((col,[l*(i+1)+j]))
            data = np.concatenate((data,[1]))

A = csc_matrix((data, (row, col)), shape=(l*w, l*w), dtype=int)
'''
# solve matrix
b_R = np.arange(l*w)
b_G = np.arange(l*w)
b_B = np.arange(l*w)

for i in range(l):
    for j in range(w):
        b_R[i*w+j] = lap[i][j][0]
        b_G[i*w+j] = lap[i][j][1]
        b_B[i*w+j] = lap[i][j][2]
        

x_R = spsolve(A, b_R).reshape([l, w])
x_G = spsolve(A, b_G).reshape([l, w])
x_B = spsolve(A, b_B).reshape([l, w])

x = np.arange(l*w*3).reshape([l, w, 3])
for i in range(l):
    for j in range(w):
        for k in range(3):
            x[i][j][0] = b_R[i*w+j]
            x[i][j][1] = b_G[i*w+j]
            x[i][j][2] = b_B[i*w+j]

'''

lap = lap.reshape([l*w, 3])
x = spsolve(A, lap)
x = x.reshape([l, w, 3]).astype(np.uint8)

print(x.shape)
showImag(x)
