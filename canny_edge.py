import numpy, skimage, skimage.color, skimage.io, pylab, scipy.signal, scipy.ndimage.filters
import os
from PIL import Image
from skimage import feature
import math

def thresholding(I, T_high, T_low, x, y):
    if(I[x][y][0] >= T_high):
        return 1.0
    """if(I[x][y][0] >= T_low):
        return 0.25"""
    if(I[x][y][0] >= T_low):
        toVisitList = [(x, y)]
        visitedList = []
        while(len(toVisitList) != 0):
            temp = toVisitList.pop()
            x2 = temp[0]
            y2 = temp[1]
            visitedList.append((x2, y2))
            if((x2 > 0) and (y2 > 0)):
                if(I[x2-1][y2-1][0] >= T_high and (x2-1, y2-1) not in visitedList):
                    return 1.0
                elif(I[x2-1][y2-1][0] >= T_low and (x2-1, y2-1) not in visitedList):
                    toVisitList.append((x2-1, y2-1))
                if(I[x2-1][y2][0] >= T_high and (x2-1, y2) not in visitedList):
                    return 1.0
                elif(I[x2-1][y2][0] >= T_low and (x2-1, y2) not in visitedList):
                    toVisitList.append((x2-1, y2))
                if(I[x2][y2-1][0] >= T_high and (x2, y2-1) not in visitedList):
                    return 1.0
                elif(I[x2][y2-1][0] >= T_low and (x2, y2-1) not in visitedList):
                    toVisitList.append((x2, y2-1))
            if((x2 < I.shape[0]-1) and (y2 < I.shape[1]-1)):
                if(I[x2+1][y2+1][0] >= T_high and (x2+1, y2+1) not in visitedList):
                    return 1.0
                elif(I[x2+1][y2+1][0] >= T_low and (x2+1, y2+1) not in visitedList):
                    toVisitList.append((x2+1, y2+1))
                if (I[x2][y2+1][0] >= T_high and (x2, y2+1) not in visitedList):
                    return 1.0
                elif (I[x2][y2+1][0] >= T_low and (x2, y2+1) not in visitedList):
                    toVisitList.append((x2, y2+1))
                if (I[x2+1][y2][0] >= T_high and (x2+1, y2) not in visitedList):
                    return 1.0
                elif (I[x2+1][y2][0] >= T_low and (x2+1, y2) not in visitedList):
                    toVisitList.append((x2+1, y2))
            if((x2 < I.shape[0]-1) and (y2 > 0)):
                if (I[x2+1][y2-1][0] >= T_high and (x2+1, y2-1) not in visitedList):
                    return 1.0
                elif (I[x2+1][y2-1][0] >= T_low and (x2+1, y2-1) not in visitedList):
                    toVisitList.append((x2+1, y2-1))
            if((x2 > 0) and (y2 < I.shape[0]-1)):
                if (I[x2-1][y2+1][0] >= T_high and (x2-1, y2+1) not in visitedList):
                    return 1.0
                elif (I[x2-1][y2+1][0] >= T_low and (x2-1, y2+1) not in visitedList):
                    toVisitList.append((x2-1, y2+1))
    return 0.0


if __name__ == '__main__':
    T_high = 0.015
    T_low = 0.001

    # img = Image.open('flower.jpg')
    # img.show()
    A = skimage.img_as_float(skimage.io.imread('flower.jpg'))

    #print(A.shape)
    lum = skimage.color.rgb2grey(A)

    #gaussian = scipy.ndimage.filters.gaussian_filter(lum, 1)

    print("Gaussian smoothing...")

    gau = numpy.array([[0.003, 0.013, 0.022, 0.013, 0.003],
                       [0.013, 0.059, 0.097, 0.059, 0.013],
                       [0.022, 0.097, 0.159, 0.097, 0.022],
                       [0.013, 0.059, 0.097, 0.059, 0.013],
                       [0.003, 0.013, 0.022, 0.013, 0.003]])

    myGaussian = scipy.signal.convolve2d(lum, gau)
    print("Computing smoothed gradients...")
    xGrad, yGrad = numpy.gradient(myGaussian)

    """edgeStrength = numpy.empty(myGaussian.shape, float)
    edgeOrientation = numpy.empty(myGaussian.shape, float)
    for x in range(0, myGaussian.shape[0]):
        for y in range(0, myGaussian.shape[1]):
            edgeStrength[x][y] = math.sqrt((xGrad[x][y])**2 + (yGrad[x][y])**2)
            edgeOrientation[x][y] = math.atan2(yGrad[x][y], xGrad[x][y])"""

    print("Computing edge strength and orientation...")

    edgeStrength = numpy.sqrt(numpy.power(xGrad, 2) + numpy.power(yGrad, 2))
    edgeOrientation = numpy.arctan2(yGrad, xGrad)

    I = numpy.empty(A.shape)

    print("Non-maximum suppression...")

    for a in range(2, myGaussian.shape[0]-2):
        for b in range(2, myGaussian.shape[1]-2):
            horizontal = min(math.fabs(edgeOrientation[a][b] - 0), math.fabs(edgeOrientation[a][b] + math.pi), math.fabs(edgeOrientation[a][b] - math.pi))
            vertical = min(math.fabs(edgeOrientation[a][b] - math.pi / 2), math.fabs(edgeOrientation[a][b] + math.pi / 2))
            rightDiagonal = min(math.fabs(edgeOrientation[a][b] - math.pi / 4), math.fabs(edgeOrientation[a][b] + math.pi / 4))
            leftDiagonal = min(math.fabs(edgeOrientation[a][b] - 3 * math.pi / 4), math.fabs(edgeOrientation[a][b] + 3 * math.pi / 4))
            if(horizontal < vertical and horizontal < rightDiagonal and horizontal < leftDiagonal):
                if(edgeStrength[a][b] < edgeStrength[a][b-1] or edgeStrength[a][b] < edgeStrength[a][b+1]):
                    I[a-2][b-2] = 0.0
                else:
                    I[a-2][b-2] = edgeStrength[a][b]
            elif(vertical < horizontal and vertical < rightDiagonal and vertical < leftDiagonal):
                if(edgeStrength[a][b] < edgeStrength[a-1][b] or edgeStrength[a][b] < edgeStrength[a+1][b]):
                    I[a-2][b-2] = 0.0
                else:
                    I[a-2][b-2] = edgeStrength[a][b]
            elif(rightDiagonal < horizontal and rightDiagonal < vertical and rightDiagonal < leftDiagonal):
                if(edgeStrength[a][b] < edgeStrength[a-1][b+1] or edgeStrength[a][b] < edgeStrength[a+1][b-1]):
                    I[a-2][b-2] = 0.0
                else:
                    I[a-2][b-2] = edgeStrength[a][b]
            else:
                if(edgeStrength[a][b] < edgeStrength[a-1][b-1] or edgeStrength[a][b] < edgeStrength[a+1][b+1]):
                    I[a-2][b-2] = 0.0
                else:
                    I[a-2][b-2] = edgeStrength[a][b]

    print("Hysteresis Thresholding...")
    for c in range(0, I.shape[0]):
        for d in range(0, I.shape[1]):
            I[c][d] = thresholding(I, T_high, T_low, c, d)

    #for c in I:
    #    print(c)


    print("Finished.")
    pylab.imshow(I)
    pylab.show()