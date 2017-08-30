import numpy, skimage, skimage.color, skimage.io, pylab, scipy.signal, scipy.ndimage.filters
import os
from PIL import Image
from skimage import feature
import math
import operator

def covariance(xGrad, yGrad, x, y):
    #return numpy.cov(xGrad[x-2:x+2][y-2:y+2], yGrad[x-2:x+2][y-2:y+2])

    fXsquared = 0.0
    fYsquared = 0.0
    fXY = 0.0
    for a in range(x-2, x+2):
        for b in range(y-2, y+2):
            fXsquared += math.pow(xGrad[a][b], 2)
            fYsquared += math.pow(yGrad[a][b], 2)
            fXY += xGrad[a][b] * yGrad[a][b]
    return [fXsquared/25.0, fXY/25.0, fXY/25.0, fYsquared/25.0]

if __name__ == '__main__':
    A = skimage.img_as_float(skimage.io.imread('flower.jpg'))
    lum = skimage.color.rgb2gray(A)

    print("Gaussian smoothing...")
    gau = numpy.array([[0.003, 0.013, 0.022, 0.013, 0.003],
                       [0.013, 0.059, 0.097, 0.059, 0.013],
                       [0.022, 0.097, 0.159, 0.097, 0.022],
                       [0.013, 0.059, 0.097, 0.059, 0.013],
                       [0.003, 0.013, 0.022, 0.013, 0.003]])

    myGaussian = scipy.signal.convolve2d(lum, gau)
    print("Computing smoothed gradients...")
    xGrad, yGrad = numpy.gradient(myGaussian)

    #C = numpy.empty((myGaussian.shape[0], myGaussian.shape[1], 4))

    Threshold = 0.0002
    L = {}

    twenty = int(myGaussian.shape[0]*0.2)+2
    forty = int(myGaussian.shape[0]*0.4)+2
    sixty = int(myGaussian.shape[0]*0.6)+2
    eighty = int(myGaussian.shape[0]*0.8)+2

    print("Finding corners...")
    for x in range(2, myGaussian.shape[0]-2):
        if(x == twenty): print("20% done")
        if(x == forty): print("40% done")
        if(x == sixty): print("60% done")
        if(x == eighty): print("80% done")
        for y in range(2, myGaussian.shape[1]-2):
            a, b, c, d = covariance(xGrad, yGrad, x, y)
            w, v = numpy.linalg.eig([[a, b], [c, d]])
            #print(sorted(w)[0])
            if(sorted(w)[0] >= Threshold):
                L[(x-2, y-2)] = w[0]
    #print(len(L))
    #sortedList = sorted(L.items(), key=operator.itemgetter(1)).reverse()

    print("Non-maximum suppression...")
    for key in L:
        tempX = key[0]
        tempY = key[1]
        if((tempX-1, tempY-1) in L and L[(tempX-1, tempY-1)] > L[key]):
            pass
        elif((tempX-1, tempY) in L and L[(tempX-1, tempY)] > L[key]):
            pass
        elif((tempX-1, tempY+1) in L and L[(tempX-1, tempY+1)] > L[key]):
            pass
        elif((tempX, tempY-1) in L and L[(tempX, tempY-1)] > L[key]):
            pass
        elif ((tempX, tempY) in L and L[(tempX, tempY)] > L[key]):
            pass
        elif ((tempX, tempY+1) in L and L[(tempX, tempY+1)] > L[key]):
            pass
        elif ((tempX+1, tempY - 1) in L and L[(tempX+1, tempY - 1)] > L[key]):
            pass
        elif ((tempX+1, tempY) in L and L[(tempX+1, tempY)] > L[key]):
            pass
        elif ((tempX + 1, tempY + 1) in L and L[(tempX + 1, tempY + 1)] > L[key]):
            pass
        else:
            pylab.plot(key[1], key[0], 'bs')

    print("Finished.")
    pylab.imshow(A)
    pylab.show()