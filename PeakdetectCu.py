# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:49:57 2021

@author: 92558
"""
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.fftpack import fft2, fftfreq
import numpy as np
from PIL import Image, ImageOps

img=Image.open('D:/STM/Copper.png')
imgs=ImageOps.grayscale(img)
imgarray=np.asarray(imgs)
ftimg=np.fft.fft2(imgarray)
sftimg=np.fft.fftshift(ftimg)

#368px*368px=4nm*4nm in real space
#rlx=4nm and rly=4nm
x=4
y=4

#obtain real fft
def abs2(x):
    return x.real**2+x.imag**2
absimg=abs2(sftimg)
im=np.log(absimg)

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance=10)
print(coordinates)
adjcoords=(coordinates-coordinates[0])/x
print(adjcoords)


# display results
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray, extent=(-im.shape[1]/x/2, im.shape[1]/x/2, im.shape[0]/y/2, -im.shape[0]/y/2))
ax[0].axis('on')
ax[0].set_title('Original')
ax[0].set_xlabel("distance [1/nm]")
ax[0].set_ylabel("distance [1/nm]")

ax[1].imshow(image_max, cmap=plt.cm.gray, extent=(-im.shape[1]/x/2, im.shape[1]/x/2, im.shape[0]/y/2, -im.shape[0]/y/2))
ax[1].axis('on')
ax[1].set_title('Maximum filter')
ax[1].set_xlabel("distance [1/nm]")
ax[1].set_ylabel("distance [1/nm]")

ax[2].imshow(im, cmap=plt.cm.gray, extent=(-im.shape[1]/x/2, im.shape[1]/x/2, im.shape[0]/y/2, -im.shape[0]/y/2))
ax[2].autoscale(False)
ax[2].plot(adjcoords[:, 1], adjcoords[:, 0], 'r.')
ax[2].axis('on')
ax[2].set_title('Peak local max')
ax[2].set_xlabel("distance [1/nm]")
ax[2].set_ylabel("distance [1/nm]")

fig.tight_layout()

plt.show()

peaknum=coordinates.shape[0]
k=np.zeros((peaknum,1))
lmbd=np.zeros((peaknum,1))
for i in range(peaknum):
    k[i][0]=np.sqrt((adjcoords[i][1])**2+(adjcoords[i][0])**2)
    lmbd[i][0]=2/np.sqrt(3)/k[i][0]
#print(k)
print(lmbd[1])