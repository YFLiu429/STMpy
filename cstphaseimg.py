# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:16:31 2021

@author: 92558
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from numpy import abs, pi, exp, log, max, real, angle, median
import cmocean
import sys

def Cstphaseimg (imgarray):
    #obtain array size as m * n
    m=(imgarray.shape)[0]
    n=(imgarray.shape)[1]  

    if n % 2 ==1:
        zeroclm=np.zeros([len(imgarray),1])
        nimgarray=np.column_stack((imgarray,zeroclm))
    else:
            nimgarray=imgarray    
    if m % 2 ==1:
        zerow=np.zeros([1,len(nimgarray[0])])
        mimgarray=np.vstack(nimgarray,zerow)
    else:
        mimgarray=nimgarray        #regulate array as even columns and rows by appending zero colums and rows        

    m=(mimgarray.shape)[0]
    n=(mimgarray.shape)[1]         #array size after regulation

    centerx=n/2
    centery=m/2

    y,x =np.mgrid[1:m+1,1:n+1]     #meshgrid x,y

    #FFT
    ftimg=np.fft.fft2(mimgarray-np.min(mimgarray)+1)        #perform 2dfft
    sftimg=np.fft.fftshift(ftimg)                           #perfrom fftshift
    #mask
    mask=2*np.exp(-4*np.pi*(((x-n/2-1)**2)/(0.01*(n**2))))+ 2*np.exp(-4*np.pi*((((y-m/2-1)**2)/(0.01*(m**2)))))
    mask=mask/np.max(mask)         #normalize the mask
    fftimg=sftimg*(1-mask)      #mask on the image

    fft_img=(log((abs(fftimg)/max(np.max(abs(fftimg))))**2))

    #plt.figure(1)
    #fft_img=np.flipud(fft_img)
    #plt.pcolormesh(fft_img)
    #plt.show()
    plt.pcolormesh(fft_img)
    #Choose a peak
    coord=plt.ginput(1)
    #print(coord)
    gx=round((coord[0][0]-1)-centerx)
    gy=round((coord[0][1]-1)-centery)
    I1=np.roll(fftimg, -gy, axis=0) #down by -gy
    I2=np.roll(I1, -gx, axis=1) #right by -gx
    #print(I2)
    plt.close()

    #make a mask
    mask=exp(-4*pi*(((x-centerx)**2+(y-centery)**2)/(gx**2+gy**2)))
    Hk=I2*mask

    #Inverse transform
    H1=np.roll(Hk, gy, axis=0) #down by gy
    H2=np.roll(H1, gx, axis=1) #right by gx
    H=np.fft.ifft2(np.fft.ifftshift(H2))
    

    B=2*real(H)
    A=abs(H)
    Praw=angle(H)

    plt.figure(2)
    #A=np.flipud(A)
    plt.pcolormesh(A)
    plt.show()
    coord2=plt.ginput(1)
    i=coord2[0][0]
    j=coord2[0][1]
    #print(coord2)
    plt.close()

    d=10
    #dy=gradient in rows
    #dx=gradient in columns
    dy,dx=np.gradient(Praw[(round(j)-d):(round(j)+d),(round(i)-d):(round(i)+d)])
    g=np.array([median(median(dx,axis=0)),median(median(dy,axis=1))])
    g=g/(2*pi)
    #print(g)

    Phase=Praw-2*pi*(g[0]*x+g[1]*y)
    P=Phase % (2*pi)-pi

    #figure,axis=plt.subplots(2,2)
    #imgarray=np.flipud(imgarray)
    #B=np.flipud(B)
    #P=np.flipud(P)

    #img=axis[0,0].pcolormesh(imgarray)
    #imgB=axis[0,1].pcolormesh(B)
    #imgA=axis[1,0].pcolormesh(A)
    #imgP=axis[1,1].pcolormesh(P,cmap='cmo.phase')
    #figure.colorbar(imgP, ax=axis[1,1])
    #plt.show()
    
    return P,g
    
    sys.exit()
