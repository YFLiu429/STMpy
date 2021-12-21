# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:09:48 2021

@author: 92558
"""
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from cstphaseimg import Cstphaseimg
from numpy import pi, exp, sqrt
#import image as grascale
img=Image.open('D:/STM/testfigure1a.png')
imgs=ImageOps.grayscale(img)
imgarray=np.flip(np.asarray(imgs), axis = 0)

P1, g1 = Cstphaseimg(imgarray)
#print(P1)
print(g1)

P2, g2 = Cstphaseimg(imgarray)
print(P2)
print(g2)

figure,axis=plt.subplots(2, figsize=(5,8))
imgP1=axis[0].pcolormesh(P1, cmap='cmo.phase')
imgP2=axis[1].pcolormesh(P2, cmap='cmo.phase')
figure.colorbar(imgP1, ax=axis[0])
figure.colorbar(imgP2, ax=axis[1])

G=np.array([[g1[0],g2[0]],[g1[1],g2[1]]])
#print(G)
A=np.linalg.inv(np.transpose(G))
print(A)
dtmA=np.linalg.det(A)
print(dtmA)
Anorm=A/np.linalg.det(A)*sqrt(2)
print(Anorm)
Aabt=np.array([[1,-1],[0,1]])

Ux=-1/(2*pi)*(Aabt[0][0]*P1+Aabt[0][1]*P2)
Uy=-1/(2*pi)*(Aabt[1][0]*P1+Aabt[1][1]*P2)

#print(Ux)
print(Uy)
figure,axis=plt.subplots(2, figsize=(5,8))
imgUx=axis[0].pcolormesh(Ux, cmap='cmo.phase')
imgUy=axis[1].pcolormesh(Uy, cmap='cmo.phase')
figure.colorbar(imgUx, ax=axis[0])
figure.colorbar(imgUy, ax=axis[1])
cpxP1=exp(1j*P1)
cpxP2=exp(1j*P2)
dy1,dx1=np.gradient(cpxP1)
dy2,dx2=np.gradient(cpxP2)

dP1dx=np.imag(exp(-1j*P1)*dx1)
dP1dy=np.imag(exp(-1j*P1)*dy1)
dP2dx=np.imag(exp(-1j*P2)*dx2)
dP2dy=np.imag(exp(-1j*P2)*dy2)

E11=-1/(2*pi)*(Aabt[0][0]*dP1dx+Aabt[0][1]*dP2dx)
E12=-1/(2*pi)*(Aabt[0][0]*dP1dy+Aabt[0][1]*dP2dy)
E21=-1/(2*pi)*(Aabt[1][0]*dP1dx+Aabt[1][1]*dP2dx)
E22=-1/(2*pi)*(Aabt[1][0]*dP1dy+Aabt[1][1]*dP2dy)

strain11=E11
strain12=1/2*(E12+E21)
strain21=1/2*(E21+E12)
strain22=E22

rotation11=1/2*(E11-E11)
rotation12=1/2*(E12-E21)
rotation21=1/2*(E21-E12)
rotation22=1/2*(E22-E22)

figure,axis=plt.subplots(2,2)
imgs11=axis[0][0].pcolormesh(strain11, cmap='cmo.gray', vmin =-0.06,vmax=0.06)
imgs12=axis[0][1].pcolormesh(strain12, cmap='cmo.gray', vmin =-0.06,vmax=0.06)
imgs21=axis[1][0].pcolormesh(strain21, cmap='cmo.gray', vmin =-0.06,vmax=0.06)
imgs22=axis[1][1].pcolormesh(strain22, cmap='cmo.gray', vmin =-0.06,vmax=0.06)
figure.colorbar(imgs11, ax=axis[0][0])
figure.colorbar(imgs12, ax=axis[0][1])
figure.colorbar(imgs21, ax=axis[1][0])
figure.colorbar(imgs22, ax=axis[1][1])

figure,axis=plt.subplots(2,2)
imgr11=axis[0][0].pcolormesh(rotation11, cmap='cmo.gray')
imgr12=axis[0][1].pcolormesh(rotation12, cmap='cmo.gray')
imgr21=axis[1][0].pcolormesh(rotation21, cmap='cmo.gray')
imgr22=axis[1][1].pcolormesh(rotation22, cmap='cmo.gray')
figure.colorbar(imgr11, ax=axis[0][0])
figure.colorbar(imgr12, ax=axis[0][1])
figure.colorbar(imgr21, ax=axis[1][0])
figure.colorbar(imgr22, ax=axis[1][1])