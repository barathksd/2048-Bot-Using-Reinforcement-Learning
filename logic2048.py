# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:52:51 2019

@author: Barath Kumar
"""

import numpy as np
import random
#a = np.array([[2, 4, 64, 128],
#              [8, 4, 0, 0],
#              [4, 2, 2, 0],
#              [2, 2, 2, 0]])

def check(a):
    #checks the available action for a given state
    l = {'up':set(),'down':set(),'right':set(),'left':set()}
    for j in [0,1,2,3]:
        i=3
        while i>0:
            
            if a[i][j]!=0 and (a[i][j]==a[i-1][j] or a[i-1][j]==0):
                l['up'].add(j)
            if a[j][i]!=0 and (a[j][i]==a[j][i-1] or a[j][i-1]==0):
                l['left'].add(j)
            if a[3-i][j]!=0 and (a[3-i][j]==a[4-i][j] or a[4-i][j]==0):
                l['down'].add(j)
            if a[j][3-i]!=0 and (a[j][3-i]==a[j][4-i] or a[j][4-i]==0):
                l['right'].add(j)
            i = i - 1
    return l

def do(a,ins):
    #does the given action, returns the next state and reward
    a = a.copy()
    l = check(a)
    s = l[ins]
    rew = 0
    ac = a.copy()
    
    if len(s)==0:
        return 'error',0
    else:
        for j in s:
            i = 0
            while i<3:
                if ins == 'up':
                    if i<3:
                        k = a[:,j]
                        k = k[k>0]
                        a[:,j] = np.pad(k,(0,4-k.size),'constant')
                    if a[i][j]!=0 and a[i][j]==a[i+1][j]:
                        a[i][j] *= 2
                        rew += a[i][j]
                        a[i+1][j] = 0
                    if a[i][j]==0:
                        a[i][j] = a[i+1][j]
                        a[i+1][j] = 0
                elif ins == 'down':
                  if i<3:
                      k = a[:,j]
                      k = k[k>0]
                      a[:,j] = np.pad(k,(4-k.size,0),'constant')
                  if a[3-i][j]!=0 and a[3-i][j]==a[2-i][j]:
                      a[3-i][j] *= 2
                      rew += a[3-i][j]
                      a[2-i][j] = 0
                  if a[3-i][j]==0:
                      a[3-i][j] = a[2-i][j]
                      a[2-i][j] = 0
                elif ins == 'left':
                    if i<3:
                        k = a[j,:]
                        k = k[k>0]
                        a[j,:] = np.pad(k,(0,4-k.size),'constant')
                    if a[j][i]!=0 and a[j][i]==a[j][i+1]:
                        a[j][i] *= 2
                        rew += a[j][i]
                        a[j][i+1] = 0
                    if a[j][i]==0:
                        a[j][i] = a[j][i+1]
                        a[j][i+1] = 0
                elif ins == 'right':
                    if i<3:
                        k = a[j,:]
                        k = k[k>0]
                        a[j,:] = np.pad(k,(4-k.size,0),'constant')
                    if a[j][3-i]!=0 and a[j][3-i]==a[j][2-i]:
                        a[j][3-i] *= 2
                        rew += a[j][3-i]
                        a[j][2-i] = 0
                    if a[j][3-i]==0:
                        a[j][3-i] = a[j][2-i]
                        a[j][2-i] = 0
                i += 1

    return a,cmax(a,rew) 

def process2(a):  
    # adds a new number in random position
    val =2
    l = np.where(a.reshape(-1)==0)[0]
    pos = random.choice(l)
    k = np.random.rand() 
    if k >= 0.90: 
        val = 4
    a[int(pos/4)][pos%4] = val
    return a

def cmax(a,rew):
    #computes reward
    cornermax = 0
    for i in [0,3]:
        for j in [0,3]:
            if a[i,j] >= cornermax:
                cornermax = a[i,j]
                
    globalmax = a.max()
    coor1 = np.where(a==globalmax)
    b = np.sort(a.reshape(-1))
    g2max = b[b.size-2]
    dist = 1
    if g2max == globalmax:
        dist = np.sqrt((coor1[0][0] - coor1[0][coor1[0].size-1])**2 + (coor1[1][0] - coor1[1][coor1[0].size-1])**2)

    else:
        coor2 = np.where(a==g2max)
        dm = 8
        for i in range(len(coor2[0])):
            
            dist = np.sqrt((coor1[0][0] - coor2[0][i])**2 + (coor1[1][0] - coor2[1][i])**2)
            if dm>dist:
                dm = dist
        dist = dm
        
    return (cornermax -globalmax/2)/(globalmax+1) + np.log(1.5*g2max+1)*(g2max)/(globalmax)*(-1*(dist-1)) + np.where(a == 0)[0].size*(cornermax-globalmax/3)/4096 + rew*(cornermax-globalmax/4)/(globalmax*globalmax)*4/3





