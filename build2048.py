# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:36:50 2019

@author: Barath Kumar
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
from logic2048 import *
import random

lmda = 0.9
gma = 0.90
al = 0.21

#a = np.zeros((4,4))
#insmat = ['up','down','right','left']

def save_model(model):    
    json_string = model.to_json()
    open('model.json', 'w').write(json_string)
    model.save_weights('weights.h5', overwrite=True)

def load_model():
    model = model_from_json(open('model.json').read())
    model.load_weights('weights.h5')
    model.compile(optimizer='adam', loss = 'mean_squared_error')
    return model

def nnBlock(input_size,f):
    
    A_input = keras.Input(input_size)
    A_flat = keras.layers.Flatten()(A_input)
    A = keras.layers.Dense(128,activation='linear')(A_flat)
    A = keras.layers.LeakyReLU(0.001)(A)
    A = keras.layers.Dropout(rate = 0.125)(A)
    A = keras.layers.Dense(32)(A) 
    A = keras.layers.LeakyReLU(0.001)(A)
    A = keras.layers.Dense(8)(A) 
    A = keras.layers.LeakyReLU(0.001)(A)
    A = keras.layers.Dense(1)(A)
    
    model = keras.Model(inputs = A_input,outputs = A)
    return model
#uncomment these for the first time 

#model = nnBlock((4,4,1),4)
#model.summary()
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])

#tb = TensorBoard(log_dir = 'C:/Users/Lenovo/Desktop/data')
 
def do_greedy(ast,model):
    global al
    global gma
    a_scaled = np.log(ast+1)
    pred1 = model.predict(a_scaled.reshape(1,4,4,1))[0] 
    lb = check(ast)
    lb_ins = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,lb.items())))
    maxval = -128
    maxr = -8
    if len(lb_ins) == 0:
        return pred1
    for ins in lb_ins:
        a_next,rew = do(ast.copy(),ins)  #perform the action, get the next state and reward
        a_next = process2(a_next)
        a_scaled = np.log(a_next+1)
        pred2 = model.predict(a_scaled.reshape(1,4,4,1))[0]
        if (pred2 > maxval and (rew>0 or (maxr<0 and rew<0))) or (maxval<-8) or (maxr<0 and rew>0):
            maxval = pred2
            maxr = np.sign(rew)*np.log10(abs(rew)+1) #scale the reward 
    return (1-al)*pred1 + al*(maxr+gma*maxval) #al = alpha, gma = gamma


def action(model,a,ep):
    
    global al
    global gma
    
    l = check(a)
    l_ins = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,l.items())))
    if len(l_ins) == 0:
        return -1,{}  #flag = -1
    rand = np.random.rand()
    pred0 = model.predict(np.log(a+1).reshape(1,4,4,1))[0]
    cdict = {}
    cdict['val'] = -8
    cdict['rew'] = -8
    
    if rand >= ep:
        for ins in l_ins:
            a_next,rew = do(a.copy(),ins)
            a_next = process2(a_next)
            
            a_scaled = np.log(a_next+1)
            pred1 = model.predict(a_scaled.reshape(1,4,4,1))[0]
            newval = pred1 + al*(-1 + gma*pred1 - pred1) # pred1 + alpha*(r+gma*vst2 - pred1)
            lb1 = check(a_next)
            lb_ins1 = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,lb1.items())))
            if len(lb_ins1) != 0:
                maxval2 = -128
                R2 = -10
                for ins1 in lb_ins1:
                    a_next2,rew2 = do(a_next.copy(),ins1)
                    a_next2 = process2(a_next2)
                    val2 = do_greedy(a_next2.copy(),model)
                    
                    if (val2 > maxval2 and rew2>0) or (maxval2<-8) or (R2<0 and rew2>0):
                        maxval2 = val2
                        R2 = rew2
                newval = pred1 + al*(np.sign(R2)*np.log10(abs(R2)+1) + gma*maxval2 - pred1)
                cdict['a'+ins+'next'] = maxval2
                cdict['a'+ins+'nextr'] = np.sign(R2)*np.log10(abs(R2)+1)
                cdict['a'+ins] = newval
                cdict['a'+ins+'r'] = np.sign(rew)*np.log10(abs(rew)+1)
               
            if (newval > cdict['val'] and (rew>0 or (cdict['rew']<0 and rew<0))) or (cdict['val']<-8) or (cdict['rew']<0 and rew>0) :
                cdict['ins'] = ins
                cdict['val'] = newval
                cdict['rew'] = np.sign(rew)*np.log10(abs(rew)+1)
        
        newval0 = (1-al)*pred0 + al*(cdict['rew']+gma*cdict['val'])
        model.fit(np.log(a+1).reshape(1,4,4,1),newval0,epochs = 2)
        return 4,cdict['ins']

    else:
        ins = random.choice(l_ins)
        a_new,rew = do(a.copy(),ins)
        a_new = process2(a_new)
        a_scaled = np.log(a_new+1)
        a_res = np.reshape(a_scaled,(1,4,4,1))
        pred = model.predict(a_res)[0]
        newval = pred
        print('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-')
        lb = check(a_new)
        lb_ins = list(map(lambda x: x[0], filter(lambda item: len(item[1])>0 ,lb.items())))
        maxval = -128
        if len(lb_ins) != 0:
            for ins1 in lb_ins:
                a_new1,rew1 = do(a_new.copy(),ins1)
                a_new1 = process2(a_new1)
                pred1 = do_greedy(a_new1,model)
            newval = pred + al*(np.sign(rew1)*np.log10(abs(rew1)+1)+gma*pred1 - pred)
        cdict['ins'] = ins
        cdict['val'] = newval
        cdict['rew'] = np.sign(rew)*np.log10(abs(rew)+1)
        newval0 = (1-al)*pred + al*(cdict['rew']+gma*cdict['val'])
        model.fit(np.log(a+1).reshape(1,4,4,1),newval0,epochs = 2)
        return 4,ins

def train(n,model,alpha,ep,gamma):
    global al
    global gma
    al = alpha
    gma = gamma
    maxlist = []
    #experience = []  # for experience replay
    ep1 = ep
    for i in range(n):
        a = np.zeros((4,4))
        a = process2(a)  #add a new number randomly
        loop = True
        ep = ep1
        while loop == True:
            print('a\n',a)
            flag,cdins  = action(model,a.copy(),ep)
            ep = ep*0.99   #epoch decay
            if flag > 0:
                a,_ = do(a.copy(),cdins)
                a = process2(a)
                
            else:
                val = model.predict(np.log(a+1).reshape(1,4,4,1))[0]
                print('a\n',a,val)
                maxlist.append((a.max(),val,a.argmax()))
                
                val2 = model.predict(np.log(a+1).reshape(1,4,4,1))
                newval2 = (1-alpha)*val2 + alpha*(-1 + gma*val2)
                model.fit(np.log(a+1).reshape(1,4,4,1),newval2,epochs = 3)
                #experience.append(([0],0,0))

#                rl = np.random.choice(len(experience),100)
#                if a.max()>= 1024:
#                    epochs = 1
#                    for index in rl:
#                        a1,newval,rew = experience[index]
#                        #print('experience', a1,newval,rew)
#                        if len(a1) > 2:
#                            a2,n2,r2 = experience[index+1]
#                            a_scaled = np.log(a1.copy()+1)
#                            nw2 = newval + alpha*(rew + gma*n2 - newval)
#                            model.fit(a_scaled.reshape(1,4,4,1),nw2,epochs = epochs)    
                loop = False
    return maxlist



def train_manual(n,model,alpha,ep,gma):
    #used to train the machine with human player
    alpha = alpha
    gma = gma
    maxlist = []
    for i in range(n):
        a = np.zeros((4,4))
        a = process2(a)
        mc = True
        while mc == True:
            print('a\n',a)
            a_copy = a.copy()
            a_scaled = np.log(a_copy+1)
            val = model.predict(a_scaled.reshape(1,4,4,1))[0]
            dt = {1:'up',2:'down',3:'left',4:'right',5:'no'}
            ins = str(dt[int(input())])
            if ins == 'no':
                break
            a_new, rew = do(a_copy.copy(),ins)
            rew = np.sign(rew)*np.log(abs(rew)+1)
            a_new = process2(a_new)
            a_scaled2 = np.log(a_new+1)
            pred = model.predict(a_scaled2.reshape(1,4,4,1))[0]
            er = alpha*(rew + gma*pred - val)
            newval = (1-alpha)*val + er
            print(newval,val,er,rew,pred)
            model.fit(a_scaled.reshape(1,4,4,1),newval,epochs = 2)
            a = a_new
            

num_of_games = 32
#m = train(num_of_games,model,0.2,0.99,0.81)

    
    
    
    