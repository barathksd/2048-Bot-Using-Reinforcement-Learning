# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:38:57 2019

@author: Barath Kumar
2048 Main Program
"""

import tkinter.messagebox
import numpy as np
from tkinter import *
from tkinter import font
from tkinter import ttk
import random
#import time 
import logic2048
from functools import partial
import build2048

timedelay = 128
p = 0
a_mat = np.zeros((4,4))
root = Tk()
root.geometry("600x400")
helv36 = font.Font(family='Helvetica', size=12, weight='bold')
helv12 = font.Font(family='Helvetica', size=12, weight='normal')
model = None

def key_press(event):
    print(event.char,event.type,event.keysym)
    if event.keysym == 'Up':
        up()
    elif event.keysym == 'Down':
        down()
    elif event.keysym == 'Right':
        right()
    elif event.keysym == 'Left':
        left()
root.bind('<KeyPress>',key_press)

lb2 = Label(root,text='All the best! ',bg='green',fg='white',font=helv36)
lb2.pack(fill=X)
lb2 = Label(root,text='2048 ',bg='orange',fg='white', borderwidth=5, relief="groove")
lb2.pack(side=LEFT,fill=Y)
lb2 = Label(root,text='2048',bg='orange',fg='white',borderwidth=5, relief="groove")
lb2.pack(side=RIGHT,fill=Y)

topfr = Frame(root,width=500,height=500)
topfr.pack()
canvas = Canvas(topfr)
canvas.pack(fill=BOTH, expand=1)

btmfr = Frame(root,width=500,height=100)
btmfr.pack(side=BOTTOM)



blist=[]
boardlist=[[0 for i in range(4)] for j in range(4)]

num = [2,4,8,16,32,64,128,256,512,1024,2048,4096]
bgcol = ['old lace','bisque','orange','tomato','orange red','red2','yellow2','gold2','goldenrod2','tan1','chocolate2','violetred2']
fgcol = ['black','black','white','white','white','white','white','white','white','white','white','white']

def pick(l):
    
    pos = random.choice(l)
    if np.random.rand() >= 0.90: return 4,pos
    else: return 2,pos

def start():
    global bt1
    global btmfr
    global a_mat
    global boardlist
    global status
    global model
    l = [i for i in range(16)]
    
   
    val,pos = pick(l)
    #val = int(a_mat[int(pos/4)][pos%4])
        
    boardlist[int(pos/4)][pos%4] = Button(topfr,text=(val),bg=bgcol[num.index(val)],fg=fgcol[num.index(val)],font=helv36)
    coor = [60+50*(pos%4),60 + 50*(int(pos/4))]
    boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])
    boardlist[int(pos/4)][pos%4].config(width=4, height=2)
    
    a_mat[int(pos/4)][pos%4] = val
    
    btup = Button(btmfr,text='up',fg='white',font=helv36,command=up)
    btdown = Button(btmfr,text='down',fg='white',font=helv36,command=down)
    btrt = Button(btmfr,text='right',fg='white',font=helv36,command=right)
    btlt = Button(btmfr,text='left',fg='white',font=helv36,command=left)
    bt2 = Button(btmfr,text='restart',bg='black',fg='white',font=helv36,command=close)
    blist.append(btup)
    blist.append(btdown)
    blist.append(btrt)
    blist.append(btlt)
    model = build2048.load_model()

    status = Label(root, text = 'Quitting...Thank You',bd=1,relief=SUNKEN,anchor=W,font=helv12)
    
    
    btup.pack(side=LEFT,padx=5, pady=1)
    btdown.pack(side=LEFT,padx=5, pady=1)
    btlt.pack(side=LEFT,padx=5, pady=1)
    btrt.pack(side=LEFT,padx=5,pady=1)
    bt2.pack(side=RIGHT)
    
    
    blist.append(bt2)
    bt1.pack_forget()
    bt1.destroy()
    process()
    
    
    

def up():
    global a_mat
    global boardlist
    global root
    global num
    global status
    global timedelay
    
    m = -1
    p = 0
    l = logic2048.check(a_mat)
    s = l['up']
    status.destroy()
    if len(s) <= 0:
        status = Label(root, text = 'Error',bd=1,relief=SUNKEN,anchor=W,font=helv12)
        status.pack(side=BOTTOM,fill=X)
        return
    status.destroy()
    init = 0
    i = 1
    
    for j in list(s):
        i = 1
        init = 0
        m = -1
        while i<4:
            if i>0 and a_mat[i-1,j]==0:
                init = init + 1
            if a_mat[i,j] != 0 and init != 0:
                p = 1
                ia = [i for i in range(i,4)]
                ep = init
                for temp in ia:
                    pos = temp*4 + j
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4),60 + 50*(int(pos/4)) - p*ep*50]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])   
                
                
                a_mat[i-ep:i - ep + len(ia),j] = a_mat[i:4,j]
                a_mat[i - ep + len(ia):4,j] = 0
                l = len(ia)
                for lp in range(1,init+1):
                    for k in range(i-lp,i-lp+l):
                        boardlist[k][j] = boardlist[k+1][j]
                for lf in range(i - init + l,4):
                    boardlist[lf][j] = 0
                
                i = i - init - 1 
                init = 0
                
            elif i>0 and a_mat[i][j] != 0 and a_mat[i-1][j]==a_mat[i][j] and i!=m:
                ia = [i for i in range(i,4)]
                l = len(ia)
                p = 1
                ep = 1
                for temp in ia:
                    pos = temp*4 + j
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4),60 + 50*(int(pos/4)) - p*ep*50]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])
               
                a_mat[i-ep:i - ep + len(ia),j] = a_mat[i:4,j]
                a_mat[i - ep + len(ia):4,j] = 0
                a_mat[i-1,j] *= 2
                
                
                boardlist[i-1][j].destroy()
                for lp in range(1,2):
                    for k in range(i-lp,i-lp+l):
                        boardlist[k][j] = boardlist[k+1][j]
                for lf in range(i - 1 + l,4):
                    boardlist[lf][j] = 0
                
                
                boardlist[i-1][j]['text'] = str(int(a_mat[i-1,j]))
                boardlist[i-1][j]['bg']=bgcol[num.index(int(a_mat[i-1,j]))]
                boardlist[i-1][j]['fg']=fgcol[num.index(int(a_mat[i-1,j]))]
                
                m = i
                
            i += 1
    print('UP')
    root.after(timedelay,process)
    
    
    
def down():
    global a_mat
    global boardlist
    global root
    global num
    global status
    global timedelay
    
    m = -1
    status.destroy()
    l = logic2048.check(a_mat)
    s = l['down']
    if len(s) <= 0:
        status = Label(root, text = 'Error',bd=1,relief=SUNKEN,anchor=W,font=helv12)
        status.pack(side=BOTTOM,fill=X)
        return
    status.destroy()
    
    
    for j in list(s):
        i = 2
        init = 0
        m = -2
        while i>= 0:
            if i<3 and a_mat[i+1,j] == 0:
                init += 1
            if a_mat[i,j] != 0 and init != 0:
                p = 1
                ia = [i for i in range(0,i+1)]
                
                ep = init
                for temp in ia:
                    pos = temp*4 + j
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4),60 + 50*(int(pos/4)) + p*ep*50]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])   
               
                a_mat[ ep  :i + ep +1 ,j] = a_mat[0:i+1,j]
                a_mat[0: ep,j] = 0
               
                for k in range(i,-1,-1):
                    boardlist[k+ep][j] = boardlist[k][j]
                for lf in range(0, ep):
                    boardlist[lf][j] = 0
                
                i = i + ep +1
                init = 0
                
            elif i<3 and a_mat[i][j] != 0 and a_mat[i+1][j]==a_mat[i][j] and i!=m:
                init = 0
                ep = 1
                p = 1 
                a_mat[ ep  :i + ep +1 ,j] = a_mat[0:i+1,j]
                a_mat[0: ep,j] = 0
                a_mat[i+1,j] *= 2
                ia = [i for i in range(0,i+1)]
                boardlist[i+1][j].destroy()
                for temp in ia:
                    pos = temp*4 + j
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4),60 + 50*(int(pos/4)) + p*ep*50]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])
                for k in range(i,-1,-1):
                    boardlist[k+ep][j] = boardlist[k][j]
                for lf in range(0, ep):
                    boardlist[lf][j] = 0
                
                boardlist[i+1][j]['text'] = str(int(a_mat[i+1,j]))
                boardlist[i+1][j]['bg'] = bgcol[num.index(int(a_mat[i+1,j]))]
                boardlist[i+1][j]['fg'] = fgcol[num.index(int(a_mat[i+1,j]))]
                m = i
                
            i = i-1
    print('DOWN')
    root.after(timedelay,process)

def right():
    global a_mat
    global boardlist
    global root
    global num
    global status
    global timedelay
    
    status.destroy()
    m = -1
    l = logic2048.check(a_mat)
    s = l['right']
    if len(s) <= 0:
        status = Label(root, text = 'Error',bd=1,relief=SUNKEN,anchor=W,font=helv12)
        status.pack(side=BOTTOM,fill=X)
        return
    status.destroy()
    for j in list(s):
        i = 2
        init = 0
        m = -2
        while i >= 0:
            if i<3 and a_mat[j,i+1] == 0:
                init += 1
            if a_mat[j,i]!= 0 and init!=0:
                p = 1
                ia = [i for i in range(0,i+1)]
                
                ep = init
                for temp in ia:
                    pos = j*4 + temp
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4)+ p*ep*50,60 + 50*(int(pos/4)) ]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])   
                
                a_mat[ j,ep  :i + ep +1] = a_mat[j,0:i+1]
                a_mat[j,0: ep] = 0
                
                for k in range(i,-1,-1):
                    boardlist[j][k+ep] = boardlist[j][k]
                for lf in range(0, ep):
                    boardlist[j][lf] = 0
                
                i = i + ep +1
                
                init = 0
            elif i<3 and a_mat[j,i] != 0 and a_mat[j,i+1]==a_mat[j,i] and i!=m:
                init = 0
                ep = 1
                p = 1 
                a_mat[ j,ep  :i + ep +1 ] = a_mat[j,0:i+1]
                a_mat[j,0: ep] = 0
                a_mat[j,i+1] *= 2
                ia = [i for i in range(0,i+1)]
                boardlist[j][i+1].destroy()
                for temp in ia:
                    pos = 4*j + temp
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4)+ p*ep*50, 60 + 50*(int(pos/4)) ]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])
                for k in range(i,-1,-1):
                    boardlist[j][k+ep] = boardlist[j][k]
                for lf in range(0, ep):
                    boardlist[j][lf] = 0
                
                boardlist[j][i+1]['text'] = str(int(a_mat[j,i+1]))
                boardlist[j][i+1]['bg'] = bgcol[num.index(int(a_mat[j,i+1]))]
                boardlist[j][i+1]['fg'] = fgcol[num.index(int(a_mat[j,i+1]))]
                m = i
                
            i = i - 1
    print('Right')
    root.after(timedelay,process)

def left():
    global a_mat
    global boardlist 
    global root
    global num
    global status
    global timedelay
    
    m = -1
    l = logic2048.check(a_mat)
    s = l['left']
    status.destroy()
    if len(s) <= 0:
        status = Label(root, text = 'Error',bd=1,relief=SUNKEN,anchor=W,font=helv12)
        status.pack(side=BOTTOM,fill=X)
        return
    status.destroy()
    for j in list(s):
        i = 1
        init = 0
        m = -2
        while i<4:
            if i>0 and a_mat[j,i-1]==0:
                init = init + 1
                
            if a_mat[j,i] != 0 and init != 0:
                p = 1
                ia = [i for i in range(i,4)]
                
                ep = init
                for temp in ia:
                    pos = j*4 + temp
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4) - p*ep*50, 60 + 50*(int(pos/4)) ]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])   
                
                    
                a_mat[j,i-ep :4 - ep] = a_mat[j,i:4]
                a_mat[j,4 - ep:4] = 0
                l = len(ia)
                for k in range(i-ep,4-ep):
                
                    boardlist[j][k] = boardlist[j][k+ep]
                for lf in range(4-ep,4):
                    boardlist[j][lf] = 0
                
                i = i - init - 1 
                
                init = 0
                
            elif i>0 and a_mat[j,i] != 0 and a_mat[j,i-1]==a_mat[j,i] and i!=m:
                ia = [i for i in range(i,4)]
                p = 1
                ep = 1
                init = 0
                
                for temp in ia:
                    pos = j*4 + temp
                    if (boardlist[int(pos/4)][pos%4] != 0):
                        coor = [60+50*(pos%4)- p*ep*50, 60 + 50*(int(pos/4)) ]
                        boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])
                
                a_mat[j,i-ep :4 - ep] = a_mat[j,i:4]
                a_mat[j,4 - ep:4] = 0
                a_mat[j,i-1] *= 2
                
                
                boardlist[j][i-1].destroy()
                for k in range(i-ep,4-ep):
                
                    boardlist[j][k] = boardlist[j][k+ep]
                for lf in range(4-ep,4):
                    boardlist[j][lf] = 0
                    
                boardlist[j][i-1]['text'] = str(int(a_mat[j,i-1]))
                boardlist[j][i-1]['bg']=bgcol[num.index(int(a_mat[j,i-1]))]
                boardlist[j][i-1]['fg']=fgcol[num.index(int(a_mat[j,i-1]))]
                m = i
            i = i+1   
    print('Left')
    root.after(timedelay,process)
            
def process():
    global a_mat
    global topfr
    global boardlist
    global blist
    global status
    global model
    
    l = np.where(a_mat.reshape(-1)==0)[0]
    #print('pl',l)
    val, pos = pick(l)
    a_mat[int(pos/4)][pos%4] = val
    boardlist[int(pos/4)][pos%4] = Button(topfr,text=(val),bg=bgcol[num.index(val)],fg=fgcol[num.index(val)],font=helv36)
    coor = [60+50*(pos%4),60 + 50*(int(pos/4))]
    boardlist[int(pos/4)][pos%4].place(x=coor[0],y=coor[1])
    boardlist[int(pos/4)][pos%4].config(width=4, height=2)
    print(a_mat,'\n')
    tot = 0
    l = logic2048.check(a_mat)
    stlist = ['up','down','right','left']
    for st in stlist:
        s = len(l[st])
        for bt in blist:
            if bt['text'] == st:
                if s <= 0:
                    bt['bg'] = 'slate gray'
                    tot += 1
                else:
                    bt['bg'] = 'black'
    if tot!=4:
        None
        #root.after(12,process2)
        
    else:
        alpha = 0.21
        gamma = 0.9
        reward = -1
        asc = np.log(a_mat+1).reshape(1,4,4,1)
        val0 = model.predict(asc)
        model.fit(asc,(1-alpha)*val0+alpha*(reward+gamma*val0),epochs = 3)
        close()
   
def process2():
    global a_mat
    global model
    _,maxins = build2048.action(model,a_mat,0)
    print(maxins)
    if maxins == 'up':
        up()
    elif maxins == 'down':
        down()
    elif maxins == 'right':
        right()
    elif maxins == 'left':
        left()
    
        
def close():
    global a_mat
    global status
    res = tkinter.messagebox.askyesno('status','Nice! you have scored {} points\n Do you want to restart?'.format(int(np.max(a_mat))))
    if res is True:
        print('Restarting')
        global boardlist
        global blist
        global bt1
        global btmfr
        status.destroy()
        for btlist in boardlist:
            for bt in btlist:
                if bt != 0:
                    bt.place_forget()
                    bt.destroy()
        for bt in blist:
            bt.pack_forget()
            bt.destroy()
        blist=[]
        boardlist=[[0 for i in range(4)] for j in range(4)]
        bt1 = Button(btmfr,text='start',bg='black',fg='white',font=helv36,command=start)
        bt1.pack(side=LEFT)
        a_mat = np.zeros((4,4))
        bt1.invoke()
        
    elif res is False:
        global root
        status = Label(root, text = 'Quitting...Thank You',bd=1,relief=SUNKEN,anchor=W,font=helv12)
        status.pack(side=BOTTOM,fill=X)
        root.after(600,root.destroy)
            
bt1 = Button(btmfr,text='Start',bg='black',fg='white',font=helv36,command=start)
bt1.pack(side=LEFT)

space =  [60,110,160,210,260]

for x in space:
    canvas.create_line(60,x,260,x, dash=(4, 2))
    canvas.create_line(x,60,x,260, dash=(4, 2))
        

root.mainloop()


