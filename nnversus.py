# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:42:18 2017

@author: Parth
"""
#%matplotlib inline


#from matplotlib import pyplot as plt
#from matplotlib import pyplot
#from scipy.interpolate import UnivariateSpline
import Qtable as table
import numpy as np
import random

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import RMSprop
import time
if(0):
    
    model1 = Sequential()
    model1.add(Dense(164, init='lecun_uniform', input_shape=(16,)))
    model1.add(Activation('relu'))

    
    model1.add(Dense(256, init='lecun_uniform'))
    model1.add(Activation('relu'))
    
    model1.add(Dense(256, init='lecun_uniform'))
    model1.add(Activation('relu'))

    
    model1.add(Dense(16, init='lecun_uniform'))
    model1.add(Activation('linear')) 
    
    rms = RMSprop()
    model1.compile(loss='mse', optimizer=rms)
    
    model2 = Sequential()
    model2.add(Dense(164, init='lecun_uniform', input_shape=(16,)))
    model2.add(Activation('relu'))

    
    model2.add(Dense(256, init='lecun_uniform'))
    model2.add(Activation('relu'))
    
    model2.add(Dense(256, init='lecun_uniform'))
    model2.add(Activation('relu'))

    
    model2.add(Dense(16, init='lecun_uniform'))
    model2.add(Activation('linear')) 
    
    rms = RMSprop()
    model2.compile(loss='mse', optimizer=rms)
    
if(1):
    model1 = load_model('nnmodel11.h5')
    model2 = load_model('nnmodel22.h5')
    

totloss = []
totgames = []
gamesperset = 1000
gamesets = 0
gamma = 0.9          #since it may take several moves to goal, making gamma high
epsilon = 0.5
#start = time.time()
for w in range(gamesets):
    for q in range(gamesperset):
        state = table.RandomState()
        counter = 0
        loss = 0
        states = []
        actions = []
        rewards =[]
        maxQ=[]
        newQallowed = []
        firstmove = True
        while(not(table.TermStateCheck(state))):
            states.append(state)
            qval = model1.predict(state.reshape(1,16), batch_size=1)
            if (random.random() < epsilon): 
                action = table.TakeAction1(state)
            else: 
                qvalsort = -np.sort(-qval)
                for k in range(16):
                    action = table.getActionFromNumber1(np.nonzero(qval==qvalsort[0][k])[1][0])
                    if(table.ActionAllowed(state,action)):
                        break
            actions.append(action)
            new_state = np.add(state, action)
            
            if(firstmove and table.TermStateCheck(new_state)):
                maxQ.append(0)
                break
            
            if(table.TermStateCheck(new_state)):
                maxQ.append(0)
                break
            

            newQ = model2.predict(new_state.reshape(1,16), batch_size=1)
            for k in range(16):
                if table.ActionAllowed(new_state,table.getActionFromNumber2(k)):
                    newQallowed.append(newQ[0][k])
            maxQ.append(np.max(np.asanyarray(newQallowed)))
            
            newQallowed = []
            firstmove = False
            
            states.append(new_state)
            qval = model2.predict(new_state.reshape(1,16), batch_size=1)
            if (random.random() < epsilon): 
                action = table.TakeAction2(state)
            else: 
                qvalsort = -np.sort(-qval)
                for k in range(16):
                    action = table.getActionFromNumber2(np.nonzero(qval==qvalsort[0][k])[1][0])
                    if(table.ActionAllowed(new_state,action)):
                        break
            actions.append(action)
            new_state_2 = np.add(new_state, action)
            
            if(table.TermStateCheck(new_state_2)):
                maxQ.append(0)
                break
    
            
            newQ = model1.predict(new_state_2.reshape(1,16), batch_size=1)
            for k in range(16):
                if table.ActionAllowed(new_state_2,table.getActionFromNumber1(k)):
                    newQallowed.append(newQ[0][k])
            maxQ.append(np.max(np.asanyarray(newQallowed)))  

            newQallowed = []
            state = new_state_2
            
        if(len(states)==1):
            rewards.append(10)
        else:
            for c in range(len(states)):
                rewards.append(1)
            rewards[len(rewards)-1] = 10
            rewards[len(rewards)-2] = -2

            
        for j in range(len(states)):
            state = states[j]
            action = actions[j]
            reward = rewards[j]
            maxQval = maxQ[j]
 
            if j%2==0:
                y = np.zeros((1,16))
                qval =  model1.predict(state.reshape(1,16), batch_size=1)
                y[:] = qval[:]
                for i in range(16):
                    action1 = table.getActionFromNumber1(i)
                    if(not(table.ActionAllowed(state,action1))):
                            y[0][i] = -4
                if reward != -2: 
                   update = (reward + (gamma * maxQval))
                else: 
                   update = reward
                y[0][table.getActionNumber(action)] = update 
                print("Game #: %s" % (q,))
                print("Gameset #: %s" %(w+1,))
                print("m1")
                model1.fit(state.reshape(1,16), y, batch_size=1, epochs=1, verbose=1)
            
            else:
                y = np.zeros((1,16))
                qval =  model1.predict(state.reshape(1,16), batch_size=1)
                y[:] = qval[:]
                for i in range(16):
                    action2 = table.getActionFromNumber2(i)
                    if(not(table.ActionAllowed(state,action2))):
                            y[0][i] = -4
                if reward != -2: 
                   update = (reward + (gamma * maxQval))
                else: 
                   update = reward
                y[0][table.getActionNumber(action)] = update 
                print("Game #: %s" % (q,))
                print("Gameset #: %s" %(w+1,))
                print("m2")
                model2.fit(state.reshape(1,16), y, batch_size=1, epochs=1, verbose=1)
                    
        if epsilon > 0.1:        
            epsilon -= (1/gamesperset)
                
    model1.save('nnmodel11.h5')
    model2.save('nnmodel22.h5')
#print(time.time()-start)    
#    pyplot.yscale('log')
#    pyplot.xscale('log')
#plt.plot(totgames, totloss, 'o')
   
wins = 0
for i in range(1):
    state = np.zeros((4,4),dtype=np.int32)
    print(state)
    while(not(table.TermStateCheck(state))):
        action = table.getActionFromNumber1(np.argmax(model1.predict(state.reshape(1,16),batch_size=1)))
#        action = table.TakeAction1(state)
        state =  np.add(state,action)
        print(state)
        if (table.TermStateCheck(state)):
            wins+=1
            break
        state = np.add(state,table.TakeAction2(state))
        print(state)
print(wins)

        