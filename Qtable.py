import numpy as np

def RandomState():
    while(1):
        z1 = np.random.randint(3,size = (4,4), dtype=np.int32)
        if np.count_nonzero(z1 == 1)==np.count_nonzero(z1 == 2) and not(TermStateCheck(z1)):
            break
    return z1

      
def TakeAction1(state):
    action = np.zeros([4,4],dtype=np.int32)
    while(1):
        i = np.random.randint(0,4)
        j = np.random.randint(0,4)
        if state[i][j]==0:
            action[i][j] = 1
            break
    return action

def TakeAction2(state):
    action = np.zeros([4,4],dtype=np.int32)
    while(1):
        i = np.random.randint(0,4)
        j = np.random.randint(0,4)
        if state[i][j]==0:
            action[i][j] = 2
            break
    return action
   
def win3(x,t):
    if x[0][0] == t and x[0][1] == t and x[0][2] == t:
        return True
    if x[1][0] == t and x[1][1] == t and x[1][2] == t:
        return True
    if x[2][0] == t and x[2][1] == t and x[2][2] == t:
        return True
    if x[1][0] == t and x[2][0] == t and x[0][0] == t:
        return True
    if x[1][1] == t and x[2][1] == t and x[0][1] == t:
        return True
    if x[1][2] == t and x[2][2] == t and x[0][2] == t:
        return True
    if x[0][0] == t and x[1][1] == t and x[2][2] == t:
        return True
    if x[0][2] == t and x[1][1] == t and x[2][0] == t:
        return True
    return False
    
    
    
    
def TermStateCheck(state):
    state_array = np.asanyarray(state)
    x1 = np.empty((3,3), dtype = np.int32)
    for i in range(3):
        for j in range(3):      
            x1[i][j] = state_array[i][j]

    if win3(x1,1) or win3(x1,2):
        return True
        
    for i in range(3):      
        for j in range(1,4):
            x1[i][j-1] = state_array[i][j]

    if win3(x1,1) or win3(x1,2):
        return True
        
    for i in range(1,4):
        for j in range(1,4):
            x1[i-1][j-1] = state_array[i][j]
    
    if win3(x1,1) or win3(x1,2):
         return True
         
    for i in range(1,4):  
         for j in range(3):
            x1[i-1][j] = state_array[i][j]

    if win3(x1,1) or win3(x1,2):
         return True
    
    if(np.count_nonzero(state_array)==16):
        return True
        
    return False   


def getActionFromNumber1(num):
    while(1):
        action = np.zeros((4,4),dtype=np.int32)
        i = np.random.randint(0,4)
        j = np.random.randint(0,4)
        action[i][j]=1
        row,col = action.nonzero()
        if(num==row[0]*4 + col[0]):
            return action
            
def getActionFromNumber2(num):
    while(1):
        action = np.zeros((4,4),dtype=np.int32)
        i = np.random.randint(0,4)
        j = np.random.randint(0,4)
        action[i][j]=2
        row,col = action.nonzero()
        if(num==row[0]*4 + col[0]):
            return action

def getActionNumber(action):
    row,col = action.nonzero()
    return row[0]*4 + col[0] 

def ActionAllowed(state,action):
    row,col = action.nonzero()
    if state[row[0]][col[0]] == 0:
        return True
    else:
        return False
