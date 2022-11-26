import cvxpy as cp
import numpy as np
import json
import os

def get_state_index(base, query):
    loc = -1
    query = np.array(query)
    for target in base:
        loc+=1
        if(np.array_equal(target, query)):
            break
    return loc

def get_element_index(array, query):
    loc = -1
    query = np.array(query)
    for target in array:
        loc+=1
        if(query == array[loc]):
            break
    return loc

reward = -5
space = np.zeros((5, 4, 3))
# health can be 0,1,2,3,4
# arrows can be 0,1,2,3
# stamina can be 0,1,2
# action 1 NOOP
# action 2 SHOOT
# action 3 DODGE
# action 4 RECHARGE
action = {"NOOP":1, "SHOOT":2, "DODGE":3, "RECHARGE":4}
check = {1:"NOOP",2:"SHOOT",3:"DODGE",4:"RECHARGE"}
#State Space
state_space = []
# state_space.append(None)
for health in range(5):
    for arrow in range(4):
        for stamina in range(3):
            state_space.append([health,arrow,stamina])
state_space = np.array(state_space)

#initial states
alpha = np.zeros(len(state_space))
alpha[len(state_space)-1]=1

# for actions
actions = []
# actions.append(None)
for i in range(len(state_space)):
    state = []
    if(state_space[i][0]==0):
        state.append(action["NOOP"])
    if((state_space[i][1])!=0 and state_space[i][2]!=0 and state_space[i][0]!=0):
        state.append(action["SHOOT"])
    if(state_space[i][2]!=0 and state_space[i][0]!=0):
        state.append(action["DODGE"])
    if(state_space[i][2]!=2 and state_space[i][0]!=0):
        state.append(action["RECHARGE"])
    actions.append(state)
print(actions)
actions = np.array(actions)

len_actions = 0
for i in range(len(actions)):
    len_actions += len(actions[i])
# print(state_space)
# place = np.where(state_space==query)
# print(place[0][0])

# Rewards
rewards = np.zeros(len_actions)
# rewards[0] = None
rewards[12:] = -5

# print(get_state_index(state_space, [4,3,2]))

# for calculating A
A = np.zeros((len(state_space), len_actions))
transition = 0
for i in range(len(state_space)):
    for a in range(len(actions[i])):
        A[i][transition]=1
        transition+=1
print(transition)
j_list_old = 0
for i in range(len(state_space)):
    j_list = []
    p_ij = []
    act = []

    for a in actions[i]:

        if a == 1: #NOOP
            j_list.append(i)
            p_ij.append(0)
            act.append(1)

        elif a == 2: #SHOOT
            #If arrow hits
            query_success = [state_space[i][0]-1, state_space[i][1]-1, state_space[i][2]-1]
            j_list.append(get_state_index(state_space, query_success))
            p_ij.append(0.5)
            act.append(2)
            #If arrow misses
            query_failure = [state_space[i][0], state_space[i][1]-1, state_space[i][2]-1]
            j_list.append(get_state_index(state_space, query_failure))
            p_ij.append(0.5)
            act.append(2)

            
        elif a == 3: #DODGE
            #If got arrow and 50 stamina consumed: 0.8*0.8
            query_arrow_50 = [state_space[i][0], state_space[i][1]+1 if state_space[i][1] < 3 else 3, state_space[i][2]-1]
            j_list.append(get_state_index(state_space, query_arrow_50))
            p_ij.append(0.64)
            act.append(3)
            #If no arrow and 50 stamina consumed: 0.2*0.8
            query_narrow_50 = [state_space[i][0], state_space[i][1], state_space[i][2]-1]
            j_list.append(get_state_index(state_space, query_narrow_50))
            p_ij.append(0.16)
            act.append(3)
            #If got arrow and 100 stamina consumed: 0.8*0.2
            query_arrow_100 = [state_space[i][0], state_space[i][1]+1 if state_space[i][1] < 3 else 3, 0]
            j_list.append(get_state_index(state_space, query_arrow_100))
            p_ij.append(0.16)
            act.append(3)
            #If no arrow and 100 stamina consumed: 0.2*0.2
            query_narrow_100 = [state_space[i][0], state_space[i][1], 0]
            j_list.append(get_state_index(state_space, query_narrow_100))
            p_ij.append(0.04)
            act.append(3)

        elif a==4: #RECHARGE
            #If recharged
            query_success = [state_space[i][0], state_space[i][1], state_space[i][2]+1]
            j_list.append(get_state_index(state_space, query_success))
            p_ij.append(0.8)
            act.append(4)

            #The state does not change if not recharged, hence ignored.
            # #If not recharged.
            query_failure = [state_space[i][0], state_space[i][1], state_space[i][2]]
            j_list.append(get_state_index(state_space, query_failure))
            p_ij.append(0.2)
            act.append(4)
    for k in range(len(j_list)):
        #j_list[k] is the index 'j' and p_ij[k] is p^a_ij
        # print(j_list)
        row = j_list[k]
        column = j_list_old + get_element_index(actions[i], act[k])
        A[row][column] -=  p_ij[k]
    # print(len(actions[i]))
    # print(A)
    j_list_old += len(actions[i])

x = cp.Variable(shape=(len_actions),name="x")
constraints = [A@x==alpha,x>=0]
objective = cp.Maximize(rewards@x)
problem = cp.Problem(objective,constraints)

solution = problem.solve()


policy = []
transition=0
for i in range(len(state_space)):
    policy_temp = []
    policy_temp.append(state_space[i].tolist())
    action = []
    for a in range(len(actions[i])):
        A[i][transition]=1
        action.append(x.value[transition])
        transition+=1
    print(action)
    index = get_state_index(action,max(action))
    policy_temp.append(check[actions[i][index]])
    policy.append(policy_temp)
data_dump = {
    "a":A.tolist(),
    "r":rewards.tolist(),
    "alpha":alpha.tolist(),
    "x":x.value.tolist(),
    "policy":policy,
    "objective":solution
}

os.mkdir('outputs')
with open("outputs/output.json", "w") as outfile: 
    outfile.write(json.dumps(data_dump))