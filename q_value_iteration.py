import numpy
from matplotlib import pyplot

num_rows = 3
num_cols = 4

actions = {
            'sever' : (-1, 0),
            'jug' : (1, 0),
            'zapad' : (0, -1),
            'istok' : (0, 1)
        }
        
left_error_action = {
            'sever' : 'zapad',
            'jug' : 'istok',
            'zapad' : 'jug',
            'istok' : 'sever'
        }
        
right_error_action = {
            'sever' : 'istok',
            'jug' : 'zapad',
            'zapad' : 'sever',
            'istok' : 'jug'
        }
        
action_index = {
            0 : 'sever',
            1 : 'jug',
            2 : 'zapad',
            3 : 'istok'
        }

rewards = numpy.array([[-0.04, -0.04, -0.04, 1],
          [-0.04, -0.04, -0.04, -1],
          [-0.04, -0.04, -0.04, -0.04]])

action_probabilities = [0.8, 0.1, 0.1]
l = 0
k = 0

def show_grid(Q):
    global l
    l+=1
    _, ax = pyplot.subplots()
    ax.matshow(Q)

    for i in range(num_cols):
        for j in range(num_rows):
            c = round(Q[j,i], 3)
            ax.text(i, j, str(c), va='center', ha='center')

    pyplot.savefig('Iteracija '+str(l)+'.png')
    pyplot.close()

def show_policy(policy):
    global k
    k+=1
    _, ax = pyplot.subplots()
    ax.matshow(policy)

    for i in range(num_cols):
        for j in range(num_rows):
            c = policy[j,i]
            ax.text(i, j, str(c), va='center', ha='center')

    pyplot.savefig('Polisa '+str(k)+'.png')
    pyplot.close()

def valid_state(row, column):
    
    if row < 0 or column < 0 or row > 2 or column > 3:
        return False
    elif row == 1 and column == 1:
        return False
    else: return True

def move(row, column, action):

        i, j = action

        if(valid_state(row + i, column + j)):
            return (row + i, column + j)
        else: return (row, column)

def q_value_iteration(num_iterations, gamma):

    Q = numpy.zeros((num_rows, num_cols))
    policy = numpy.zeros((num_rows, num_cols))

    for i in range(0, num_iterations):

        Q_temp = numpy.zeros((num_rows, num_cols))

        change = False
        
        for row in range(0, num_rows):
            for col in range(0, num_cols):

                if(col == 3 and row == 0):
                    Q_temp[row, col] = rewards[row, col]
                    continue
                elif(col == 3 and row == 1):
                    Q_temp[row, col] = rewards[row, col]
                    continue
                elif(col == 1 and row == 1):
                    continue

                action_values = []

                for action in actions.items():
                    correct_next_state = move(row, col, actions[action[0]])
                    correct_next_row, correct_next_col = correct_next_state
                    value_1 = action_probabilities[0] * Q[correct_next_row, correct_next_col]
                    
                    left_error_state = move(row, col, actions[left_error_action[action[0]]])
                    left_error_row, left_error_col = left_error_state
                    value_2 = action_probabilities[1] * Q[left_error_row, left_error_col]
                    
                    right_error_state = move(row, col, actions[right_error_action[action[0]]])
                    right_error_row, right_error_col = right_error_state
                    value_3 = action_probabilities[2] * Q[right_error_row, right_error_col]

                    action_values.append(value_1 + value_2 + value_3)

                Q_temp[row, col] = rewards[row, col] + gamma * numpy.max(action_values)
                policy[row, col] = numpy.argmax(action_values)      
                if( abs(Q[row, col] - Q_temp[row, col]) > 0.000001):
                    change = True
        
        Q = Q_temp
        policy_nice = numpy.vectorize(action_index.get)(policy)
        policy_nice[0, 3] = '/'
        policy_nice[1, 3] = '/'
        policy_nice[1, 1] = '/'
        if not change:
            print("Konvergencija ostvarena nakon "+str(i)+" iteracija")
            return Q, policy

Q, policy = q_value_iteration(100, 0.9)
print(Q)