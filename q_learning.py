import numpy
import random
from matplotlib import pyplot

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

num_rows = 3
num_columns = 4
num_actions = 4
factor = 1

expected_Q = numpy.array([[ 0.50941556,  0.64958636,  0.79536224,  1.        ],
                        [ 0.39851114,  0.,          0.48644046, -1.        ],
                        [ 0.2964662,   0.25396025,  0.34478829,  0.12994225]])

def valid_state(row, column):

    if row < 0 or column < 0 or row > 2 or column > 3:
        return False
    elif row == 1 and column == 1:
        return False
    else: return True

l = 0
def show_grid(Q):
    global l
    global expected_Q
    Q_ = numpy.array(Q, copy=True)
    Q_[0, 3] = 1
    Q_[1, 3] = -1
    _, ax = pyplot.subplots()
    ax.matshow(Q_)

    for i in range(num_columns):
        for j in range(num_rows):
            c = numpy.round_(Q_[j,i], 3)
            b = numpy.round_(expected_Q[j,i], 3)
            ax.text(i, j, str(c), va='top', ha='center')
            ax.text(i, j, str(b), va='bottom', ha='center')

    pyplot.savefig('Epizoda'+str(l*factor)+'.png')
    l+=1
    pyplot.close()

class Simulator:
    
    def __init__(self):
        self.row = 2
        self.column = 0

    def move(self, action):

        i, j = action
        if(valid_state(self.row + i, self.column + j)):
            self.row = self.row + i
            self.column = self.column + j

    def reset_game(self):

        self.row = 2
        self.column = 0

    def resolve_action(self, action):

        temp = random.uniform(0.0, 1.0)

        if temp <= 0.8:
            self.move(actions[action_index[action]])
        elif temp <= 0.9 and temp > 0.8:
            self.move(actions[left_error_action[action_index[action]]])
        elif temp <= 1.0 and temp > 0.9: 
            self.move(actions[right_error_action[action_index[action]]])

        if (self.row == 0 or self.row == 1) and self.column == 3:
            game_ended = True
        else:
            game_ended = False

        return self.row, self.column, game_ended

class Agent:

    def __init__(self, simulator):
        self.simulator = simulator
        self.Q = numpy.zeros((num_rows, num_columns, num_actions))

    def learn_from_simulator(self, epsilon, gamma, alpha = 0.1, adaptive_alpha = False):

        for episode in range(1, 100000):

            if adaptive_alpha:
                alpha = numpy.log(episode + 1)/(episode + 1)

            game_ended = False

            Q_prime = numpy.array(self.Q, copy=True)

            while not game_ended:

                current_row = self.simulator.row
                current_column = self.simulator.column

                if random.uniform(0.0, 1.0) >= epsilon:
                    action = random.randint(0, 3)
                else: 
                    action = numpy.argmax(self.Q[self.simulator.row, self.simulator.column])

                next_row, next_column, game_ended = self.simulator.resolve_action(action)
                
                Q_sample = rewards[next_row, next_column] + gamma * numpy.max(self.Q[next_row, next_column])
                Q_temp = self.Q[current_row, current_column, action]

                self.Q[current_row, current_column, action] = Q_temp + alpha * (Q_sample - Q_temp)

                if game_ended:
                    self.simulator.reset_game()
            
            diff = abs(Q_prime - self.Q)
            if numpy.all(diff < 0.01):
                print("Ucenje je trajalo", episode, "epizoda")
                break
            
            if episode%factor == 0:
                for_display = numpy.max(self.Q, axis = 2)
                show_grid(for_display)
        

simulator = Simulator()
agent = Agent(simulator)
agent.learn_from_simulator(0.9, 0.9, alpha = 0.01, adaptive_alpha=True)

policy = numpy.argmax(agent.Q, axis = 2)
policy_nice = numpy.vectorize(action_index.get)(policy)
policy_nice[0, 3] = 'None'
policy_nice[1, 3] = 'None'
policy_nice[1, 1] = 'None'
print(policy_nice)