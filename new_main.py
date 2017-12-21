from keras.models import Sequential
from keras.layers import Dense
import time
import scipy.misc as sp
import gym
import gym_ple
import numpy as np
import random
from ple.games.flappybird import FlappyBird
from ple import PLE
#
# model = Sequential()
# model.add(Dense(units=64, activation="relu", input_dim=100))
# model.add(Dense(units=64, activation="softmax"))

game_steps = 1000

def compute_reward(state):
    if state['next_pipe_dist_to_player'] < 50:
        if state['player_y'] > state['next_pipe_bottom_y'] or state['player_y'] < state['next_pipe_top_y']:
            return -1000
    grade = 0
    cateta_opusa = state['next_pipe_bottom_y'] - (state['next_pipe_bottom_y'] - state['next_pipe_top_y']) / 2 - state['player_y']
    cateta_alaturata = state['next_pipe_dist_to_player'] + 12
    arctan = np.arctan(cateta_opusa * 1.0 / cateta_alaturata)
    grade = (arctan * 180.0) / np.pi
    print(grade)
    return grade * -1 + 30


def generate_next_state(current_action, current_state, action):
    if action not in [0, 1]:
        return [0, "Invalid action: {}".format(action)]
    if current_action not in [0, 1]:
        return [0, "Invalid current_action: {}".format(current_action)]

    next_state=current_state.copy()
    if current_state['player_vel'] == -8:
        if current_action == 0:
            if action == 0:
                next_state['player_vel'] = 0
                next_state['next_pipe_dist_to_player'] -= 4
                next_state['next_next_pipe_dist_to_player'] -= 4
            else:
                next_state['player_vel'] = -8
                next_state['player_y'] += next_state['player_vel']
                next_state['next_pipe_dist_to_player'] -= 4
                next_state['next_next_pipe_dist_to_player'] -= 4
        else:
            if action == 0:
                next_state['player_vel'] = -8
                next_state['player_y'] += next_state['player_vel']
                next_state['next_pipe_dist_to_player'] -= 4
                next_state['next_next_pipe_dist_to_player'] -= 4
            else:
                next_state['player_vel'] = -7
                next_state['player_y'] += next_state['player_vel']
                next_state['next_pipe_dist_to_player'] -= 4
                next_state['next_next_pipe_dist_to_player'] -= 4
    else:
        if action == 0:
            next_state['player_vel'] = -8
            next_state['player_y'] += next_state['player_vel']
            next_state['next_pipe_dist_to_player'] -= 4
            next_state['next_next_pipe_dist_to_player'] -= 4
        else:
            next_state['player_vel'] += 1
            next_state['player_y'] += next_state['player_vel']
            next_state['next_pipe_dist_to_player'] -= 4
            next_state['next_next_pipe_dist_to_player'] -= 4

    reward = compute_reward(next_state)

    return [1, next_state, reward]

def process_state(state):
    # return np.array([state.values()])
    return np.array([state])

if __name__ == '__main__':
    game = FlappyBird(width=288, height=512, pipe_gap=100)
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    env.init()
    for i in range(game_steps):
        if env.game_over():
            env.reset_game()

        observation = env.getGameState()
        print(
            "player y position {}\n"
            "players velocity {}\n"
            "next pipe distance to player {}\n"
            "next pipe top y position {}\n"
            "next pipe bottom y position {}\n"
            "next next pipe distance to player {}\n"
            "next next pipe top y position {}\n"
            "next next pipe bottom y position {}\n".format(observation[0]["player_y"], observation[0]['player_vel'],
                                                           observation[0]["next_pipe_dist_to_player"], observation[0]['next_pipe_top_y'],
                                                           observation[0]["next_pipe_bottom_y"], observation[0]['next_next_pipe_dist_to_player'],
                                                           observation[0]["next_next_pipe_top_y"], observation[0]["next_next_pipe_bottom_y"])
        )
        # action = agent.pickAction(reward, observation)
        #nn = random.randint(0, 1)
        compute_reward(observation[0])
        nn = int(input("Insert action 0 sau 1"))
        reward = env.act(env.getActionSet()[nn])
        print("Iteration {} - reward {}".format(i, reward))
        time.sleep(0.1)


