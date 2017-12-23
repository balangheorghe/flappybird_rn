import tensorflow as tf
from keras.models import load_model
from keras import backend
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam


L1_constant = 0.01
L2_constant = 0.01
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.9
INPUT_SIZE = 8


def build_neural_network_model():
    model = Sequential()
    model.add(Dense(units=100, use_bias=True,
                    kernel_initializer="random_uniform", bias_initializer="zeros",
                    kernel_regularizer=regularizers.l2(L2_constant), bias_regularizer=regularizers.l1(L1_constant),
                    activity_regularizer=regularizers.l1_l2(L1_constant, L2_constant),
                    input_dim=INPUT_SIZE))
    model.add(Activation("relu"))
    model.add(Dense(units=2, use_bias=True,
                    kernel_initializer="random_uniform", bias_initializer="zeros",
                    kernel_regularizer=regularizers.l2(L2_constant), bias_regularizer=regularizers.l1(L1_constant),
                    activity_regularizer=regularizers.l1_l2(L1_constant, L2_constant)
                    ))
    model.add(Activation("relu"))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss="mse", optimizer=adam)
    return model

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

game_steps = 50000

q_dictionary = dict()
reward_dictionary = dict()

def compute_reward(state, passed=0, old_y=0):
    if state['next_pipe_dist_to_player'] < 50:
        if state['player_y'] > state['next_pipe_bottom_y'] or state['player_y'] < state['next_pipe_top_y']:
            return -1000

    if passed != 0 and state['player_y'] > old_y:
        return -1000

    if state['player_y'] <= 0:
        return -1000

    if state['player_y'] > 400:
        return -1000

    grade = 0
    # cateta_opusa = state['next_pipe_bottom_y'] - (state['next_pipe_bottom_y'] - state['next_pipe_top_y']) / 2 - state['player_y']
    # cateta_alaturata = state['next_pipe_dist_to_player'] + 12
    cateta_opusa = state['next_pipe_bottom_y'] - (state['next_pipe_bottom_y'] - state['next_pipe_top_y']) / 2 - (state['player_y'] + 12)

    if state['next_pipe_top_y']-50 > state['next_next_pipe_top_y']:
        cateta_opusa += 12
    elif state['next_pipe_bottom_y'] > state['next_next_pipe_bottom_y']+50:
        cateta_opusa -= 12
    elif state['next_next_pipe_dist_to_player'] - state['next_next_pipe_dist_to_player'] < 50:
        if state['next_pipe_top_y'] > state['next_next_pipe_top_y']:
            if state['next_pipe_top_y'] > state['next_next_pipe_top_y'] + 15:
                cateta_opusa += 12
            else:
                cateta_opusa += 6
        elif state['next_pipe_bottom_y'] > state['next_next_pipe_bottom_y']:
            if state['next_pipe_bottom_y'] > state['next_next_pipe_bottom_y'] + 15:
                cateta_opusa -= 12
            else:
                cateta_opusa -= 6

    cateta_alaturata = state['next_pipe_dist_to_player']
    arctan = np.arctan(cateta_opusa * 1.0 / cateta_alaturata)
    grade = (arctan * 180.0) / np.pi
    if grade > 0:
        grade *= -1
    return grade + 30


def generate_next_state(current_action, current_state, action, passed=0, old_y=0):
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

    reward = compute_reward(next_state, passed, old_y)

    return [1, next_state, reward]

def process_state(state):
    # return np.array([state.values()])
    return np.array([state])


def main_train(learning = True):
    final_score = 0
    previous_action = 1
    model = build_neural_network_model()
    game = FlappyBird(width=288, height=512, pipe_gap=100)
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    env.init()
    passed = 0
    old_y=0
    for i in range(game_steps):
        if i == game_steps - 1:
            print("Score: {}".format(final_score))
        if env.game_over():
            print("Final Score: {}".format(final_score))
            time.sleep(5)
            final_score = 0
            env.reset_game()

        observation = env.getGameState()
        # print(
        #     "player y position {}\n"
        #     "players velocity {}\n"
        #     "next pipe distance to player {}\n"
        #     "next pipe top y position {}\n"
        #     "next pipe bottom y position {}\n"
        #     "next next pipe distance to player {}\n"
        #     "next next pipe top y position {}\n"
        #     "next next pipe bottom y position {}\n".format(observation[0]["player_y"], observation[0]['player_vel'],
        #                                                    observation[0]["next_pipe_dist_to_player"], observation[0]['next_pipe_top_y'],
        #                                                    observation[0]["next_pipe_bottom_y"], observation[0]['next_next_pipe_dist_to_player'],
        #                                                    observation[0]["next_next_pipe_top_y"], observation[0]["next_next_pipe_bottom_y"])
        # )

        current_state = observation[0]

        if str(current_state) not in q_dictionary:
            q_dictionary[str(current_state)] = dict()
        if 0 not in q_dictionary[str(current_state)]:
            q_dictionary[str(current_state)][0] = 0
        if 1 not in q_dictionary[str(current_state)]:
            q_dictionary[str(current_state)][1] = 0

        for action in [0, 1]:
            returned_object = generate_next_state(previous_action, current_state, action, passed, old_y)
            if returned_object[0] == 0:
                raise NameError("Error. {}".format(returned_object[1]))
            else:
                next_state = returned_object[1]
                reward = returned_object[2]
                if str(next_state) not in q_dictionary:
                    q_dictionary[str(next_state)] = dict()
                if 0 not in q_dictionary[str(next_state)]:
                    q_dictionary[str(next_state)][0] = 0
                if 1 not in q_dictionary[str(next_state)]:
                    q_dictionary[str(next_state)][1] = 0

                q_dictionary[str(current_state)][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR *
                                                                        max(q_dictionary[str(next_state)][0],
                                                                            q_dictionary[str(next_state)][1]) -
                                                                        q_dictionary[str(current_state)][action])

        action_to_take = 0
        if (q_dictionary[str(current_state)][1] > q_dictionary[str(current_state)][0]):
            action_to_take = 1

        # returned_object = generate_next_state(previous_action, current_state, 0, passed, old_y)
        # if returned_object[0] == 0:
        #     raise NameError("Error. {}".format(returned_object[1]))
        # else:
        #     reward_to_take = returned_object[2]
        #     next_state = returned_object[1]
        #
        # vector = model.predict(np.matrix(list(next_state.values())))
        # target_to_learn = list()
        # target_to_learn.append(reward_to_take + DISCOUNT_FACTOR * vector[0][0])
        #
        # returned_object = generate_next_state(previous_action, current_state, 1, passed, old_y)
        # if returned_object[0] == 0:
        #     raise NameError("Error. {}".format(returned_object[1]))
        # else:
        #     reward_to_take = returned_object[2]
        #     next_state = returned_object[1]
        # vector = model.predict(np.matrix(list(next_state.values())))
        # target_to_learn.append(reward_to_take + DISCOUNT_FACTOR * vector[0][1])
        # model.fit(np.matrix(list(current_state.values())), np.matrix(target_to_learn))

        returned_object = generate_next_state(previous_action, current_state, action_to_take, passed, old_y)
        if returned_object[0] == 0:
            raise NameError("Error. {}".format(returned_object[1]))
        else:
            reward_to_take = returned_object[2]
            next_state = returned_object[1]

        target_to_learn = [0, 0]
        vector = model.predict_on_batch([np.matrix(list(next_state.values()))])
        value_to_learn = (reward_to_take + DISCOUNT_FACTOR * vector[0][action_to_take])
        if action_to_take == 0:
            target_to_learn[action_to_take] = value_to_learn
            target_to_learn[1] = q_dictionary[str(current_state)][1]
        else:
            target_to_learn[action_to_take] = value_to_learn
            target_to_learn[0] = q_dictionary[str(current_state)][0]

        model.train_on_batch([np.matrix(list(current_state.values()))], [np.matrix(target_to_learn)])

        if observation[0]['next_pipe_dist_to_player'] - 4 < 0:
            passed = 4
            old_y = observation[0]['next_pipe_top_y']

        # action = agent.pickAction(reward, observation)
        #nn = random.randint(0, 1)
        # compute_reward(observation[0])
        # nn = int(input("Insert action 0 sau 1"))
        # reward = env.act(env.getActionSet()[nn])
        env_reward = env.act(env.getActionSet()[action_to_take])
        if env_reward == 1:
            final_score += 1
        # if env_reward == 1:
        #     action_to_take = 1
        #     env.act(env.getActionSet()[action_to_take])
        #     env.act(env.getActionSet()[action_to_take])
        previous_action = action_to_take
        if passed !=0:
            passed -= 1
    model.save("model.h5", overwrite=True)


def main_test():
    final_score = 0
    previous_action = 1
    # model = build_neural_network_model()
    game = FlappyBird(width=288, height=512, pipe_gap=100)
    env = PLE(game, fps=30, display_screen=True, state_preprocessor=process_state)
    model = load_model("model.h5")
    env.init()
    passed = 0
    old_y = 0
    for i in range(game_steps):
        if i == game_steps - 1:
            print("Score: {}".format(final_score))
        if env.game_over():
            print("Final Score: {}".format(final_score))
            time.sleep(5)
            final_score = 0
            env.reset_game()

        observation = env.getGameState()

        vector = model.predict_on_batch([np.matrix(list(observation[0].values()))])
        a_star = np.argmax(vector)
        print(vector[0][0], vector[0][1], a_star)
        time.sleep(0.05)
        env_reward = env.act(env.getActionSet()[a_star])
        if env_reward == 1:
            final_score += 1

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    backend.set_session(session=session)
    # main_train()
    main_test()