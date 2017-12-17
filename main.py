import scipy.misc as sp

from ple.games.flappybird import FlappyBird
from ple import PLE
from time import sleep
import PIL.Image

import random

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
print("Allowed Actions: {}".format(p.getActionSet()))
#agent = myAgentHere(allowed_actions=p.getActionSet())

p.init()
reward = 0.0
nb_frames = 1000

for i in range(nb_frames):
    if p.game_over():
        p.reset_game()

    if i == 1:
        sp.imsave("flappy.jpg", p.getScreenGrayscale().transpose())
        print(p.getScreenGrayscale().transpose().shape)

    observation = p.getScreenRGB()
    sleep(0.05)
    #action = agent.pickAction(reward, observation)
    nn = int(input("Insert action 0 sau 1"))
    reward = p.act(p.getActionSet()[nn])
    print("Iteration i - reward {}".format(reward))
