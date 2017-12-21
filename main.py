import scipy.misc as sp

from ple.games.flappybird import FlappyBird
from ple import PLE
from time import sleep
import PIL.Image
import numpy as np

import random

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
print("Allowed Actions: {}".format(p.getActionSet()))
#agent = myAgentHere(allowed_actions=p.getActionSet())

p.init()
reward = 0.0
nb_frames = 2000

def transform_image(img, background, podium, podium_index, city_index = None, city_pixels = None, get_bird_pixels=None):
    height = len(img)
    width = len(img[0])
    for i in range(0, podium_index):
        counter = 0
        while counter < width:
            if city_pixels != None:
                if i >= city_index - 5:
                    if img[i][counter] in city_pixels:
                        img[i][counter] = 0
                        counter += 1
                        continue

            if bird_pixels != None:
                if img[i][counter] in bird_pixels:
                    img[i][counter] = background - 50
                    counter += 1
                    continue

            init_counter = counter
            while counter < width and img[i][counter] == background:
                img[i][counter] = 0
                counter += 1

            if counter - init_counter < 3:
                for j in range(init_counter, counter):
                    img[i][j] = background

            counter2 = counter
            while counter2 < width and img[i][counter2] != background:
                counter2 += 1

            if counter2 - counter > 7:
                for j in range(counter, counter2):
                    img[i][j] = background
            else:
                for j in range(counter, counter2):
                    img[i][j] = 0

            counter = counter2

    for i in range(podium_index, height):
        img[i] = np.full_like(img[i], podium)

def compute_podium(img):
    bincount_vector = np.bincount(img[len(img) - 1])
    podium = np.argmax(bincount_vector)

    for j in range(len(img) - 1, 0, -1):
        if img[j][0] != podium:
            break

    for j in range(len(img) - 1, 0, -1):
        if img[j][0] < 100:
            break

    while img[j][0] < 100:
        j += 1

    return podium, j

"""
def get_bird_pixels(img, background):
    height = int(len(img) / 2)
    width = int(len(img[0]) / 2)
    pixels = set()
    for i in range(0, podium_index):
        counter = 0
        while counter < width:
            init_counter = counter
            while counter < width and img[i][counter] == background:
                counter += 1

            counter2 = counter
            while counter2 < width and img[i][counter2] != background:
                counter2 += 1

            if counter2 - counter > 7:
                for j in range(counter, counter2):
                    pixels.add(img[i][j])
            counter = counter2

    print(pixels)
    return pixels
"""

def get_city_pixels(img, background, podium_index):
    city_pixels = set()
    for i in range(podium_index, 0, -1):
        if img[i][0] == background:
            break
        city_pixels = city_pixels.union(set(img[i]))
    return city_pixels, i

podium = 0
podium_index = 0
background = 0
bird_pixels = None
ok = False
city_pixels = None
city_index = None

for i in range(nb_frames):
    if p.game_over():
        p.reset_game()
        ok = True

    img = p.getScreenGrayscale().transpose()

    if i == 1 or ok == True:
        ok = False
        bincount_vector = np.bincount(img[0])
        background = np.argmax(bincount_vector)
        podium, podium_index = compute_podium(img)
        #bird_pixels = get_bird_pixels(img, background)
        city_pixels, city_index = get_city_pixels(img, background, podium_index)
        transform_image(img, background, podium, podium_index + 1, city_index, city_pixels)


    if i % 10 == 0:
        resized_img = sp.imresize(img, (80, 80))
        sp.imsave("flappy_{}_resized.jpg".format(i), resized_img)
        handle = open("flappy_{}_rawimage.jpg".format(i), "w")
        for item in img:
            for item2 in item:
                handle.write("{} ".format(item2))
            handle.write("\n")
        handle.close()

        transform_image(img, background, podium, podium_index+1, city_index, city_pixels)

        sp.imsave("flappy_{}.jpg".format(i), p.getScreenGrayscale().transpose())
        sp.imsave("flappy_{}_formatted.jpg".format(i), img)

        handle = open("flappy_{}_formatted_rawimage.jpg".format(i), "w")
        for item in img:
            for item2 in item:
                handle.write("{} ".format(item2))
            handle.write("\n")
        handle.close()


    observation = p.getScreenRGB()
    #action = agent.pickAction(reward, observation)
    nn = random.randint(0, 1)
    #nn = int(input("Insert action 0 sau 1"))
    reward = p.act(p.getActionSet()[nn])
    print("Iteration {} - reward {}".format(i, reward))


# img = sp.imread("flappy_226.jpg")
# new_img = list()
# for item in img:
#     ls = []
#     for item2 in item:
#         if item2 != 107:
#             ls.append(107)
#         else:
#             ls.append(0)
#     new_img.append(list(ls))

# sp.imsave("test_img.jpg", new_img)
# handle = open("file226", "w")
# for item in img:
#     for item2 in item:
#         handle.write("{} ".format(item2))
#     handle.write("\n")
# handle.close()

