import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import json

class Gravity:
    def __init__(self, size):                           # initialisiation function
        self.size = size
        block_y = 0                                     # 0 because blocks fall from the "sky"
        block_x = random.randint(0, self.size - 1)      # random x
        basket_x = random.randint(1, self.size - 2)
        self.state = [block_y, block_x, basket_x]

    def observe(self):
        canvas = [0] * self.size**2                     # make a 10 by 10 grid
        canvas[self.state[0] * self.size + self.state[1]] = 1
        canvas[(self.size - 1) * self.size + self.state[2] - 1] = 1
        canvas[(self.size - 1) * self.size + self.state[2] + 0] = 1
        canvas[(self.size - 1) * self.size + self.state[2] + 1] = 1
        return np.array(canvas).reshape(1, -1)

    def act(self, action):                              # action
        block_y, block_x, basket = self.state

        # action takes a value of 0, 1 or 2, based on wether we move left, stay or right
        basket_x += (int(action) - 1)

        # Make sure we don't got offscreen (move right when all the way on the rigth)
        basket_x = max(1, basket_x)
        basket_x = min(self.size - 2, basket_x)
        block_y += 1
        self.state = [block_y, block_y, basket_x]

        # rewarding
        reward = 0
        if block_y == self.size -1:
            if abs(block_x - basket_x) <= 1:
                reward = 1 # we catch the block
            else:
                reward = -1 # we missed the block

        game_over = block_y == self.size -1
        return self.observe(), reward, game_over

    def reset(self):
        self.__init__(self.size)

if __name__ == '__main__':
    # we are defining some importants
    GRID_DIM = 10

    # exploration constant
    EPSILON = 0.1

    # learning constant
    LEARNING_RATE = 0.2;

    # sum ()(actual - expected)**2)
    LOSS_FUNCTION = "mse"   # how far are we off
    HIDDEN_LAYER1_SIZE = 100
    HIDDEN_LAYER1_ACTIVATION = "relu"

    HIDDEN_LAYER2_SIZE = 100
    HIDDEN_LAYER2_ACTIVATION = "relu"

    BATCH_SIZE = 50         # 50 training examples at a time
    EPOCHS = 1000           # train our neural network over 1000 iterations

    # setup model
    # we imported this v
    model = Sequential      # feed forward network

    # our first layer
    model.add(
        Dense(
            HIDDEN_LAYER1_SIZE,
            input_shape = (GRID_DIM**2,),
            activation = HIDDEN_LAYER1_ACTIVATION
        )
    )

    # our second layer
    model.add(
        Dense(
            HIDDEN_LAYER2_SIZE,
            activation = HIDDEN_LAYER2_ACTIVATION
        )
    )

    # our output layer
    model.add(Dense(3))
    model.compile(sgd(lr = LEARNING_RATE), LOSS_FUNCTION)

    # setup environment
    env = Gravity(GRID_DIM)

    # run model
    win_cnt = 0
    for epoch in range(EPOCHS):
        env.reset()
        game_over = False
        input_t = env.observe()
        while not gam_over:
            input_tm1 = input_t

            if random.random() <= EPSILON:
                # take a random action
                action = random.randint(0,2)
            else:
                # take what our neural network tells us is best
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            input_t, reward, game_over = env.act(action)
            if reward == 1:
                # if rewarded, caught the block and won
                win_cnt += 1
        #output our win count every epoch
        print("Epoch: {:06d} / {06d} | Win count {}".format(epoch, EPOCHS, win_cnt))

    # save model weights "knowledge of the newtwork" the network accumulates
    model.save_weights("model.h5", overwrite = True)

    # here we store the newtowrk's structure
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
