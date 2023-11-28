import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from collections import deque

class DQNAgent:
    def __init__(self, state_shape, action_size, max_mem=10):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=max_mem)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.state_shape),
            # Conv2D(64, kernel_size=(3, 3), activation='relu'),
            # Conv2D(128, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            # Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
def train_dqn(episode):
    loss = []
    agent = DQNAgent(state_size, action_size)
    for e in range(episode):
        state = get_state()  # Define your own function to get the state
        state = np.reshape(state, [1, state_size])
        done = False
        i = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = step(action)  # Define your own function to take a step
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, episode, i, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("./save/mario-dqn.h5")