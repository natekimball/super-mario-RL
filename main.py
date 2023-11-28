import pygame
from classes.Dashboard import Dashboard
from classes.Level import Level
from classes.Menu import Menu
from classes.Sound import Sound
from entities.Mario import Mario

from classes.Sound import Sound
from entities.Mario import Mario
from DQNAgent import DQNAgent  # Assuming you have DQNAgent in a separate Python file
import numpy as np
import sys

windowSize = 640, 480
state_size = windowSize + (3,)  # Define the size of the state
action_size = 6  # Define the size of the action space
batch_size = 32  # Define the batch size for training the DQN agent


pygame.mixer.pre_init(44100, -16, 2, 4096)
pygame.init()
screen = pygame.display.set_mode(windowSize)
max_frame_rate = 60
dashboard = Dashboard("./img/font.png", 8, screen)
sound = Sound()
level = Level(screen, sound, dashboard)
menu = Menu(screen, dashboard, level, sound)

while not menu.start:
    menu.update()

mario = Mario(0, 0, level, screen, dashboard, sound)
clock = pygame.time.Clock()

# def get_state():
#     state = []
#     # Add Mario's position to the state
#     state.append(mario.rect.x)
#     state.append(mario.rect.y)
#     # Add the positions of enemies to the state
#     for enemy in level.enemiesGroup:
#         state.append(enemy.rect.x)
#         state.append(enemy.rect.y)
#     # Add the positions of power-ups to the state
#     for powerup in level.powerupsGroup:
#         state.append(powerup.rect.x)
#         state.append(powerup.rect.y)
#     return np.array(state)

def get_state():
    state = pygame.surfarray.array3d(pygame.display.get_surface())
    print(state.shape)
    return state

def step(action):
    if action == 0:
        mario.moveRight = True
    elif action == 1:
        mario.moveLeft = True
    elif action == 2:
        mario.jump = True
    elif action == 3:
        mario.moveRight = True
        mario.jump = True
    elif action == 4:
        mario.moveLeft = True
        mario.jump = True
    elif action == 5:
        mario.moveRight = False
        mario.moveLeft = False
        mario.jump = False
    # Update the game state
    level.drawLevel(mario.camera)
    dashboard.update()
    reward = mario.update()
    # Get the new state
    next_state = get_state()
    # Calculate the reward
    # reward = calculate_reward()  # Define your own function to calculate the reward
    # Check if the game is done
    done = mario.restart
    return next_state, reward, done

# def calculate_reward():
#     reward = 0
#     # Give a positive reward for moving right
#     if mario.moveRight:
#         reward += 1
#     # Give a negative reward for moving left
#     if mario.moveLeft:
#         reward -= 1
#     # Give a positive reward for collecting a power-up
#     for powerup in level.powerupsGroup:
#         if pygame.sprite.collide_rect(mario, powerup):
#             reward += 10
#             powerup.kill()
#     # Give a negative reward for colliding with an enemy
#     for enemy in level.enemiesGroup:
#         if pygame.sprite.collide_rect(mario, enemy):
#             reward -= 10
#     # Give a large positive reward for reaching the goal
#     if mario.rect.x >= level.endX:
#         reward += 100
#     # Give a large negative reward for dying
#     if mario.restart:
#         reward -= 100
#     return reward

def play():
    while not mario.restart:
        pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
        if mario.pause:
            mario.pauseObj.update()
        else:
            level.drawLevel(mario.camera)
            dashboard.update()
            mario.update()
        pygame.display.update()
        clock.tick(max_frame_rate)
    return 'restart'

def train(iters):
    agent = DQNAgent(state_size, action_size)  # Initialize the DQN agent

    history = np.zeros(iters)
    for i in range(iters):
        while not mario.restart:
            pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
            if mario.pause:
                mario.pauseObj.update()
            else:
                state = get_state()  # Get the current state of the game
                action = agent.act(state)  # Let the agent choose an action
                next_state, reward, done = step(action)  # Take the action and get the new state, reward and done
                agent.remember(state, action, reward, next_state, done)  # Let the agent remember the state-action-reward sequence
                if done:
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)  # Train the agent
                # level.drawLevel(mario.camera)
                # dashboard.update()
                # mario.update()
            pygame.display.update()
            clock.tick(max_frame_rate)
        history[i] = mario.dashboard.score
    return history

if __name__ == "__main__":
    if len(sys.argv) > 1:
        history = train(int(sys.argv[1]))
    else:
        exitmessage = 'restart'
        while exitmessage == 'restart':
            exitmessage = play()