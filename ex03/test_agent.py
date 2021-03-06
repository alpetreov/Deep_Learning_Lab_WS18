from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    history = 5
    history_counter = 0
    state_list = []
    while True:
    
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        state = rgb2gray(state)
        state = np.expand_dims(state, axis=0)
        # if step == 0:
            # state_list.append(state)
            # state_list.append(state)
            # state_list.append(state)
            # state_list.append(state)
            # state_list.append(state)
        # else:
            # state_list.pop()
            # state_list.insert(0,state)
        # state_array = np.array(state_list)
        # state_array = np.reshape(state_array, [-1,96,96,5])
        #print(step)
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        
        pred = agent.predict.eval(feed_dict={agent.X: state})
        print(pred)
        a = id_to_action(pred)    
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps:
            break
        history_counter += 1 
    return episode_reward


if __name__ == "__main__":

    rendering = True                      # set rendering=False if you want to evaluate faster
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = Model(0.0001,1)
    agent.load("models/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    with agent.sess:
        for i in range(n_test_episodes):        
            episode_reward = run_episode(env, agent, rendering=rendering)
            episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
