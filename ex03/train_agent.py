from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_batch, y_batch, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.
    
    # create two empty lists in order to store the discretized actions
    y_new = []
    # use the action_to_id function in loop to discretize the labels
    for i in range(y_batch.shape[0]):
        y_new.append(action_to_id(y_batch[i]))
    # cast the lists to np arrays
    y_new = np.array(y_new)
    # one hot encode the labels for easier classification
    y_prep = one_hot(y_new)
    
    # Change X_train and X_valid to gray scale
    X_prep = rgb2gray(X_batch)
    
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_prep, y_prep


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")
    tensorboard_eval = Evaluation(tensorboard_dir)
    
    # TODO: specify your neural network in model.py 
    agent = Model(lr, tensorboard_eval.sess)
    
    num_examples = X_train.shape[0]
	X_valid, y_valid = preprocessing(X_valid, y_valid)
    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    for epoch in range(n_minibatches):
        avg_loss = 0
        for iteration in range(num_examples // batch_size):
            #this cycle is for dividing step by step the heavy work of each neuron
            X_batch, y_batch = sample_minibatch(X_train, y_train, iteration, batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={X: X_batch, y: y_batch})
            avg_loss += c / (num_examples // batch_size)
            
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        tensorboard_eval.write_episode_data(epoch, {"loss": avg_loss, "acc_train": acc_train, "acc_valid": acc_valid})
        print("Epoch:",epoch+1, "Train accuracy:", acc_train, "valid accuracy:", acc_valid, "loss:", avg_loss) 
    
    model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    tensorboard_eval.close_session()
    print("Model saved in file: %s" % model_dir)

def sample_minibatch(X_train, y_train, iteration, batch_size):
    X_train_batch = X_train[iteration*batch_size:iteration*batch_size+batch_size]
    y_train_batch = y_train[iteration*batch_size:iteration*batch_size+batch_size]
	X_train_batch_p, y_train_batch_p = preprocessing(X_train_batch, y_train_batch)
    return X_train_batch_p, y_train_batch_p

if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=10, batch_size=50, lr=0.0001)
 
