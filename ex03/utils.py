import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4
RIGHT_AND_BRAKE = 5
RIGHT_AND_ACC = 6
LEFT_AND_BRAKE = 7
LEFT_AND_ACC = 8

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    gray =  2 * gray.astype('float32') - 1 
    return gray 


def action_to_id(a):
    """ 
    this method discretizes actions
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT
    elif all(a == [1.0, 0.0, 0.0]): 
        return RIGHT
    elif all(a == [0.0, 1.0, 0.0]): 
        return ACCELERATE
    elif all(a == [0.0, 0.0, np.float32(0.2)]): 
        return BRAKE
    elif all(a == [1.0, 0.0, np.float32(0.2)]):
        # I am oversimplifiyng my action inputs to make everything easier
        return RIGHT #return RIGHT_AND_BRAKE
    elif all(a == [1.0, 1.0, 0.0]): 
        return RIGHT #return RIGHT_AND_ACC
    elif all(a == [-1.0, 0.0, np.float32(0.2)]): 
        return LEFT #return LEFT_AND_BRAKE
    elif all(a == [-1.0, 1.0, 0.0]): 
        return LEFT #return LEFT_AND_ACC
    else:       
        return STRAIGHT                                        # STRAIGHT = 0
