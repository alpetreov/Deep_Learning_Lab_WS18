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

def id_to_action(id):
    if id == LEFT:
        return np.array([-1.0, 0.0, 0.0])
    elif id == RIGHT:
        return np.array([1.0, 0.0, 0.0])
    elif id == ACCELERATE:
        return np.array([0.0, 1.0, 0.0])
    elif id == BRAKE:
        return np.array([0.0, 0.0, np.float32(0.2)])
    else:
        return np.array([0.0, 0.0, 0.0])

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

def preprocessing_y(y_batch):
    # create two empty lists in order to store the discretized actions
    y_new = []
    # use the action_to_id function in loop to discretize the labels
    for i in range(y_batch.shape[0]):
        y_new.append(action_to_id(y_batch[i]))
    # cast the lists to np arrays
    y_new = np.array(y_new)
    # one hot encode the labels for easier classification
    return one_hot(y_new)

def preprocessing_x(X_batch):
    return rgb2gray(X_batch)
        
class Uniform_Sampling:
    
    def __init__(self, y_train_preprocessed):
        unique, index, counts = np.unique(y_train_preprocessed, axis=0, return_index=True, return_counts=True) 
        self.indx_four = []
        self.indx_three = []
        self.indx_two = []
        self.indx_one = []
        self.indx_zero = []
        for i in range(y_train_preprocessed.shape[0]):
            if all(y_train_preprocessed[i] == unique[0]):
                self.indx_four.append(i)
            elif all(y_train_preprocessed[i] == unique[1]):
                self.indx_three.append(i)
            elif all(y_train_preprocessed[i] == unique[2]):
                self.indx_two.append(i)
            elif all(y_train_preprocessed[i] == unique[3]):
                self.indx_one.append(i)
            elif all(y_train_preprocessed[i] == unique[4]):
                self.indx_zero.append(i)
                
    
    def produce_random_batch(self, X_train, y_train_p, batch_size):
        indx_batch = np.random.randint(0,4+1,batch_size)
        y_indx = []
        for i in range(indx_batch.shape[0]):
            if indx_batch[i] == 0:
                new_indx = np.random.randint(0,len(self.indx_zero))
                y_indx.append(self.indx_zero[new_indx])
            elif indx_batch[i] == 1:
                new_indx = np.random.randint(0,len(self.indx_one))
                y_indx.append(self.indx_one[new_indx])
            elif indx_batch[i] == 2:
                new_indx = np.random.randint(0,len(self.indx_two))
                y_indx.append(self.indx_two[new_indx])
            elif indx_batch[i] == 3:
                new_indx = np.random.randint(0,len(self.indx_three))
                y_indx.append(self.indx_three[new_indx])
            elif indx_batch[i] == 4:
                new_indx = np.random.randint(0,len(self.indx_four))
                y_indx.append(self.indx_four[new_indx])
        X_return = preprocessing_x(X_train[y_indx])
        return X_return, y_train_p[y_indx]