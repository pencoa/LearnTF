import tensorflow as tf

#initialize variables/model parameters

# define the training loop operations
def inference(X):
# compute inference model over data X and return the result

def loss(X, Y):
# compute loss over training data X and expected outputs Y

def inputs():
# read/generate input training data X and expected outputs Y

def train(total_loss):
 # train / adjust model parameters according to computed total loss

def evaluate(sess, X, Y):
 # evaluate the resulting trained model
