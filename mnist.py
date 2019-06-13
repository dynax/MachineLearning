import tensorflow as tf 
from sklearn import linear_model, metrics
import os
import time
# for debug
import utils as ut

def getData(preProcess=None, path=None):
    """
    [trainData, trainLabel, testData, testLabel] = loadData(path)
        Load the minist data
    """
    # load the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # preprocess the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

class LinearModel():
    def __init__(self):
        self._initModel()

    def _initModel(self):
        self.model = linear_model.LogisticRegression(max_iter=30)

    def train(self, trainSample, trainLabel):
        self.model.fit(trainSample, trainLabel)

    def evaluate(self, sample, label):
        pred = self.model.predict(sample)
        accuracy = metrics.accuracy_score(label, pred)
        print("Accuracy is %5.3f" % accuracy)
        return pred

    def predict(self, sample):
        return self.model.predict(sample)

class SimpleCNN():
    def __init__(self, net="v1"):
        if net=="v1":
            self._initModelV1()

    def _initModelV1(self):
        pass

def run_lr():
    start_time = time.time()
    PATH_RESULT = "result/mnist"
    print("Loading data...")
    trainData, trainLabel, testData, testLabel = getData()
    print("Training...")
    LR = LinearModel()
    LR.train(trainData.reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2]), trainLabel)
    print("Evaluating...")
    LR.evaluate(testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2]), testLabel)
    pred = LR.predict(testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2]))

    end_time = time.time()
    print("Total time: %d s" % (end_time-start_time))

def run_simple_cnn():
    start_time = time.time()
    PATH_RESULT = "result/mnist"
    print("Loading data...")
    trainData, trainLabel, testData, testLabel = getData()
    print("Training...")
    # !! traning here

    end_time = time.time()
    print("Total time: %d s" % (end_time-start_time))

if __name__ == "__main__":
    pass