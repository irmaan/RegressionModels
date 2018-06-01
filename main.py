import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD



def leastSquares(trainX, testX, trainY, testY):
    firstTerm = inv(np.dot(trainX.T, trainX))
    secondTerm = np.dot(trainX.T, trainY.reshape(len(trainY), 1))
    B = np.dot(firstTerm ,secondTerm )
    yHat = np.dot(testX, B.reshape(len(B), 1))

    leastSquaresError.append(mean_squared_error(testY, yHat))


def ridgeReg(trainX, testX, trainY, testY, lamb):
    firstTerm = inv(np.dot(trainX.T, trainX) + (lamb * np.identity(trainX.shape[1])))
    secondTerm = np.dot(trainX.T, trainY.reshape(len(trainY), 1))
    B = np.dot(firstTerm ,secondTerm )
    yHat = np.dot(testX, B.reshape(len(B), 1))

    ridgeError.append(mean_squared_error(testY, yHat))


def lassoReg(trainX, testX, trainY, testY, lamb):
    clf = Ridge(alpha=lamb)
    clf.fit(trainX, trainY)
    yHat = clf.predict(testX)

    lassoError.append(mean_squared_error(testY, yHat))



def plsAlg(trainX, testX, trainY, testY):
    pls2 = PLSRegression(n_components=5)
    pls2.fit(trainX, trainY)
    yHat = pls2.predict(testX)

    plsError.append(mean_squared_error(testY, yHat))


def PCR(trainX, testX, trainY, testY):
    firstTerm = inv(np.dot(trainX.T, trainX))
    secondTerm = np.dot(trainX.T, trainY.reshape(len(trainY), 1))
    B = np.dot(firstTerm ,secondTerm )
    yHat = np.dot(testX, B.reshape(len(B), 1))

    pcrError.append(mean_squared_error(testY, yHat))




def printResults():
    print ('LS mean =', np.mean(leastSquaresError))
    print('LS Std Dev =', np.std(leastSquaresError))
    print ('Ridge Regression mean =', np.mean(ridgeError))
    print ('Ridge Std Dev =', np.std(ridgeError))
    print ('Lasso mean =', np.mean(lassoError))
    print ('Lasso Std Dev =', np.std(lassoError))
    print ('PLS mean =', np.mean(plsError))
    print ('PLS Std Dev =', np.std(plsError))
    print ('PCR mean =', np.mean(pcrError))
    print ('PCR Std Dev =', np.std(pcrError))


data = pd.read_csv('spamdata.csv')
X = data.iloc[: , :-1]
y = data.iloc[: , -1]
X=np.array(X)
y=np.array(y)

leastSquaresError = []
pcrError = []
plsError = []
ridgeError = []
lassoError = []

pca = PCA(n_components=11)
lamb = 5

pcaX = pca.fit_transform(X)
kFolds = KFold(len(X), n_folds=10, shuffle=True, random_state=1)

for trainIndex, testIndex in kFolds:
    trainX, testX = X[trainIndex], X[testIndex]
    trainY, testY = y[trainIndex], y[testIndex]
    pcaTrainX, pcaTestX = pcaX[trainIndex], pcaX[testIndex]
    leastSquares(trainX, testX, trainY, testY)
    ridgeReg(trainX, testX, trainY, testY, lamb)
    lassoReg(trainX, testX, trainY, testY, lamb)
    plsAlg(trainX, testX, trainY, testY)
    PCR(pcaTrainX, pcaTestX, trainY, testY)

printResults()
