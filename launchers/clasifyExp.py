import os
import sys
import json
import tensorflow as tf

import matplotlib

from infogan.misc.cnnBlocks import getPredandLabels

matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from infogan.misc.alt2 import runSession
from infogan.misc.datasets import DataFolder
from launchers.discriminatorTest import trainsetTransform, testsetTransform, loadDatatransform, transformEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def trainCNNsupervised(dataFolder,batch_size,trainSplit = 0.7):
    # do CNN clasify with images
     # we do a 0.3 tesSplit and 0.01 val split to get 0.7 train set and 0.3 testSet
    l, y1, y2, seed, runTime, trainLoss, valLoss, valAc = runSession(dataFolder, (1-trainSplit),0.01,batch_size,'temp',1e-4,'temp',False,epochs=10)
    fullRes = classification_report(y1, y2)
    acc = accuracy_score(y1, y2)

    return acc, fullRes

def classifyWithDiscriminator(sess,datafolder,batch_size,imageSize):
    times = 4
    #Load discriminator
    model_input = sess.graph.get_tensor_by_name("Placeholder:0")
    classOutput = sess.graph.get_tensor_by_name("Softmax:0")

    # Do clasification in test and get acccuracy repeat a number of times to avoid to choose same points as train
    accList = []
    for i in range(times):
        dataset = DataFolder(datafolder, batch_size, testProp=0.3, validation_proportion=0.5, out_size=imageSize)

        #Get the prediction in test set
        #CHECK QUE SE MANTENGA EL SPLIT
        ypred, ytrue = getPredandLabels(dataset.dataObj, sess, classOutput, model_input)

        fullRes = classification_report(ytrue, ypred)
        accList.append(accuracy_score(ytrue, ypred))
    print "Accuracy mean ",np.mean(np.array(accList)),"Accuracy std ",np.std(np.array(accList))
    return np.mean(np.array(accList)), fullRes


def flatImage(x):
    return x.reshape(x.shape[0],-1)


def main(configPath):

    expName = "clasifyExp"

    res = {}
    with open(configPath, 'r') as f:
        res = json.load(f)

    dataFolder = res['dataFolder']
    modelPath = res['modelPath']
    batch_size = res['batch_size']
    imageSize = res['imageSize']
    outFolder = os.path.join("/home/user/Escritorio/reportes",res['outFolder'])
    batch_size = res['batch_size']

    if res.has_key('semiSup'):
        semiSup = res['semiSup']
    else:
        semiSup = False
    if res.has_key('trainSplit'):
        trainSplit = res['trainSplit']
    else:
        trainSplit = 0.7

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    # Define dataset
    #We define here TRAIN 50% AND VAL 50%
    dataset = DataFolder(dataFolder,batch_size,testProp=0.01, validation_proportion=0.5, out_size=imageSize)

    # Load saved GAN
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath + '.meta')
        new_saver.restore(sess, modelPath)
        # Define dtransform to use (raw,cond)
        transformList = loadDatatransform(res, sess,addEnc=True)
        transformList.append(('rawImageFlat',flatImage))

        result = {}

        for name,datatransform in transformList:
            #Transform dataseet
            trainX, realLabels = trainsetTransform(datatransform, dataset)
            tX,ty = testsetTransform(datatransform, dataset) #Here we ask for validation set in reality

            realLabels = np.where(realLabels)[1]
            ty = np.where(ty)[1]

            # do LSVM clasify
            clf = SGDClassifier()
            clf.fit(trainX, realLabels)
            pred = clf.predict(tX)

            fullRes = classification_report(ty, pred)
            acc =  accuracy_score(ty, pred)
            result[name] = {"acc" : acc, "full" : fullRes}

        # If we train semi supervised test the discriminator %class
        if semiSup:
            acc, fullRes = classifyWithDiscriminator(sess,dataFolder,batch_size,imageSize)
            result['Discriminator semi sup'] = {"acc": acc, "full": fullRes}
    tf.reset_default_graph()



    # Do supervised training with same number of points as GAN
    if trainSplit != 0.7:
        acc, fullRes = trainCNNsupervised(dataFolder, batch_size, trainSplit=trainSplit)
        result['CNN-alt2 lessPoints'] = {"acc": acc, "full": fullRes}

    # Do supervised training with 70% train split
    acc,fullRes = trainCNNsupervised(dataFolder,batch_size,trainSplit=0.7)
    result['CNN-alt2'] = {"acc": acc, "full": fullRes}

    data=pd.DataFrame.from_dict(result)

    writer = pd.ExcelWriter(os.path.join(outFolder,expName+'.xlsx'))
    data.to_excel(writer, 'Sheet1')
    writer.save()



    pass

if __name__ == '__main__':
    configFile = None
    if len(sys.argv) > 1:
        print "About to open ",sys.argv[1]
        configFile = sys.argv[1]
        main(configFile)
    else:
        raise(Exception("You should give a configFilePath"))


