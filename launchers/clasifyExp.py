import os
import sys
import json
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from infogan.misc.alt2 import runSession
from infogan.misc.datasets import DataFolder
from launchers.discriminatorTest import trainsetTransform, testsetTransform, loadDatatransform, transformEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

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
    outFolder = "/home/user/Escritorio/reportes"
    batch_size = res['batch_size']

    # Define dataset
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
            tX,ty = testsetTransform(datatransform, dataset)

            realLabels = np.where(realLabels)[1]
            ty = np.where(ty)[1]

            # do LSVM clasify
            clf = SGDClassifier()
            clf.fit(trainX, realLabels)
            pred = clf.predict(tX)

            fullRes = classification_report(ty, pred)
            acc =  accuracy_score(ty, pred)
            result[name] = {"acc" : acc, "full" : fullRes}
            
    tf.reset_default_graph()
    # do CNN clasify with images
    l, y1, y2, seed, runTime, trainLoss, valLoss, valAc = runSession(dataFolder, 0.3,0.3,batch_size,'temp',1e-4,'temp',False,epochs=10)
    fullRes = classification_report(y1, y2)
    acc = accuracy_score(y1, y2)
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


