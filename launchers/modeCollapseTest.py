import os
import sys
import json
import tensorflow as tf
import pylab as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from infogan.misc.alt2 import runSession
from infogan.misc.datasets import DataFolder
from launchers.discriminatorTest import trainsetTransform, testsetTransform, loadDatatransform, transformEncoder, \
    transformFeature_Norm, showDimRed
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier

def experimentPlot(dataset,sess,d_feat,d_in,batch_size,outFolder):
    pointsToSample = 10000

    #First do the PCA2 of the real data
    cVectorTransform = lambda x: transformFeature_Norm(x, sess, d_feat, d_in)
    trainX, rlbs = trainsetTransform(cVectorTransform, dataset)
    pca = PCA(n_components=2)

    showDimRed(trainX, rlbs, str('PCA_Real'), pca,outFolder)

    #Generate PCA2 with saved model and C vector

    #Create generated dataset
    points=0
    pointsToSample = pointsToSample - (pointsToSample % batch_size) #Make pointsSample divisible by batch_size. To exactly fit the array
    generatedSample = np.empty((pointsToSample, 2))

    while (points != pointsToSample):
        generatedBatch = sess.run(d_in)[0:batch_size] #Input is not neaded to generate random sample (random block included inside)
        #Use C transform
        cTransformed = cVectorTransform(generatedBatch)
        #Do the same pca2
        sample = pca.transform(cTransformed)

        generatedSample[points:points+batch_size] = sample
        points += batch_size

    #plot sublot both images
    plt.plot(generatedSample)
    plt.savefig(os.path.join(outFolder,'generatedSamplePCA.png'))

def modeCollapseClasify(dataset,outGenerator,epochs,sess,batch_size,clasifier, outFolder):
    expName = "ModeCollapseExp"
    for i in range(epochs):
        # Get generated samples for half the batch
        generatedBatch = sess.run(outGenerator)[0:batch_size]

        # the other half is real data
        realBatch = dataset.next_batch(batch_size)[0]  # todo check if this is reight

        # Asign labels 0 to generated 1 to real 0. And concatenate
        batchLabels = np.array([0 for i in range(batch_size)] + [1 for i in range(batch_size)])

        # Add batches and add a little noise
        batchData = np.vstack([generatedBatch, realBatch])  # todo check if this is reight
        batchData = batchData + np.random.normal(0, 0.3, batchData.shape)

        clasifier.partial_fit(batchData, batchLabels)
    threshold = 0.9
    # Now at test set get all data with realProb > threshold
    # This are weird points that could mean a mode collapse in the generator

    predictedTest = clasifier.predict_proba(dataset.dataObj.test_data)
    predicted = clasifier.predict(dataset.dataObj.test_data)
    fullRes = classification_report(np.array([1 for i in range(predicted.shape[0])]), predicted)
    print fullRes

    weirdPoints = filter(lambda x: x > threshold, predictedTest)

    result = {}
    result['threshold'] = threshold
    result['MissPoints'] = len(weirdPoints)

    data = pd.DataFrame.from_dict(result)

    writer = pd.ExcelWriter(os.path.join(outFolder, expName + '.xlsx'))
    data.to_excel(writer, 'Sheet1')
    writer.save()

def main(configPath):

    res = {}
    with open(configPath, 'r') as f:
        res = json.load(f)

    # Define parameters
    epochsClasifier = 50
    discrInputName = res['discrInputName']
    featName = res['discrLastFeatName']
    dataFolder = res['dataFolder']
    modelPath = res['modelPath']
    imageSize = res['imageSize']
    outFolder = os.path.join("/home/user/Escritorio/reportes",res['outFolder']) #TODO  WERE TO STORE??
    batch_size = int(res['batch_size'] * 0.5)

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    # Define read dataset train and test
    dataset = DataFolder(dataFolder,batch_size,testProp=0.2, validation_proportion=0.3, out_size=imageSize)


    #Set MLP for clasification
    clf = MLPClassifier( alpha=1e-5, hidden_layer_sizes = (200, 1))

    # Load saved GAN
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath + '.meta')
        new_saver.restore(sess, modelPath)

        #Get the generator input
        outputGenerator = sess.graph.get_tensor_by_name(discrInputName)
        featuresDiscr = sess.graph.get_tensor_by_name(featName)

        #Do the clasify exp for mode Collapse
        modeCollapseClasify(dataset, outputGenerator, epochsClasifier, sess, batch_size, clf, outFolder)

        # Do the other experiment
        experimentPlot(dataset, sess, featuresDiscr, outputGenerator, batch_size, outFolder)

    tf.reset_default_graph()


    pass

if __name__ == '__main__':
    configFile = None
    if len(sys.argv) > 1:
        print "About to open ",sys.argv[1]
        configFile = sys.argv[1]
        main(configFile)
    else:
        raise(Exception("You should give a configFilePath"))


