import numpy as np


import matplotlib

from traditionalClusteringTests.dataUtils import showDimRed, showResults, OneHotToInt

matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import rescale
import os
import pickle
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph

from infogan.misc.datasets import DataFolder
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy import spatial
from sklearn import metrics, cluster
import sys
import json

#TODO generalize to VAL TEST SET
#TODO This can scale to large datasets ????
#TODO FIX THIS GLOBAL PARAMETERES
#----------------------PARAMETERS ALL DEFINED IN CONFIG FILE --------------------
from infogan.misc.utilsTest import generate_new_color

if len(sys.argv) > 1:
    print os.getcwd()
    configFile = sys.argv[1]
    print "Loading config file ", configFile

    res = {}

    with open(configFile, 'r') as f:
        res = json.load(f)

    dataFolder = res['train_Folder']
    if res.has_key('exp_name'):
        exp_name = res['exp_name']
    else:
        filename=os.path.split(configFile)[-1]
        assert(len(filename.split(".")) == 2)
        exp_name = filename.split(".")[0]

    if res.has_key('modelPath'):
        modelPath = res['modelPath']
    else:
        train_dataset = res['train_dataset']
        modelPath = os.path.join("ckt",train_dataset,exp_name,"last.ckpt")


    # Define methods to use for label (BOOLEAN)
    doCluster = res['doCluster']
    doEncoderLabel = res['doEncoderLabel']
    showTNSE = res['showTNSE']

    # Define in and outputs to use (String)
    discrInputName = res['discrInputName']
    discrLastFeatName = res['discrLastFeatName']
    discrEncoderName = res['discrEncoderName']
    # Define output folder for results (STRING)

    if res.has_key('outFolder'):
        outFolder = res['outFolder']
    else:
        filename = os.path.split(configFile)[-1]
        assert (len(filename.split(".")) == 2)
        exp_name = filename.split(".")[0]
        outFolder = os.path.join("imagenesTest",exp_name)

    # This has to be the saved batch_size (INT)
    batch_size = res['batch_size']
    # The image size used in the arch (int) (square image)
    imageSize = res['imageSize']

    # Define data_transform to use "convFeat" OR "convFeatEncoder" (STRING)
    nameDataTransform = res['nameDataTransform']
    nClustersTofind = res['nClustersTofind']




#----------------------PARAMETERS --------------------

if __name__ == '__main__':
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)


def pool_features(feat, pool_type='avg'):
    if len(feat.shape) >= 3:
        if pool_type == 'avg':
            feat = feat.mean(axis=(1, 2))
        if pool_type == 'max':
            feat = feat.mean(axis=(1, 2))
    return feat.reshape((feat.shape[0], feat.shape[-1]))

def trainsetTransform(data_transform, dataset):#TODO this wont scale if dataset is big
    trainX = np.array([]).reshape(0, 0)
    realLabels = []
    # From train data create clustering
    for ii in range(dataset.batch_idx['train']):
        x, batch_labels = dataset.next_batch(dataset.batch_size)

        d_features_norm = data_transform(x)

        if trainX.shape[0] == 0:  # Is empty
            trainX = d_features_norm
        else:
            trainX = np.concatenate((trainX, d_features_norm), axis=0)
        realLabels = realLabels + list(batch_labels)
    return trainX,realLabels

def testsetTransform(data_transform, dataset):#TODO this wont scale if dataset is big ALSO THIS ISNT TEST SET !!! IS VAL
    trainX = np.array([]).reshape(0, 0)
    realLabels = []
    # From train data create clustering
    for ii in range(dataset.batch_idx['val']):
        x, batch_labels = dataset.next_batch(dataset.batch_size)

        d_features_norm = data_transform(x)

        if trainX.shape[0] == 0:  # Is empty
            trainX = d_features_norm
        else:
            trainX = np.concatenate((trainX, d_features_norm), axis=0)
        realLabels = realLabels + list(batch_labels)
    return trainX,realLabels

def transformEncoder(x,sess,d_in,d_encoder):
    encoderOut = sess.run(d_encoder, {d_in: x})
    result = normalize(encoderOut, axis=1, norm='l2')
    return result

def transformFeatureAndEncoder(x,sess,d_feat,d_encoder,d_in):
    d_features = sess.run(d_feat, {d_in : x})
    d_features = pool_features(d_features, pool_type='avg')
    encoderOut = sess.run(d_encoder, {d_in : x})
    result=normalize(np.hstack([d_features,encoderOut ]), axis=1, norm='l2')
    return result

def transformFeatureAndEncoderDontNorm(x,sess,d_feat,d_encoder,d_in):
    d_features = sess.run(d_feat, {d_in : x})
    d_features = normalize(pool_features(d_features, pool_type='avg'), axis=1, norm='l2')
    encoderOut = sess.run(d_encoder, {d_in : x})
    result=np.hstack([d_features,encoderOut ])
    return result

def transformFeature_Norm(x,sess,d_feat,d_in):
    d_features = sess.run(d_feat, {d_in : x})
    d_features = pool_features(d_features, pool_type='avg')
    return normalize(d_features, axis=1, norm='l2')



def clusterLabeling(sess,dataset,data_transform,clusterAlg,trainX):

    print "Learning the clusters."


    #Now predict validation and train data
    realLabels = []
    predited = []
    transformed = np.array([]).reshape(0, 0)

    #Predict TrainData data
    for ii in range(dataset.batch_idx['val']):
        x, batch_labels = dataset.next_batch(dataset.batch_size, split="val")

        d_features_norm = data_transform(x)

        if transformed.shape[0] == 0:  # Is empty
            transformed = d_features_norm
        else:
            transformed = np.concatenate((transformed, d_features_norm), axis=0)
        realLabels = realLabels + list(batch_labels)

    if hasattr(clusterAlg, 'predict'):
        clusterAlg.fit(trainX)
        predited = list(clusterAlg.predict(transformed))
    else:
        predited = list(clusterAlg.fit_predict(transformed))

    realLabels = OneHotToInt(realLabels)

    return transformed,predited,realLabels


def encoderLabeling(sess,dataset,d_in,data_transform,d_encoder):
    #Now predict validation and train data
    realLabels = []
    predited = []
    transformed = np.array([]).reshape(0, 0)


    #Predict TrainData data
    for ii in range(dataset.batch_idx['val']):
        x, batch_labels = dataset.next_batch(dataset.batch_size, split="val")

        d_features_norm = data_transform(x)

        if transformed.shape[0] == 0:  # Is empty
            transformed = d_features_norm
        else:
            transformed = np.concatenate((transformed, d_features_norm), axis=0)

        pred = np.argmax(sess.run(d_encoder, {d_in : x}), axis=1)

        predited = predited + list(pred)
        realLabels = realLabels + list(batch_labels)

    realLabels = OneHotToInt(realLabels)
    return transformed,predited,realLabels






def loadDatatransform(values,sess,addEnc=False):
    # Define in and outputs to use (String)
    discrInputName = values['discrInputName']
    discrLastFeatName = values['discrLastFeatName']
    discrEncoderName = values['discrEncoderName']
    nameDataTransform = values['nameDataTransform']

    validTransforms = nameDataTransform.split(',')
    transformList = []
    d_in = None
    d_feat = None
    d_encoder = None

    if not (discrInputName == 'None'):
        d_in = sess.graph.get_tensor_by_name(discrInputName)
    if not (discrLastFeatName == 'None'):
        d_feat = sess.graph.get_tensor_by_name(discrLastFeatName)
    if not (discrEncoderName == 'None') and (('cen' in validTransforms) or ('cend' in validTransforms)):
        d_encoder = sess.graph.get_tensor_by_name(discrEncoderName)

    # Check if we have the data to run test
    assert (not (d_in is None))
    assert (not (d_feat is None))
    assert (not (doEncoderLabel) or (doEncoderLabel and not (d_encoder is None)))

    # Check and define data transform c,cen,cend define valid transform of data

    for elem in validTransforms:
        if elem == "c":
            transformList.append((elem, lambda x: transformFeature_Norm(x, sess, d_feat, d_in)))
        elif elem == "cen":
            transformList.append((elem, lambda x: transformFeatureAndEncoder(x, sess, d_feat, d_encoder, d_in)))
        elif elem == 'cend':
            transformList.append((elem, lambda x: transformFeatureAndEncoderDontNorm(x, sess, d_feat, d_encoder, d_in)))
        else:
            raise Exception("ERROR DATA TRANSFORM NOT DEFINED")
    if addEnc and values['doEncoderLabel']:
        transformList.append(('encoder', lambda x: transformEncoder(x, sess, d_in, d_encoder)))

    return transformList






def main():
    #Get dataset
    dataset = DataFolder(dataFolder,batch_size,testProp=0.01, validation_proportion=0.5, out_size=imageSize)


    #Load discriminator
    print os.getcwd()

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath+'.meta')
        new_saver.restore(sess, modelPath)

        #Load datatrasnform to use
        transformList = loadDatatransform(res, sess)

        #Define grid of images for table1
        nAlgCluster = 1
        if doCluster:
            nAlgCluster += 8 #TODO DONT HARDCODE THIS
        if doEncoderLabel:
            nAlgCluster += 1
        if showTNSE:
            nAlgCluster += 1

        for indT,tupleTranform in enumerate(transformList):
            showBlokeh = True
            transformName = tupleTranform[0]
            dtransform = tupleTranform[1]
            currentCol = 0

            trainX, rlbs = trainsetTransform(dtransform, dataset)
            rlbs = OneHotToInt(rlbs)

            if showTNSE:
                print "About to show TSNE with real labels (may take a while)"
                print "Showing TSNE with real labels"
                model = TSNE(n_components=2)
                showDimRed(trainX, rlbs, transformName+str(' TSNE_Real'), model,outFolder)
                currentCol += 1

            # SHOW PCA2 of data with REAL labels
            pca = PCA(n_components=2)
            transformed = showDimRed(trainX, rlbs, transformName+str(' PCA_Real'), pca,outFolder)

            currentCol += 1

            if doCluster:

                trainX,_ = trainsetTransform(dtransform, dataset)

                #----------------Define cluster methods----------------------------------------


                connectivity = kneighbors_graph(trainX, n_neighbors=10, include_self=False)
                connectivity = 0.5 * (connectivity + connectivity.T)

                two_means = cluster.KMeans(n_clusters=nClustersTofind)
                ward = cluster.AgglomerativeClustering(n_clusters=nClustersTofind, linkage='ward',
                                                       connectivity=connectivity)
                spectral = cluster.SpectralClustering(n_clusters=nClustersTofind,
                                                      eigen_solver='arpack',
                                                      affinity="nearest_neighbors")
                dbscan = cluster.DBSCAN(eps=.2)
                affinity_propagation = cluster.AffinityPropagation(preference=-0.7)

                average_linkage = cluster.AgglomerativeClustering(
                    linkage="average", affinity="cityblock", n_clusters=nClustersTofind,
                    connectivity=connectivity)

                birch = cluster.Birch(n_clusters=nClustersTofind)
                clustering_algorithms = [
                    two_means, spectral]

                # ----------------Define cluster methods----------------------------------------
                for clusterAlg in clustering_algorithms:
                    points,predClust,realsLab = clusterLabeling(sess,dataset,dtransform,clusterAlg,trainX)
                    name = clusterAlg.__class__.__name__
                    print "Showing results for Cluster ",name

                    if showBlokeh: #This will plot once per transformation
                        showResults(dataset, points, predClust, realsLab, transformName + " " + 'Cluster ' + str(name),outFolder,showBlokeh=True)
                        showBlokeh = False
                    else:
                        showResults(dataset,points,predClust,realsLab,transformName+" "+'Cluster '+str(name),outFolder)
                    currentCol += 1



            if doEncoderLabel:
                d_in = sess.graph.get_tensor_by_name(discrInputName)
                d_encoder = sess.graph.get_tensor_by_name(discrEncoderName)
                points,predEncoder,realsLab = encoderLabeling(sess,dataset,d_in,dtransform,d_encoder)

                print "Showing results for Encoder labeling"
                showResults(dataset,points,predEncoder,realsLab,transformName+" "+'Encoder',outFolder)
                currentCol += 1
        plt.savefig(os.path.join(outFolder,'table1.png'))



if __name__ == '__main__':
    # device_name = "/cpu:0"
    # with tf.device(device_name):
    main()



