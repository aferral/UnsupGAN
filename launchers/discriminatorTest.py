import numpy as np
import pylab as plt
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import rescale
import os

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

#----------------------PARAMETERS ALL DEFINED IN CONFIG FILE --------------------
from infogan.misc.utilsTest import generate_new_color

configFile = sys.argv[1]
print os.getcwd()
print "Loading config file ",configFile

res = {}
with open(configFile,'r') as f:
    res = json.load(f)

dataFolder = res['dataFolder']
modelPath = res['modelPath']

#Define methods to use for label (BOOLEAN)
doCluster = res['doCluster']
doEncoderLabel = res['doEncoderLabel']
showTNSE = res['showTNSE']

#Define in and outputs to use (String)
discrInputName = res['discrInputName']
discrLastFeatName = res['discrLastFeatName']
discrEncoderName = res['discrEncoderName']
#Define output folder for results (STRING)
outFolder = res['outFolder']
#This has to be the saved batch_size (INT)
batch_size = res['batch_size']
#The image size used in the arch (int) (square image)
imageSize = res['imageSize']

#Define data_transform to use "convFeat" OR "convFeatEncoder" (STRING)
nameDataTransform = res['nameDataTransform']
nClustersTofind = res['nClustersTofind']


#----------------------PARAMETERS --------------------


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

def OneHotToInt(labels):
    #If realLabels are in one hot pass to list of integer labels
    if labels[0].shape[0] > 1:
        labels = np.where(labels)[1].tolist()
    else:
        labels = labels.tolist()
    return labels

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


def showDimRed(points, labels, name,dimRalg, ax=None):

    transform_op = getattr(dimRalg, "transform", None)
    if callable(transform_op):
        dimRalg.fit(points)
        transformed = dimRalg.transform(points)
    else:
        transformed = dimRalg.fit_transform(points)

    objPlot = None
    if ax is None:
        plt.figure()
        objPlot = plt
    else:
        objPlot = ax

    allscatter = []
    n_classes = len(set(labels))

    colors = []
    for i in range(0, n_classes):
        colors.append(generate_new_color(colors, pastel_factor=0.9))

    for c in range(n_classes):
        elements = np.where(np.array(labels) == c)
        temp = objPlot.scatter(transformed[elements, 0], transformed[elements, 1],
                   facecolors='none', label='Class ' + str(c), c=colors[c])
        allscatter.append(temp)
    objPlot.legend(tuple(allscatter),
           tuple(["class " + str(c) for c in range(n_classes)]),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
    if ax is None:
        plt.savefig(os.path.join(outFolder,name+'.png'))

def showResults(dataset,points,labels,realLabels,name,ax=None):
    outNameFile = "Results for "+str(name)+'.txt'
    log = ""


    #SHOW PCA2 of data with PREDICTED labels
    log += ("Showing PCA2 with predicted labels"+'\n')
    pca = PCA(n_components=2)
    showDimRed(points, labels, name + str('PCA_Predicted'),pca,ax=ax)
    print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)


    n_classes = len(set(labels))
    labels = np.array(labels)

    globalClassScore = []

    show = False
    if n_classes > 60:
        show = False
        log += ("The number of cluster is > 60 So no images will be shown "+ '\n')

    for i in range(n_classes):
        tempFolder = name+' Predicted '+str(i)
        if not os.path.exists(os.path.join(outFolder,tempFolder)):
            os.makedirs(os.path.join(outFolder,tempFolder))

        #Get all index of that class
        elements = np.where(labels == i)

        #Make a KD-tree to seach nearest points
        tree = spatial.KDTree(points)

        log += ("For class "+str(i)+" there are "+str(elements[0].shape[0])+'\n')
        #Get the real classes for those points
        rl = np.array(realLabels)[elements]
        #Show distribution
        dist = Counter(rl)
        log += ("Showing Real distribution for that generated Label "+str(dist)+'\n')
        pdist = [(elem,dist[elem]*1.0/sum(dist.values())) for elem in set(dist.elements())]
        log += ("%dist "+str(pdist)+'\n')

        #place the clasification score
        classScore = max(pdist, key = lambda x : x[1])[1]
        log += ("Clasification score " + str(classScore) + '\n')
        globalClassScore.append(classScore)

        if show:
            #Calculate centroid
            selected = points[elements]
            centroid = np.mean(selected,axis=0)
            #Get X closest points from that centroid
            toShow =10
            distances,indexs = tree.query(centroid,k=toShow)
            #Get those points images
            for j in range(toShow):
                image = dataset.dataObj.validation_data[indexs[j]] #TODO DONT USE THE PRIVATE VARIABLES
                label = dataset.dataObj.validation_labels[indexs[j]] #HERE IS THE PROBLEM FOR THE VAL - TRAIN CASE. If you merge the you cant get the images back

                title = 'PredictedLabel '+str(i)+" Cls " + str(j) + " dist " + str(distances[j]) + " RealLabel " + str(label)
                toSave = rescale(image , 2)
                toSave = (toSave / 255).astype(np.float)
                imsave(os.path.join(outFolder,tempFolder,title+'.png'), toSave)

    #Log all the global information
    log += ('\n')
    log += ('Global metrics \n')
    log += ('\n')
    log += ("The ARI was " + str(metrics.adjusted_rand_score(realLabels, labels)) + '\n')
    log += ("The NMI was " + str(metrics.normalized_mutual_info_score(realLabels, labels)) + '\n')
    log += ("Predicted cluster number is " + str(n_classes) + '\n')
    log += ("The mean classificationScore is " + str(np.mean(np.array(globalClassScore))) + '\n')
    log += ("The std classificationScore is " + str(np.std(np.array(globalClassScore))) + '\n')

    with open(os.path.join(outFolder,outNameFile),'w+') as f:
        f.write(log)


def main():
    #Get dataset
    #TODO by the moment it just uses train to cluster and predict
    dataset = DataFolder(dataFolder,batch_size,testProp=0.01, validation_proportion=0.5, out_size=imageSize)


    #Load discriminator
    print os.getcwd()

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath+'.meta')
        new_saver.restore(sess, modelPath)

        #Load in and outs to use
        d_in = None
        d_feat = None
        d_encoder = None

        if not(discrInputName == 'None'):
            d_in  = sess.graph.get_tensor_by_name(discrInputName)
        if not(discrLastFeatName == 'None'):
            d_feat  = sess.graph.get_tensor_by_name(discrLastFeatName)
        if not(discrEncoderName == 'None'):
            d_encoder  = sess.graph.get_tensor_by_name(discrEncoderName)

        #Check if we have the data to run test
        assert( not(d_in is None))
        assert(not(d_feat is None) )
        assert( not(doEncoderLabel) or (doEncoderLabel and not(d_encoder is None) ))

        #Check and define data transform c,cen,cend define valid transform of data
        validTransforms = nameDataTransform.split(',')
        transformList = []

        for elem in validTransforms:
            if elem == "c":
                transformList.append(lambda x : transformFeature_Norm(x,sess,d_feat,d_in))
            elif elem == "cen":
                transformList.append(lambda x : transformFeatureAndEncoder(x,sess,d_feat,d_encoder,d_in))
            elif elem == 'cend':
                transformList.append(lambda x: transformFeatureAndEncoderDontNorm(x, sess, d_feat, d_encoder, d_in))
            else:
                raise Exception("ERROR DATA TRANSFORM NOT DEFINED")

        #Define grid of images for table1
        nAlgCluster = 0
        if doCluster:
            nAlgCluster += 8 #TODO DONT HARDCODE THIS
        if doEncoderLabel:
            nAlgCluster += 1

        f, axarr = plt.subplots(len(transformList), nAlgCluster)

        for indT,dtransform in enumerate(transformList):
            currentCol = 0

            trainX, rlbs = trainsetTransform(dtransform, dataset)
            rlbs = OneHotToInt(rlbs)

            if showTNSE:
                print "About to show TSNE with real labels (may take a while)"
                print "Showing TSNE with real labels"
                model = TSNE(n_components=2)
                showDimRed(trainX, rlbs, str('TSNE_Real'), model,ax=axarr[indT,currentCol])
                currentCol += 1

            # SHOW PCA2 of data with REAL labels
            pca = PCA(n_components=2)
            showDimRed(trainX, rlbs, str('PCA_Real'), pca,ax=axarr[indT,currentCol])
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
                    two_means, affinity_propagation, spectral, ward, average_linkage,
                    dbscan, birch]

                # ----------------Define cluster methods----------------------------------------
                for clusterAlg in clustering_algorithms:
                    points,predClust,realsLab = clusterLabeling(sess,dataset,dtransform,clusterAlg,trainX)
                    name = clusterAlg.__class__.__name__
                    print "Showing results for Cluster ",name
                    showResults(dataset,points,predClust,realsLab,'Cluster '+str(name),ax=axarr[indT,currentCol])
                    currentCol += 1

            if doEncoderLabel:
                points,predEncoder,realsLab = encoderLabeling(sess,dataset,d_in,dtransform,d_encoder)

                print "Showing results for Encoder labeling"
                showResults(dataset,points,predEncoder,realsLab,'Encoder',ax=axarr[indT,currentCol])
                currentCol += 1
        plt.show()



if __name__ == '__main__':
    main()



