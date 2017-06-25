import numpy as np


import matplotlib



matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from launchers.plot import plotBlokeh
import tensorflow as tf
from skimage.io import imsave
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

    dataFolder = res['dataFolder']
    if res.has_key('modelPath'):
        modelPath = res['modelPath']
    else:
        train_dataset = res['train_dataset']
        exp_name  = res['exp_name']
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
    outFolder = res['outFolder']
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


def showDimRed(points, labels, name,dimRalg,outF):

    transform_op = getattr(dimRalg, "transform", None)
    if callable(transform_op):
        dimRalg.fit(points)
        transformed = dimRalg.transform(points)
    else:
        transformed = dimRalg.fit_transform(points)

    objPlot = None

    plt.figure()
    objPlot = plt


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
           loc='upper left',
           ncol=3,
           fontsize=8)

    plt.savefig(os.path.join(outF,name+'.png'))

    return transformed

def showResults(dataset,points,labels,realLabels,name,ax=None,showBlokeh=False):
    outNameFile = "Results for "+str(name)+'.txt'
    log = ""


    #SHOW PCA2 of data with PREDICTED labels

    pca = PCA(n_components=2)

    transformed = showDimRed(points, labels, name + str('PCA_Predicted'), pca,outFolder)
    print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)

    #plot the 3pca of the transform
    plot3Dpca(dataset, points, labels, name, outFolder)

    if showBlokeh: #TODO showBlokeh is just true once in a transformation to be sure to plot just one real labels per transform
        plotInteractive(transformed, realLabels, dataset,name,outFolder)
        #plot the 3pca of the real labels. Just one per transform
        plot3Dpca(dataset, points, realLabels, name.split()[0]+"REAL_LABELS", outFolder)
        pass

    n_classes = len(set(labels))
    labels = np.array(labels)

    globalClassScore = []

    show = False
    if n_classes > 60:
        show = False
        log += ("The number of cluster is > 60 So no images will be shown "+ '\n')

    for i in range(n_classes):


        #Get all index of that class
        elements = np.where(labels == i)



        log += ("For class "+str(i)+" there are "+str(elements[0].shape[0])+'\n')
        #Get the real classes for those points
        rl = np.array(realLabels)[elements]
        #Show distribution
        dist = Counter(rl)
        log += ("Showing Real distribution for that generated Label "+str(dist)+'\n')
        totalPointsClass = sum(dist.values())
        pdist = [(elem,dist[elem]*1.0/ totalPointsClass) for elem in set(dist.elements())]
        log += ("%dist "+str(pdist)+'\n')

        #place the clasification score
        if len(pdist) > 0:
            classScore = max(pdist, key = lambda x : x[1])[1]
            classScore = classScore * ( totalPointsClass * 1.0 / labels.shape[0]) #Ponderate all classScore according to N of points
        else:
            classScore = 0
        log += ("Clasification score " + str(classScore) + '\n')
        globalClassScore.append(classScore)

        if show:
            # Make a KD-tree to seach nearest points
            tree = spatial.KDTree(points)
            tempFolder = name + ' Predicted ' + str(i)
            if not os.path.exists(os.path.join(outFolder, tempFolder)):
                os.makedirs(os.path.join(outFolder, tempFolder))
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
    log += ("The mean classificationScore is " + str(np.sum(np.array(globalClassScore))) + '\n')
    log += ("The std classificationScore is " + str(np.std(np.array(globalClassScore))) + '\n')
    log += ("PCA2 with predicted variance" + str(pca.explained_variance_ratio_)+ '\n')

    with open(os.path.join(outFolder,outNameFile),'w+') as f:
        f.write(log)

    if not(ax is None):
        return res


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

def plot3Dpca(dataset,points,labels,pickleName,outFolder):
    #Create 3d PCA and export pickle
    if type(labels) == list:
        labels = np.array(labels)

    fitted=PCA(n_components=3)
    fitted.fit(points)
    transformed = fitted.transform(points)

    names, labels, iamgeSave, saverls = fixrealLabels(dataset, labels)
    transformed = transformed[0:labels.shape[0]]
    assert (transformed.shape[0] == labels.shape[0])
    assert (transformed.shape[0] == names.shape[0])
    pickleName = os.path.join(outFolder,pickleName+" exp"+str(fitted.explained_variance_ratio_)+'.pkl')
    with open(pickleName, 'wb') as f:
        pickle.dump([transformed, labels, names, dataset.dataObj.getNclasses(), dataset.getFolderName()],f,-1)


# Save points,labels,name,fileNames TODO still here uses validation hardcoded
# TODO HERE FOR SOME REASON THE VAL DATA-LABELS ARE DESFASES BY 1 BATCH SIZE
# if you are in the points[0], realLabels[0] the corresponding saverls[128] iamgeSave[128]
def fixrealLabels(dataset,realLabels):
    iamgeSave = dataset.dataObj.validation_data  # TODO DONT USE THE PRIVATE VARIABLES
    saverls = np.where(dataset.dataObj.validation_labels)[1]  # HERE IS THE PROBLEM FOR THE VAL - TRAIN CASE. If you merge the you cant get the images back
    desfase = saverls.shape[0] - np.array(realLabels).shape[0]
    # THIS WILL MAKE THE saverls and realLabels coincide in values
    saverls = np.roll(saverls[:-1 * desfase], -dataset.dataObj.batch_size)
    names = [dataset.dataObj.getValFilename(i) for i in range(saverls.shape[0])]
    iamgeSave = np.roll(iamgeSave[:-1 * desfase], -dataset.dataObj.batch_size, axis=0)
    names = np.roll(names[:-1 * desfase], -dataset.dataObj.batch_size)

    realLabels = realLabels[0:names.shape[0]]
    return names,realLabels,iamgeSave,saverls


def plotInteractive(transformed,realLabels,dataset,name,outputFolder): #HERE IT MUST USE THE VALIDATION SET

    savepoints = transformed

    names,realLabels,iamgeSave,saverls = fixrealLabels(dataset,realLabels)

    assert (np.array_equal(saverls, realLabels))

    plotBlokeh([savepoints, realLabels, iamgeSave, names],name,outputFolder, test=False)
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
                        showResults(dataset, points, predClust, realsLab, transformName + " " + 'Cluster ' + str(name),showBlokeh=True)
                        showBlokeh = False
                    else:
                        showResults(dataset,points,predClust,realsLab,transformName+" "+'Cluster '+str(name))
                    currentCol += 1



            if doEncoderLabel:
                d_in = sess.graph.get_tensor_by_name(discrInputName)
                d_encoder = sess.graph.get_tensor_by_name(discrEncoderName)
                points,predEncoder,realsLab = encoderLabeling(sess,dataset,d_in,dtransform,d_encoder)

                print "Showing results for Encoder labeling"
                showResults(dataset,points,predEncoder,realsLab,transformName+" "+'Encoder')
                currentCol += 1
        plt.savefig(os.path.join(outFolder,'table1.png'))



if __name__ == '__main__':
    main()



