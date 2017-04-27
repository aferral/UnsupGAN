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

#TODO generalize to VAL TEST SET
#TODO This can scale to large datasets ????

#----------------------PARAMETERS --------------------
#Discriminator TEST
dataFolder = "data/MFPT96Scalograms"
modelPath = 'ckt/t-dataFolder_v-dataFolder_o-64_c-4_2017_04_18_17_30_17_2000.ckpt'

#Define methods to use for label

doCluster = True
doEncoderLabel = True

#Define in and outputs to use
discrInputName = "apply_op_4/Tanh:0"
discrLastFeatName = "custom_fully_connected_6/Linear/add:0"
discrEncoderName = "Softmax_1:0"
outFolder = "imagenesTest"
#This has to be the saved batch_size
batch_size = 128

nClustersTofind = 3 #Some cluster methods requiere the number of labels to seach

#----------------------PARAMETERS --------------------

def pool_features(feat, pool_type='avg'):
    if len(feat.shape) >= 3:
        if pool_type == 'avg':
            feat = feat.mean(axis=(1, 2))
        if pool_type == 'max':
            feat = feat.mean(axis=(1, 2))
    return feat.reshape((feat.shape[0], feat.shape[-1]))

def extractPointsLstConv(sess, dataset, lastConvTensor, inputTensor):
    trainX = np.array([]).reshape(0, 0)
    # From train data create clustering
    for ii in range(dataset.batch_idx['train']):
        x, _ = dataset.next_batch(dataset.batch_size)

        d_features = sess.run(lastConvTensor, {inputTensor: x})
        d_features = pool_features(d_features, pool_type='avg')
        d_features_norm = normalize(d_features, axis=1, norm='l2')

        if trainX.shape[0] == 0:  # Is empty
            trainX = d_features_norm
        else:
            trainX = np.concatenate((trainX, d_features_norm), axis=0)
        return trainX

def clusterLabeling(sess,dataset,d_in,d_feat,clusterAlg,trainX):

    print "Learning the clusters."
    #Learn the clusters
    clusterAlg.fit(trainX)

    #Now predict validation and train data
    realLabels = []
    predited = []
    transformed = np.array([]).reshape(0, 0)


    #Predict TrainData data
    for ii in range(dataset.batch_idx['train']):
        x, batch_labels = dataset.next_batch(dataset.batch_size, split="train")

        d_features = sess.run(d_feat, {d_in : x})
        d_features = pool_features(d_features, pool_type='avg')
        d_features_norm = normalize(d_features, axis=1, norm='l2')

        if transformed.shape[0] == 0:  # Is empty
            transformed = d_features_norm
        else:
            transformed = np.concatenate((transformed, d_features_norm), axis=0)

        pred = clusterAlg.predict(d_features_norm)

        predited = predited + list(pred)
        realLabels = realLabels + list(batch_labels)

    #If realLabels are in one hot pass to list of integer labels
    if realLabels[0].shape[0] > 1:
        realLabels = np.where(realLabels)[1].tolist()
    else:
        realLabels = realLabels.tolist()

    return transformed,predited,realLabels


def encoderLabeling(sess,dataset,d_in,d_feat,d_encoder):
    #Now predict validation and train data
    realLabels = []
    predited = []
    transformed = np.array([]).reshape(0, 0)


    #Predict TrainData data
    for ii in range(dataset.batch_idx['train']):
        x, batch_labels = dataset.next_batch(dataset.batch_size, split="train")

        d_features = sess.run(d_feat, {d_in: x})
        d_features = pool_features(d_features, pool_type='avg')
        d_features_norm = normalize(d_features, axis=1, norm='l2')

        if transformed.shape[0] == 0:  # Is empty
            transformed = d_features_norm
        else:
            transformed = np.concatenate((transformed, d_features_norm), axis=0)

        pred = np.argmax(sess.run(d_encoder, {d_in : x}), axis=1)

        predited = predited + list(pred)
        realLabels = realLabels + list(batch_labels)
    # If realLabels are in one hot pass to list of integer labels
    if realLabels[0].shape[0] > 1:
        realLabels = np.where(realLabels)[1].tolist()
    else:
        realLabels = realLabels.tolist()
    return transformed,predited,realLabels


def showDimRed(points, labels, name,dimRalg):
    dimRalg.fit(points)

    transform_op = getattr(dimRalg, "transform", None)
    if callable(transform_op):
        transformed = dimRalg.transform(points)
    else:
        transformed = dimRalg.fit_transform(points)

    plt.figure()
    allscatter = []
    n_classes = len(set(labels))
    for c in range(n_classes):
        elements = np.where(np.array(labels) == c)
        temp = plt.scatter(transformed[elements, 0], transformed[elements, 1],
                   facecolors='none', label='Class ' + str(c), c=np.random.rand(3, 1))
        allscatter.append(temp)
    plt.legend(tuple(allscatter),
           tuple(["class " + str(c) for c in range(n_classes)]),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
    plt.savefig(os.path.join(outFolder,name+'.png'))

def showResults(dataset,points,labels,realLabels,name):
    #Mostrar plot pca2 del clustering o de classify
    outNameFile = "Results for "+str(name)+'.txt'
    log = ""


    #SHOW PCA2 of data with PREDICTED labels
    log += ("Showing PCA2 with predicted labels"+'\n')
    pca = PCA(n_components=2)
    print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)
    showDimRed(points, labels, name + str('PCA_Predicted'),pca)

    # SHOW 	TSNE of data with REAL labels
    log += ("Showing PCA2 with real labels"+'\n')
    pca = PCA(n_components=2)
    showDimRed(points, realLabels, name + str('PCA_Real'),pca)

    # SHOW 	TSNE of data with PREDICTED labels
    log += ("Showing TSNE with predicted labels" + '\n')
    model = TSNE(n_components=2)
    print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)
    showDimRed(points, labels, name + str('TSNE_Predicted'), model)

    # SHOW PCA2 of data with REAL labels
    log += ("Showing TSNE with real labels" + '\n')
    model = TSNE(n_components=2)
    showDimRed(points, realLabels, name + str('TSNE_Real'), model)

    log += ("The ARI was "+str(metrics.adjusted_rand_score(realLabels, labels) )+'\n')
    log += ("The NMI was "+str( metrics.normalized_mutual_info_score(realLabels, labels))+'\n')

    n_classes = len(set(labels))
    labels = np.array(labels)

    for i in range(n_classes):
        tempFolder = name+' Predicted '+str(i)
        if not os.path.exists(os.path.join(outFolder,tempFolder)):
            os.makedirs(os.path.join(outFolder,tempFolder))

        #Get all index of that class
        elements = np.where(labels == i)\

        #Make a KD-tree to seach nearest points
        tree = spatial.KDTree(points)

        log += ("For class "+str(i)+" there are "+str(elements[0].shape[0])+'\n')
        #Get the real classes for those points
        rl = np.array(realLabels)[elements]
        #Show distribution
        dist = Counter(rl)
        log += ("Showing Real distribution for that generated Label "+str(dist)+'\n')

        #Calculate centroid
        selected = points[elements]
        centroid = np.mean(selected,axis=0)
        #Get X closest points from that centroid
        toShow =10
        distances,indexs = tree.query(centroid,k=toShow)
        #Get those points images
        for j in range(toShow):
            image = dataset.dataObj.train_data[indexs[j]] #TODO DONT USE THE PRIVATE VARIABLES
            label = dataset.dataObj.train_labels[indexs[j]] #HERE IS THE PROBLEM FOR THE VAL - TRAIN CASE. If you merge the you cant get the images back

            title = 'PredictedLabel '+str(i)+" Cls " + str(j) + " dist " + str(distances[j]) + " RealLabel " + str(label)
            toSave = rescale(image , 2)
            toSave = (toSave / 255).astype(np.float)
            imsave(os.path.join(outFolder,tempFolder,title+'.png'), toSave)

    with open(os.path.join(outFolder,outNameFile),'w+') as f:
        f.write(log)


def main():
    #Get dataset
    #TODO by the moment it just uses train to cluster and predict
    dataset = DataFolder(dataFolder,batch_size,testProp=0.01, validation_proportion=0.001)


    #Load discriminator
    print os.getcwd()

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath+'.meta')
        new_saver.restore(sess, modelPath)

        #Load in and outs to use
        d_in = None
        d_feat = None
        d_encoder = None

        if not(discrInputName is None):
            d_in  = sess.graph.get_tensor_by_name(discrInputName)
        if not(discrLastFeatName is None):
            d_feat  = sess.graph.get_tensor_by_name(discrLastFeatName)
        if not(discrEncoderName is None):
            d_encoder  = sess.graph.get_tensor_by_name(discrEncoderName)

        #Check if we have the data to run test
        assert( not(d_in is None))
        assert(not(d_feat is None) )
        assert( not(doEncoderLabel) or (doEncoderLabel and not(d_encoder is None) ))

        if doCluster:


            trainX = extractPointsLstConv(sess, dataset, d_feat, d_in)

            #----------------Define cluster methods----------------------------------------
            # estimate bandwidth for mean shift-
            bandwidth = cluster.estimate_bandwidth(trainX, quantile=0.3)

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(trainX, n_neighbors=10, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # create clustering estimators
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=nClustersTofind)
            ward = cluster.AgglomerativeClustering(n_clusters=nClustersTofind, linkage='ward',
                                                   connectivity=connectivity)
            spectral = cluster.SpectralClustering(n_clusters=nClustersTofind,
                                                  eigen_solver='arpack',
                                                  affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=.2)
            affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                               preference=-200)

            average_linkage = cluster.AgglomerativeClustering(
                linkage="average", affinity="cityblock", n_clusters=nClustersTofind,
                connectivity=connectivity)

            birch = cluster.Birch(n_clusters=nClustersTofind)
            clustering_algorithms = [
                two_means, affinity_propagation, ms, spectral, ward, average_linkage,
                dbscan, birch]

            # ----------------Define cluster methods----------------------------------------
            for clusterAlg in clustering_algorithms:
                points,predClust,realsLab = clusterLabeling(sess,dataset,d_in,d_feat,clusterAlg,trainX)
                #Show results
                print "Showing results for Cluster ",str(clusterAlg)[0:20]
                showResults(dataset,points,predClust,realsLab,'Cluster')


        if doEncoderLabel:

            points,predEncoder,realsLab = encoderLabeling(sess,dataset,d_in,d_feat,d_encoder)

            #Show results
            print "Showing results for Encoder labeling"
            showResults(dataset,points,predEncoder,realsLab,'Encoder')

if __name__ == '__main__':
    main()



