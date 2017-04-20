import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import rescale
import os
from infogan.misc.datasets import DataFolder
from sklearn.cluster import KMeans
from collections import Counter


#TODO generalize to VAL TEST SET


def pool_features(feat, pool_type='avg'):
    if len(feat.shape) >= 3:
        if pool_type == 'avg':
            feat = feat.mean(axis=(1, 2))
        if pool_type == 'max':
            feat = feat.mean(axis=(1, 2))
    return feat.reshape((feat.shape[0], feat.shape[-1]))
		#Get last layer vector



def clusterLabeling(dataset,d_in,d_feat):
	
	trainX = np.array([]).reshape(0, 0)

	#From train data create clustering
        for ii in range(dataset.batch_idx['train']):
		x, _ = dataset.next_batch(dataset.batch_size)

		d_features = sess.run(d_feat, {dataset: x})
		d_features = pool_features(d_features, pool_type='avg')
		d_features_norm = normalize(d_features, axis=1, norm='l2')

		if trainX.shape[0] == 0:  # Is empty
			trainX = d_features
		else:
			trainX = np.concatenate((trainX, d_features), axis=0)

        print "Learning the clusters."
	#Learn the clusters
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(trainX_norm)


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

		pred = kmeans.predict(d_features_norm)
		
		predited = predited + list(pred)
		realLabels = realLabels + list(batch_labels)

	return transformed,predicted,realLabels


def encoderLabeling(dataset,d_in,d_feat,d_encoder):
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
	return datasetRep,labels


def showPCA2(points,labels):
	pca = PCA(n_components=2)
	pca.fit(points)
	# Show a few statistics of the data
	print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)
	print "PCA 2 comp of the data (using train)"

	transformed = pca.transform(points)

	plt.figure()
	allscatter = []
	n_classes = len(set(labels))
	for c in range(n_classes):
		elements = np.where(labels == c)
		temp = plt.scatter(transformed[elements, 0], transformed[elements, 1],
				   facecolors='none', label='Class ' + str(c), c=np.random.rand(3, 1))
		allscatter.append(temp)
	plt.legend(tuple(allscatter),
	       tuple(["class " + str(c) for c in range(n_classes)]),
	       scatterpoints=1,
	       loc='lower left',
	       ncol=3,
	       fontsize=8)
	plt.show()

def showResults(dataset,points,labels,realLabels):
	#Mostrar plot pca2 del clustering o de classify
	print "Showing PCA2 with predicted labels"
	showPCA2(points,labels)

	print "Showing PCA2 with real labels"
	showPCA2(points,realLabels)

	print "The ARI was ",2
	print "The NMI was ",2

	n_classes = len(set(labels))
	labels = np.array(labels)

	for i in range(n_classes):
		#Get all index of that class
		elements = np.where(labels == i)\

		#Make a KD-tree to seach nearest points
		tree = spatial.KDTree(points)

		print "For class ",i," there are ",elements.shape[0]
		#Get the real classes for those points
		rl = realLabels[elements]
		#Show distribution
		dist = Counter(rl)
		print "Showing Real distribution for that generated Label ",str(dist)

		#Calculate centroid
		centroid = np.mean(elements,axis=0)
		#Get X closest points from that centroid
		toShow =10
		distances,indexs = tree.query(centroid,k=toShow)
		#Get those points images
		for j in range(toShow):
			image = dataset.dataObj.trainData[elements[indexs[j]]]
			plt.figure()
			plt.imshow(image)
			plt.title('Close '+str(j)+" dist "+str(distances[j]))
		plt.show()

		


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
#This has to be the saved batch_size 
batch_size = 128

#----------------------PARAMETERS --------------------


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
			points,predClust,realsLab = clusterLabeling(dataset,d_in,d_feat)

			#Show results
			print "Showing results for Cluster labeling"
			showResults(points,predClust,realsLab)
		if doEncoderLabel:

			points,predEncoder,realsLab = encoderLabeling(dataset,d_in,d_feat,d_encoder)

			#Show results
			print "Showing results for Encoder labeling"
			showResults(pointsDataset,labelsEncoder,realsLab)

if __name__ == '__main__':
	main()
	


