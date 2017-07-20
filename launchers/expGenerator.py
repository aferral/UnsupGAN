import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
import sys
from infogan.misc.dataset import Dataset
from skimage.io import imsave
from skimage.transform import resize
from traditionalClusteringTests.dataUtils import inverseNorm

#--------------------PARAMETROS--------------------

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
	filename = os.path.split(configFile)[-1]
	assert (len(filename.split(".")) == 2)
	exp_name = filename.split(".")[0]

if res.has_key('modelPath'):
	modelPath = res['modelPath']
else:
	train_dataset = res['train_dataset']
	modelPath = os.path.join("ckt", train_dataset, exp_name, "last.ckpt")


name= exp_name
batchSize = res['batch_size']
nSamples = 20

noiseSize = 100
cSize = res['categories']
inputSize=noiseSize+cSize

isTan = True

layerInputName = "concat:0"
layerOutputName = res['discrInputName']
useDataset = Dataset(dataFolder)


#--------------------PARAMETROS--------------------

def runSess(sess, tensor, tensorInput, val):
	return sess.run(tensor, {tensorInput : val } )

#Generate n samples from set catActiva (we left only catActiva in the C input as 1 ex: catActiva=1 [0 1 0 0] )
def doSampleFromSetC(sess,layerOut,layerInput,catActiva,nSamples):
	assert(nSamples < batchSize) #i was lazy and i didnt want to run a lot of times the network

	valInput = np.random.rand(batchSize,inputSize)
	valInput[:,0:cSize] = 0  #Setting all C inputs to 0
	valInput[:,catActiva] = 1 #Setting catActiva as 1 to get one-hot encoding in first part of the input.
	activations = runSess(sess, layerOut, layerInput, valInput)

	if isTan: #the result was in -1 to +1 units
		resultado = (activations + 1.0 ) * 0.5

	return activations[0:nSamples]


with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph(modelPath+'.meta')
	new_saver.restore(sess, modelPath)
	sigm = sess.graph.get_tensor_by_name(layerOutputName)
	entrada = sess.graph.get_tensor_by_name(layerInputName)


	for catAct in range(cSize):
		imagesSampled = doSampleFromSetC(sess, sigm, entrada, catAct, nSamples)
		shape=imagesSampled[0].shape
		nImages= imagesSampled.shape[0]

		factor=3
		expandedShape = tuple([elem*factor for elem in shape[0:2]]+[shape[-1]])


		allInOne = np.empty(tuple([expandedShape[0]*nImages]+list(expandedShape[1:])))
		for i in range(nImages):
			expandedImage = resize(imagesSampled[i], expandedShape)
			allInOne[expandedShape[0]*i:expandedShape[0]*(i+1),:]=imagesSampled[i]
		print allInOne.shape
		if len(allInOne.shape) == 3 and allInOne.shape[-1] == 1:
			allInOne = allInOne.reshape(allInOne.shape[0:-1])
		imsave(name+"_cat_"+str(catAct)+'.png',allInOne)
		

