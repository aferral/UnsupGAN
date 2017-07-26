import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
import sys
from infogan.misc.dataset import Dataset
from scipy.misc import imsave
from skimage.transform import resize
from traditionalClusteringTests.dataUtils import inverseNorm
from skimage.transform import rescale
#--------------------PARAMETROS--------------------

print os.getcwd()
configFile = sys.argv[1]
isTan = bool(sys.argv[2])
print "Loading config file ", configFile

res = {}

with open(configFile, 'r') as f:
	res = json.load(f)

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
nSamples = 10

noiseSize = 100
cSize = res['categories']
inputSize=noiseSize+cSize

layerInputName = "concat:0"
layerOutputName = res['discrInputName']


#--------------------PARAMETROS--------------------

def runSess(sess, tensor, tensorInput, val):
	return sess.run(tensor, {tensorInput : val } )

#Generate n samples from set catActiva (we left only catActiva in the C input as 1 ex: catActiva=1 [0 1 0 0] )
def doSampleFromSetC(sess,layerOut,layerInput,catActiva,nSamples,fixedNoiseSamples):
	assert(nSamples < batchSize) #i was lazy and i didnt want to run a lot of times the network.

	valInput = fixedNoiseSamples
	valInput[:,noiseSize:noiseSize+cSize] = 0  #Setting all C inputs to 0 (they are after the noise values)
	valInput[:,noiseSize+catActiva] = 1 #Setting catActiva as 1 to get one-hot encoding in first part of the input.
	imagesBatch = runSess(sess, layerOut, layerInput, valInput)

	if isTan: #the result was in -1 to +1 units
		imagesBatch = (imagesBatch + 1.0 ) * 0.5

	return imagesBatch[0:nSamples]

def oldTest(outGen,inputGen):
	testV = np.random.rand(1, noiseSize)
	testCvector = np.zeros((1, cSize))

	joint = np.hstack([testV, testCvector])
	print 'Ej cat : ', joint[0, -cSize:]

	testVector = np.repeat(joint,batchSize,axis=0)
	testVector[0:cSize,-10:] = np.eye(cSize)

	resultado=sess.run(outGen, {inputGen: testVector})

	for i in range(10):
		toSave = rescale(resultado[i].reshape((28,28)) , 10)
		imsave(os.path.join('test'+str(i)+'.png'), toSave)




with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph(modelPath+'.meta')
	new_saver.restore(sess, modelPath)
	sigm = sess.graph.get_tensor_by_name(layerOutputName)
	entrada = sess.graph.get_tensor_by_name(layerInputName)


	temp=[]
	fixedNoiseSamples = np.random.rand(batchSize,inputSize)
	for catAct in range(cSize):
		imagesSampled = doSampleFromSetC(sess, sigm, entrada, catAct, nSamples,fixedNoiseSamples)

		if ("MNIST" in exp_name):
			shape = (28, 28, 1)
		else:
			shape = imagesSampled[0].shape

		nImages= imagesSampled.shape[0]
		factor=3
		expandedShape = tuple([elem*factor for elem in shape[0:2]]+[shape[-1]])


		allInOne = np.empty(tuple([expandedShape[0]*nImages]+list(expandedShape[1:])))
		for i in range(nImages):

			if ("MNIST" in exp_name):
				expandedImage = resize(imagesSampled[i].reshape((28,28)), expandedShape)
			else:
				expandedImage = resize(imagesSampled[i], expandedShape)


			allInOne[expandedShape[0]*i:expandedShape[0]*(i+1),:] = expandedImage
		print allInOne.shape
		if len(allInOne.shape) == 3 and allInOne.shape[-1] == 1: #Make sure that allInOne is (x,y) and not (x,y,1)
			allInOne = allInOne.reshape(allInOne.shape[0:-1])

		temp.append(np.copy(allInOne))

	out=np.empty((allInOne.shape))
	for elem in temp:
		out=np.hstack([out,elem])
	imsave("Csamples "+name+" right incresing C, Up random samples "+'.png',out)

	oldTest(sigm, entrada)

