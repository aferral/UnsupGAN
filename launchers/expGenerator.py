import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
import sys
from scipy.misc import imsave
from skimage.transform import resize
from skimage.transform import rescale



#Generate n samples from set catActiva (we left only catActiva in the C input as 1 ex: catActiva=1 [0 1 0 0] )
def doSampleFromSetC(sess,layerOut,layerInput,catActiva,nSamples,fixedNoiseSamples,batchSize,noiseSize,cSize,isTan):
	assert(nSamples < batchSize) #i was lazy and i didnt want to run a lot of times the network.

	valInput = np.copy(fixedNoiseSamples)
	valInput[:,-10:] = 0  #Setting all C inputs to 0 (they are after the noise values)
	valInput[:,-catActiva] = 1 #Setting catActiva as 1 to get one-hot encoding in first part of the input.

	print 'Showing cat code row0: ', valInput[0, -cSize:]
	print 'Showing cat code row1: ', valInput[1, -cSize:]
	print " "
	imagesBatch = sess.run(layerOut, {layerInput : valInput } )

	if isTan: #the result was in -1 to +1 units
		imagesBatch = (imagesBatch + 1.0 ) * 0.5

	return imagesBatch[0:nSamples],valInput[0:nSamples]

def oldTest(sess,outGen,inputGen,batchSize,noiseSize,cSize):

	testV = np.random.rand(1, noiseSize)
	print testV
	testCvector = np.zeros((1, cSize))

	joint = np.hstack([testV, testCvector])


	testVector = np.repeat(joint,batchSize,axis=0)
	testVector[0:cSize,-10:] = np.eye(cSize)
	print 'Ej cat row0 : ', testVector[0, -cSize:]
	print 'Ej cat row1 : ', testVector[1, -cSize:]

	resultado=sess.run(outGen, {inputGen: testVector})

	for i in range(10):
		toSave = rescale(resultado[i].reshape((28,28)) , 10)
		imsave(os.path.join('test'+str(i)+'.png'), toSave)
	return testVector[0:cSize]

def main(configFile,isTan):
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

	batchSize = res['batch_size']
	nSamples = 1

	noiseSize = 100
	cSize = res['categories']
	inputSize = noiseSize + cSize

	layerInputName = "concat:0"
	layerOutputName = res['discrInputName']

	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph(modelPath+'.meta')
		new_saver.restore(sess, modelPath)
		sigm = sess.graph.get_tensor_by_name(layerOutputName)
		entrada = sess.graph.get_tensor_by_name(layerInputName)


		temp=[]
		t1=[[] for i in range(cSize)]
		fixedNoiseSamples = np.random.rand(batchSize,inputSize)
		for catAct in range(cSize):
			print "Cact ",catAct
			print "Input Noise row0 ",fixedNoiseSamples[0,-20:]
			print "Input Noise row1 ",fixedNoiseSamples[1,-20:]
			imagesSampled,inputs = doSampleFromSetC(sess, sigm, entrada, catAct, nSamples,fixedNoiseSamples,batchSize,noiseSize,cSize,isTan)
			t1[catAct].append(inputs)

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
					expandedImage = resize(np.copy(imagesSampled[i]).reshape((28,28)), expandedShape)
				else:
					expandedImage = resize(np.copy(imagesSampled[i]), expandedShape)
				allInOne[expandedShape[0]*i:expandedShape[0]*(i+1),:] = np.copy(expandedImage)

			print allInOne.shape
			if len(allInOne.shape) == 3 and allInOne.shape[-1] == 1: #Make sure that allInOne is (x,y) and not (x,y,1)
				allInOne = allInOne.reshape(allInOne.shape[0:-1])
			imsave("Csamples " + exp_name + " right incresing C, Up random samples "+str(catAct)+ ' .png', allInOne)
			temp.append(np.copy(allInOne))

		out=np.hstack(temp)
		imsave("Csamples "+exp_name+" right incresing C, Up random samples "+'.png',out)

		t2=oldTest(sess,sigm, entrada,batchSize,noiseSize,cSize)

if __name__ == "__main__":
	print os.getcwd()
	configFile = sys.argv[1]
	isTan = bool(sys.argv[2])
	main(configFile, isTan)