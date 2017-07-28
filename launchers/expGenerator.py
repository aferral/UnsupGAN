import numpy as np
import tensorflow as tf
import json
import os
import sys
from scipy.misc import imsave
from skimage.transform import resize
from skimage.transform import rescale

def plotSample(sess,layerOut,layerInput,val,nSamples,batchSize,isTan,isMnist,gridShape,outName,factor):
	assert(nSamples < batchSize)  # i was lazy and i didnt want to run a lot of times the network.
	imagesBatch = sess.run(layerOut, {layerInput: val})

	imagesBatch = imagesBatch[0:nSamples]
	if isTan: #the result was in -1 to +1 units
		imagesBatch = (imagesBatch + 1.0 ) * 0.5

	if isMnist:
		imageShape = (28, 28, 1)
	else:
		imageShape = imagesBatch[0].shape
	assert(len(imageShape)==3)
	assert(len(gridShape) == 2)

	outImage = np.zeros((imageShape[0]*gridShape[0]*factor,imageShape[1]*gridShape[1]*factor,imageShape[2]))
	#Fill image from rows then columns (typical matrix order)

	rows=gridShape[0]
	cols=gridShape[1]
	for i in range(rows):
		for j in range(cols):
			iSample = i*rows+j
			if iSample >= nSamples:
				break
			xinf,xsup = i*imageShape[0]*factor,(i+1)*imageShape[0]*factor
			yinf,ysup = j*imageShape[1]*factor,(j+1)*imageShape[1]*factor
			outImage[xinf:xsup , yinf:ysup] = rescale(imagesBatch[iSample].reshape(imageShape), factor)

	if len(outImage.shape) == 3 and outImage.shape[-1] == 1:  # Make sure that allInOne is (x,y) and not (x,y,1)
		outImage = outImage.reshape(outImage.shape[0:-1])
	imsave(outName+ '.png', outImage)



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

	noiseSize = 100
	factor=2
	cSize = res['categories']
	inputSize = noiseSize + cSize
	isMNIST= "MNIST" in exp_name

	layerInputName = "concat:0"
	layerOutputName = res['discrInputName']

	assert(cSize>0)
	assert(cSize<11) #You need to fix the setup grid is 10x10 for the categorical

	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph(modelPath+'.meta')
		new_saver.restore(sess, modelPath)
		outLayer = sess.graph.get_tensor_by_name(layerOutputName)
		entrada = sess.graph.get_tensor_by_name(layerInputName)

		#---------------------------Sample aleatoria---------------------------
		listC=np.random.randint(cSize, size=64)
		matC = np.zeros((64,cSize))
		matC[np.arange(64),listC]=1

		val=np.random.rand(batchSize,inputSize)
		val[0:64,-cSize:]=matC
		plotSample(sess,outLayer,entrada,val,64,batchSize,isTan,isMNIST,(8,8),exp_name+" randomSample",factor=factor)

		# ---------------------------Z fix random C---------------------------
		# En este experimento logre que fueran todos diferentes entre C y luego se repiten
		# Que es lo que se espera que ocurra
		listC = np.random.randint(cSize, size=64)
		matC = np.zeros((64, cSize))
		matC[np.arange(64), listC] = 1

		val = np.random.rand(1, inputSize)
		val = np.repeat(val, batchSize, axis=0)
		val[0:64, -cSize:] = matC
		plotSample(sess, outLayer, entrada, val, 64, batchSize, isTan, isMNIST, (8, 8),exp_name+" ZfixedRandomC",factor=factor)

		# ---------------------------C fixed random Z---------------------------
		# En este experimento salen derrepente muy mezclados. Cuidado el C vector no es para esto
		# Al parecer no es global sino "local"
		matC = np.zeros((64, cSize))
		catTouse=np.random.randint(cSize)
		matC[np.arange(64), catTouse] = 1

		val = np.random.rand(batchSize, inputSize)
		val[0:64, -cSize:] = matC
		plotSample(sess, outLayer, entrada, val, 64, batchSize, isTan, isMNIST, (8, 8), exp_name+" CfixedRandomZ_"+str(catTouse),factor=factor)

		# ---------------------------C fixed by row---------------------------
		gridX,gridY= cSize, 10
		samples=gridX*gridY
		assert(samples<batchSize)

		matC = np.zeros((samples, cSize))
		for i in range(cSize):
			matC[i*gridY:(i+1)*gridY, i] = 1

		val = np.random.rand(batchSize, inputSize)
		val[0:samples, -cSize:] = matC
		plotSample(sess, outLayer, entrada, val, samples, batchSize, isTan, isMNIST, (gridX, gridY), exp_name+" CfixRow",factor=factor)

		# ---------------------------C fijo Zfijo cruzados---------------------------
		gridX, gridY = cSize, 10
		samples = gridX * gridY
		assert (samples < batchSize)

		matC = np.zeros((samples, cSize))
		for i in range(cSize):
			matC[i * gridY:(i + 1) * gridY, i] = 1

		realNoiseM = np.random.rand(batchSize, inputSize)
		randomNoises = np.random.rand(gridY, inputSize)

		indexs = [i for i in range(samples)]
		for i in range(cSize):
			realNoiseM[np.array(filter(lambda x: x % cSize == i, indexs)), :] = randomNoises[i]

		realNoiseM[0:samples, -cSize:] = matC
		plotSample(sess, outLayer, entrada, realNoiseM, samples, batchSize, isTan, isMNIST, (gridX, gridY), exp_name+" CZcross",factor=factor)


if __name__ == "__main__":
	print os.getcwd()
	configFile = sys.argv[1]
	isTan = bool(sys.argv[2])
	main(configFile, isTan)