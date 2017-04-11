import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import rescale
import os

#--------------------PARAMETROS--------------------

modelPath = 'ckt/mnist/testC/t1.ckpt'
#modelPath = 'ckt/mnist/testN/t2.ckpt'
#modelPath = 'ble.ckpt'

isCat = True
ncat = 10
nNoise = 100
batchSize = 128

dimToTest = 2
testCat = 9

outFolder = 'imagenesTest'
#--------------------PARAMETROS--------------------

def rG(sess,fun,inp,val):
	return sess.run(fun, {inp : val } )
#Variar una dimension y mostrar 128 imagenes
def test1(sess,fun,inp,v0,dim):

	testVector = np.repeat(v0,batchSize,axis=0)
	testVector[:,dim] = np.linspace(-1,1,batchSize)

	resultado = rG(sess,fun,inp,testVector)

	for i in range(batchSize):
		toSave = rescale(resultado[i].reshape((28,28)) , 10)
		imsave(os.path.join(outFolder,'test'+str(i)+'.png'), toSave)

def test2(sess,fun,inp,v0,nc):
	testVector = np.repeat(v0,batchSize,axis=0)
	testVector[0:nc,-10:] = np.eye(nc)

	resultado = rG(sess,fun,inp,testVector)

	for i in range(10):
		toSave = rescale(resultado[i].reshape((28,28)) , 10)
		imsave(os.path.join(outFolder,'test'+str(i)+'.png'), toSave)
#Crear una ventana con perillitas
def test3():
	#Consigue dataset TRAIN
	#clusteriza
	#consigue labels
	#Mostrar lista de imagenes de cada label en carpetas separadas.
	#Mostrar centroide, mostrar distancia a el centroide
	pass


def main():
	print os.getcwd()

	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph(modelPath+'.meta')
		new_saver.restore(sess, modelPath)
		sigm = sess.graph.get_tensor_by_name('Sigmoid:0')
		entrada = sess.graph.get_tensor_by_name('concat:0')
		testV = np.random.rand(1,nNoise)

		print 'Ej ruido : ',testV[0,0:10]
		
		if isCat:
			testCvector = np.zeros((1, ncat))
			testCvector[np.arange(1), testCat] = 1

			joint=np.hstack([testV,testCvector])
			print 'Ej cat : ',testV[0,-ncat:]
			#test1(sess,sigm,entrada,joint,dimToTest)
			test2(sess,sigm,entrada,joint,ncat)
		else:
			test1(sess,sigm,entrada,testV,dimToTest)
			

		


if __name__ == '__main__':
	main()
