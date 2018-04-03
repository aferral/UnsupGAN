from sklearn import cluster
from sklearn.decomposition import PCA
import numpy as np
from infogan.misc.datasets import DataFolder
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os

from launchers.discriminatorTest import trainsetTransform, clusterLabeling
from traditionalClusteringTests.dataUtils import OneHotToInt, showDimRed, \
    showResults


def stringNow():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d-%H_%M_%S')


class AbstractUnsupModel(object):
    def __init__(self,dataset):
        self.dataset=dataset
        self.batchSize =dataset.batch_size
        self.imageShape = dataset.image_shape

        self.inputSizeFlatten = reduce(lambda x,y : x*y, self.imageShape)



    def train(self):
        raise NotImplementedError()

    def getLatentRepresentation(self,imageBatch):
        raise NotImplementedError()

    def reconstruction(self, imageBatch):
        raise NotImplementedError()

    def evaluate(self,outName,show = 3,nClustersTofind=3):

        outImageShape = self.imageShape[:-1] if (len(self.imageShape) == 3 and self.imageShape[2] == 1) else self.imageShape


        images, labeles = self.dataset.next_batch(self.batchSize)
        batchRecons = self.reconstruction(images)

        # Check dataset format (min - max values)
        minDataset = images.min()
        maxDataset = images.max()

        # Check reconstruction format (min - max values)
        minRec = batchRecons.min()
        maxRec = batchRecons.max()

        formatImageDataset = lambda x : x
        formatImageRec =  lambda x : x

        if (minDataset < 0 or maxDataset > 1):
            formatImageDataset = lambda image: (image - minDataset) / (maxDataset-minDataset)
            print("Dataset image not in 0-1 range. Range ({0} / {1})".format(minDataset,maxDataset))
        if (minRec < 0 or maxRec > 1):
            formatImageRec = lambda image: (image - minRec) / (maxRec-minRec)
            print("Rec image not in 0-1 range. Range ({0} / {1})".format(minRec,maxRec))


        for i in range(show):
            original = images[i].reshape(outImageShape)
            recon = batchRecons[i].reshape(outImageShape)

            plt.figure('original')
            plt.imshow(formatImageDataset(original))

            plt.figure('Reconstruction')
            plt.imshow(formatImageRec(recon))

            plt.show()


        transformName = outName
        labNames={}
        showBlokeh=True

        # Definir transformacion como activaciones en salida encoder
        transfFun = lambda x: self.getLatentRepresentation(x)

        # Tomar activaciones
        trainX, rlbs = trainsetTransform(transfFun, self.dataset)
        rlbs = OneHotToInt(rlbs)

        # Crear carpeta de resultados
        if not os.path.exists(outName):
            os.makedirs(outName)

        # Mostrar labels reales con PCA 2
        pca = PCA(n_components=2)
        transformed = showDimRed(trainX, rlbs, 'latentRep PCA_Real',
                                 pca, outName)

        kmeans = cluster.KMeans(n_clusters=nClustersTofind)

        spectral = cluster.SpectralClustering(n_clusters=nClustersTofind,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
        algorithms = [kmeans,spectral]

        for clusterAlg in algorithms:
            # Categorizar con K means o spectral
            points, predClust, realsLab = clusterLabeling(self.dataset,
                                                          transfFun, clusterAlg,
                                                          trainX)
            name = clusterAlg.__class__.__name__
            print "Showing results for Cluster ", name



            showResults(self.dataset, points, predClust, np.array(realsLab),
                        transformName + " " + 'Cluster ' + str(name), outName,
                        labelNames=labNames)




def checkSessionDecorator(func):
    def func_wrapper(*args, **kwargs):
        #args[0] should be self
        assert(args[0].activeSession),'No active session'
        return func(*args, **kwargs)

    return func_wrapper

class AutoencoderVanilla(AbstractUnsupModel):
    """
    Example of use

    with AutoencoderVanilla() as autoencoder:
        autoencoder.train()
        autoencoder.evaluate()

    The with is needed to start and end the tensorflow session

    """
    def __init__(self, datasetObject, iterations=1000, units=100, learningRate=0.01):

        super(AutoencoderVanilla, self).__init__(datasetObject)
        self.logsDir = 'autoencodersVanilla'
        self.iterations = iterations
        self.activeSession = None
        self.units=units
        self.learningRate=learningRate

    def defArch(self):
        self.inputBatch = tf.placeholder(tf.float32, (self.batchSize, self.imageShape[0], self.imageShape[1], self.imageShape[2]), 'input')
        flattenInput = tf.reshape(self.inputBatch, (self.batchSize, self.inputSizeFlatten), name='flattenInput')

        enc1 = tf.contrib.layers.fully_connected(flattenInput,
                                                             self.units*2,
                                                             activation_fn=tf.nn.relu)

        enc2 = tf.contrib.layers.fully_connected(enc1, self.units, activation_fn=tf.nn.relu)


        self.hiddenLayer = tf.contrib.layers.fully_connected(enc2, 20, activation_fn=tf.nn.relu)

        dec1 = tf.contrib.layers.fully_connected(self.hiddenLayer, self.units, activation_fn=tf.nn.relu)

        dec2 = tf.contrib.layers.fully_connected(dec1, self.units*2,activation_fn=tf.nn.relu)

        self.reconstructionLayer = tf.contrib.layers.fully_connected(dec2, self.inputSizeFlatten, activation_fn=tf.nn.tanh)

        self.reconError = tf.losses.mean_squared_error(flattenInput, self.reconstructionLayer)

        self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.reconError)

        tf.summary.scalar('ReconstructionError', self.reconError)
        pass

    @checkSessionDecorator
    def getLatentRepresentation(self,x):
        return self.activeSession.run(self.hiddenLayer, feed_dict={self.inputBatch: x})


    def __enter__(self):
        tf.reset_default_graph()
        # Initialize
        self.defArch()

        self.activeSession = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.merged = tf.summary.merge_all()


        modelPath = os.path.join('logs',self.logsDir, stringNow(), 'train')
        self.train_writer = tf.summary.FileWriter(modelPath, self.activeSession.graph)
        self.val_writer = tf.summary.FileWriter(os.path.join('logs',self.logsDir, stringNow(), 'val'), self.activeSession.graph)



        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.activeSession.close()
        pass


    @checkSessionDecorator
    def train(self):
        for i in range(self.iterations):
            if i % 30 == 0:  # Record summaries and test-set accuracy
                images, labels = self.dataset.next_batch(self.batchSize, 'val')
                summary, error = self.activeSession.run([self.merged, self.reconError], feed_dict={self.inputBatch: images})
                print('Iteration {1} reconstruction Error {0} '.format(error, i))
                self.val_writer.add_summary(summary, i)

            else:  # Record train set summaries, and train
                images, labels = self.dataset.next_batch(self.batchSize, 'train')
                summary, _ = self.activeSession.run([self.merged, self.train_step], feed_dict={self.inputBatch: images})
                self.train_writer.add_summary(summary, i)


        self.train_writer.close()
        self.val_writer.close()

    @checkSessionDecorator
    def reconstruction(self, imageBatch):
        return self.activeSession.run(self.reconstructionLayer, feed_dict={self.inputBatch: imageBatch})


class ConvAutoencoder(AutoencoderVanilla):

    def __init__(self, datasetObject, iterations=1000, units=32, learningRate=0.01):

        super(ConvAutoencoder, self).__init__(datasetObject,iterations=iterations, units=units, learningRate=learningRate)
        self.logsDir = 'ConvAutoencoder'

    @checkSessionDecorator
    def getLatentRepresentation(self,x):
        return self.activeSession.run(tf.reduce_mean(self.hiddenRep,axis=[1,2]), feed_dict={self.inputBatch: x})


    def defArch(self):
        self.inputBatch = tf.placeholder(tf.float32, (self.batchSize, self.imageShape[0], self.imageShape[1], self.imageShape[2]), 'input')

        # Encoder
        conv1 = tf.layers.conv2d(inputs=self.inputBatch,filters=self.units*2,kernel_size=[3, 3],padding="same",strides=(2,2),
                                 activation=tf.nn.relu,name='Conv1',kernel_initializer=tf.contrib.layers.xavier_initializer())

        conv2 = tf.layers.conv2d(inputs=conv1,filters=self.units,kernel_size=[4, 4],padding="same",strides=(2,2),
                                 activation=tf.nn.relu,name='Conv2',kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.outEncoder = conv2

        self.hiddenRep = tf.layers.conv2d(inputs=self.outEncoder, filters=20, kernel_size=[5, 5], padding="same",strides=(1,1),
                                 activation=tf.nn.relu, name='HidenRep',kernel_initializer=tf.contrib.layers.xavier_initializer())


        # Decoder
        convT1 = tf.layers.conv2d_transpose(self.hiddenRep, self.units, kernel_size=[4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, name='convT1',kernel_initializer=tf.contrib.layers.xavier_initializer())
        convT2 = tf.layers.conv2d_transpose(convT1, self.units*2, kernel_size=[3, 3], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, name='convT2',kernel_initializer=tf.contrib.layers.xavier_initializer())

        outDecoder = convT2


        self.reconstructionLayer = tf.layers.conv2d_transpose(outDecoder,self.inputBatch.shape[3],
                                                              kernel_size=[3, 3],strides=(1, 1),padding='same',
                                                              activation=tf.nn.tanh,kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                              name='Recons')

        self.reconError = tf.losses.mean_squared_error(self.inputBatch, self.reconstructionLayer)

        self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.reconError)

        tf.summary.scalar('ReconstructionError', self.reconError)
        pass


#CONV AE


# SPARSE AE


# VAE



