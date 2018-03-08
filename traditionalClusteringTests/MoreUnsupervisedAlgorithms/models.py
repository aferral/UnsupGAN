from infogan.misc.datasets import DataFolder
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os

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

    def reconstruction(self, imageBatch):
        raise NotImplementedError()

    def evaluate(self,show = 3):

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

        # Tomar activaciones

        # Mostrar labels reales con PCA 2

        # Categorizar con K means o spectral

        # Mostrar graficos predicted vs real

        # Calcular NMI ARI PURITY

        # Evaluar labeling


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
        hiddenLayer = tf.contrib.layers.fully_connected(flattenInput, self.units, activation_fn=tf.nn.sigmoid)
        self.reconstruction = tf.contrib.layers.fully_connected(hiddenLayer, self.inputSizeFlatten, activation_fn=tf.nn.tanh)

        self.reconError = tf.losses.mean_squared_error(flattenInput, self.reconstruction)

        self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.reconError)

        tf.summary.scalar('ReconstructionError', self.reconError)
        pass



    def __enter__(self):

        # Initialize
        self.defArch()

        self.activeSession = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(os.path.join('logs',self.logsDir, stringNow(), 'train'), self.activeSession.graph)
        self.val_writer = tf.summary.FileWriter(os.path.join('logs',self.logsDir, stringNow(), 'val'), self.activeSession.graph)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.activeSession.close()
        pass


    @checkSessionDecorator
    def train(self):
        for i in range(self.iterations):
            if i % 10 == 0:  # Record summaries and test-set accuracy
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
        return self.activeSession.run(self.reconstruction, feed_dict={self.inputBatch: imageBatch})


class ConvAutoencoder(AutoencoderVanilla):

    def __init__(self, datasetObject, iterations=1000, units=32, learningRate=0.01,oneLayer=False):

        super(ConvAutoencoder, self).__init__(datasetObject,iterations=iterations, units=units, learningRate=learningRate)
        self.logsDir = 'ConvAutoencoder'
        self.onelayer=oneLayer

    def defArch(self):
        self.inputBatch = tf.placeholder(tf.float32, (self.batchSize, self.imageShape[0], self.imageShape[1], self.imageShape[2]), 'input')

        # Encoder
        conv1 = tf.layers.conv2d(inputs=self.inputBatch,filters=self.units,kernel_size=[3, 3],padding="same",strides=(2,2),
                                 activation=tf.nn.relu,name='Conv1')
        if not self.onelayer:
            conv2 = tf.layers.conv2d(inputs=conv1, filters=self.units*2, kernel_size=[3, 3], padding="same",strides=(2,2),
                                     activation=tf.nn.relu, name='Conv2')
            outEncoder = conv2
        else:
            outEncoder = conv1

        # Decoder
        if not self.onelayer:
            convT1 = tf.layers.conv2d_transpose(outEncoder, self.units, kernel_size=[3, 3], strides=(2, 2), padding='same',
                                                activation=tf.nn.relu, name='convT1')
            outDecoder = convT1
        else:
            outDecoder = conv1

        self.reconstruction = tf.layers.conv2d_transpose(outDecoder,self.inputBatch.shape[3],kernel_size=[3, 3],strides=(2, 2),padding='same',activation=tf.nn.tanh,name='Recons')

        self.reconError = tf.losses.mean_squared_error(self.inputBatch, self.reconstruction)

        self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.reconError)

        tf.summary.scalar('ReconstructionError', self.reconError)
        pass


#CONV AE


# SPARSE AE


# VAE



