from sklearn import cluster
from sklearn.decomposition import PCA
import os
import tensorflow as tf

from infogan.misc.datasets import DataFolder
from traditionalClusteringTests.MoreUnsupervisedAlgorithms.models import \
    AutoencoderVanilla, checkSessionDecorator
import numpy as np


def xavier_init(f_in,f_out,constant=1):
    """ Iniciamos los pesos de la red con Xavier"""

    low = -constant*np.sqrt(6.0/(f_in+f_out))
    high = constant*np.sqrt(6.0/(f_in+f_out))
    return tf.random_uniform((f_in,f_out),minval=low,maxval=high,dtype=tf.float32)


class ConvVAE(AutoencoderVanilla):

    def __init__(self, datasetObject, iterations=1000, units=32, learningRate=0.01,nz=100):

        super(ConvVAE, self).__init__(datasetObject,iterations=iterations, units=units, learningRate=learningRate)
        self.logsDir = 'CONV_VAE'
        self.nz = nz


    def defArch(self):
        self.inputBatch = tf.placeholder(tf.float32, (self.batchSize, self.imageShape[0], self.imageShape[1], self.imageShape[2]), 'input')

        self.scaledImages = tf.multiply(tf.add(self.inputBatch,1),0.5)

        checkMax = tf.Assert(tf.less_equal(tf.reduce_max(self.scaledImages), 1), [self.scaledImages],name='assertMaxInput')
        checkMin = tf.Assert(tf.greater_equal(tf.reduce_min(self.scaledImages), 0),[self.scaledImages],name='assertMinInput')

        # conv1 k 3 3 str 2 2 f 100 relu
        # conv2 k 3 3 str 2 2 f 50 relu
        # conv 3 k 3 3 str 2 2 f 30 relu
        # reshape 24x24x50 to flat (12*12*30 vect)
        #
        # mlp_mean,mlp_std de n_z units no act
        #
        # calcZ mult by eps
        #
        # mlp de z_units a 12*12*30
        #
        # convT1 k 3 3 str 2 2 30 relu
        # convT2 k 3 3 str 2 2 50 relu
        # convT3 k 3 3 str 2 2 100 relu
        #
        # convTf k 3 3 str 1 1 outF 1 softmax? o lineal

        n_z = self.nz

        f1=self.units
        f2=int(self.units*0.5)
        f3 = int(self.units * 0.25)


        # Encoder


        conv1 = tf.layers.conv2d(inputs=self.scaledImages,
                                 filters=f1,kernel_size=[3, 3],padding="same",strides=(2,2),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(),
                                 activation=tf.nn.relu,name='Conv1')

        conv2 = tf.layers.conv2d(inputs=conv1,
                                 filters=f2,kernel_size=[3, 3],padding="same",strides=(2,2),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(),
                                 activation=tf.nn.relu,name='Conv2')

        conv3 = tf.layers.conv2d(inputs=conv2,
                                 filters=f3,kernel_size=[3, 3],padding="same",strides=(2,2),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(),
                                 activation=tf.nn.relu,name='Conv3')

        conv3_n_filters = int(self.units*0.25)
        imageShape = self.imageShape

        flattenSize = reduce(lambda x,y : x*y,conv3._shape_as_list()[1:])

        reshapeEncoder = tf.reshape(conv3,(self.batchSize,flattenSize))

        # Latent representation

        self.z_mean = tf.layers.dense(reshapeEncoder, n_z,activation=None,use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),name='z_mean')

        self.z_log_sigma_sq = tf.layers.dense(reshapeEncoder, n_z,activation=None,use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),name='z_log_sigma_sq')

        eps = tf.random_normal((self.batchSize, n_z), 0, 1, dtype=tf.float32)
        z = tf.add(self.z_mean,tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)),eps),name='Z')

        self.hiddenLayer = z




        # Decoder

        preprocessDecoder = tf.layers.dense(z, flattenSize,activation=tf.nn.relu,use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),name='preformat_decoder')

        reshapeDecoder = tf.reshape(preprocessDecoder,conv3._shape_as_list())

        convT1 = tf.layers.conv2d_transpose(reshapeDecoder,f1,[3,3],
                     strides=(2, 2),
                     padding='same',
                     activation=tf.nn.relu,
                     use_bias=True,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),name='ConvT1')

        convT2 = tf.layers.conv2d_transpose(convT1,f2,[3,3],
                     strides=(2, 2),
                     padding='same',
                     activation=tf.nn.relu,
                     use_bias=True,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),name='ConvT2')

        convT3 = tf.layers.conv2d_transpose(convT2,f3,[3,3],
                     strides=(2, 2),
                     padding='same',
                     activation=tf.nn.relu,
                     use_bias=True,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),name='ConvT3')

        self.reconstructionLayer  = tf.layers.conv2d_transpose(convT3,self.scaledImages.shape[-1],[3,3],
                     strides=(1, 1),
                     padding='same',
                     activation=tf.nn.sigmoid,
                     use_bias=True,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),name='reconstruct')



        # Define loss function
        P = -tf.reduce_sum(self.scaledImages * tf.log(1e-10 + self.reconstructionLayer)
                       + (1-self.scaledImages) * tf.log(1e-10 + 1 - self.reconstructionLayer), [1,2,3])


        KL = -0.5 * tf.reduce_sum(1 + 2*self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(2*self.z_log_sigma_sq), 1) # KL divergence: KL[Q(z|X)||P(z)]


        self.reconError = tf.reduce_mean(P + KL)

        self.train_step = tf.group(checkMax,checkMin,tf.train.AdamOptimizer(self.learningRate).minimize(self.reconError))


        # Summaries

        tf.summary.scalar('ReconstructionError', self.reconError)


    @checkSessionDecorator
    def train(self):
        for i in range(self.iterations):
            if i % 300 == 0:  # Record summaries and test-set accuracy
                images, labels = self.dataset.next_batch(self.batchSize, 'val')
                summary, error = self.activeSession.run([self.merged, self.reconError], feed_dict={self.inputBatch: images})
                print('Iteration {1} reconstruction Error {0} '.format(error, i))
                self.val_writer.add_summary(summary, i)

                # Save a image sample
                folderImages = os.path.join('logs', self.logsDir,'sampleTrain')

                import matplotlib.pyplot as plt
                res = self.reconstructionLayer.eval({self.inputBatch: images})
                plt.imshow(res[0].reshape(96, 96))
                plt.savefig(os.path.join(folderImages,'image_{0}.png'.format(i)))

            else:  # Record train set summaries, and train
                images, labels = self.dataset.next_batch(self.batchSize, 'train')
                summary, _ = self.activeSession.run([self.merged, self.train_step], feed_dict={self.inputBatch: images})
                self.train_writer.add_summary(summary, i)


        self.train_writer.close()
        self.val_writer.close()

    @checkSessionDecorator
    def getLatentRepresentation(self,x):
        assert(x.min() >= -1),"Images should be in -1 +1 Range. Actual {0}".format(x.min())
        assert (x.max() <= 1), "Images should be in -1 +1 Range. Actual {0}".format(x.max())
        """
        Get the Z vector sampled from a multivariable gaussian given by the Mean_vect(x) , std_vect(x) distribution 
        :param x: batch of images Should be -1 +1 Range [batch, imagH,imgW,channels]
        :return: Z vector
        """
        return self.activeSession.run(self.hiddenLayer, feed_dict={self.inputBatch: x})

    @checkSessionDecorator
    def reconstruction(self, imageBatch):
        assert(imageBatch.min() >= -1),"Images should be in -1 +1 Range. Actual {0}".format(imageBatch.min())
        assert (imageBatch.max() <= 1), "Images should be in -1 +1 Range. Actual {0}".format(imageBatch.max())
        """
        Returns the decoder
        :param imageBatch: Should be -1 +1 Range. [batch, imagH,imgW,channels]
        :return: A batch of the reconstruction given the decoder [batch, imagH,imgW,channels]
        """
        return self.activeSession.run(self.reconstructionLayer, feed_dict={self.inputBatch: imageBatch})



if __name__ =='__main__':
    batchSize = 20
    dataset = DataFolder("data/MFPT96Scalograms", batchSize, testProp=0.3,
                         validation_proportion=0.3, out_size=(96, 96, 1))

    with ConvVAE(dataset, iterations=10000, units=100,
                            learningRate=0.001) as vae:
        vae.train()
        vae.evaluate('VAE_conv')
