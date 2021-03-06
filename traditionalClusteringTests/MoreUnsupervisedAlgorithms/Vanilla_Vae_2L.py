
import tensorflow as tf
import numpy as np
import os
import time

from skimage import exposure

from infogan.misc.datasets import DataFolder, MnistDataset
from traditionalClusteringTests.MoreUnsupervisedAlgorithms.models import AbstractUnsupModel

# TODO probar con redes de varios elementos pocas epocas
# TODO probar con tanh

def xavier_init(f_in,f_out,constant=1):
    """ Iniciamos los pesos de la red con Xavier"""

    low = -constant*np.sqrt(6.0/(f_in+f_out))
    high = constant*np.sqrt(6.0/(f_in+f_out))
    return tf.random_uniform((f_in,f_out),minval=low,maxval=high,dtype=tf.float32)



class Vanilla_VAE(AbstractUnsupModel):
    # initialization Method

    def __init__(self,dataset, network_architecture, vae_parameters, transf_function=tf.nn.softplus):

        super(Vanilla_VAE, self).__init__(dataset)

        self.activeSession = None


        #Unpack some Parameters

        self.network_architecture = network_architecture

        self.transfer_function = transf_function

        self.batch_size = vae_parameters['batch_size']

        self.dropout_encoder = vae_parameters['dropout_encoder']

        self.dropout_decoder = vae_parameters['dropout_decoder']

        #We define the input as a TF placeholder of [any number of dimention, n_input]
        self.input=tf.placeholder(tf.float32, [None,self.imageShape[0],self.imageShape[1],self.imageShape[2]],name='imageInput')

        self.x = tf.reshape(self.input, (-1,self.network_architecture['n_input']),name='flattenInput')


        self.dp_prob_encoder = tf.placeholder(tf.float32,name='dropout_prob_Encoder')
        self.dp_prob_decoder = tf.placeholder(tf.float32,name='dropout_prob_Decoder')
        #PlaceHolder for the learning rate

        self.learning_rate=tf.placeholder(tf.float32)

        #Call of create_network method.
        self._create_network()

        #Call of create_loss_optimizer method.
        self._create_loss_optimizer()

        #Initiate the tensorflow variables
        self.init = tf.global_variables_initializer()







    def _create_network(self):

        #Method that creates the network and defines the means and deviation of the latent and output space
        #in form of TF tensors ready to evaluate.

        #First, we initializate the weights and bias with the initialize_weights method via Xavier function.

        network_weights = self._initialize_weights(**self.network_architecture)



        #Then, we define the mean and deviation of the latent space as the output of the encoder method.
        self.z_mean, self.z_log_sigma_sq = self._encoder(network_weights["weights_encoder"],
                                            network_weights["biases_encoder"])


        # We extract from the network_architecture dictionary the latent space dimention.
        n_z = self.network_architecture["n_z"]

        #Reparametrization trick: sample epsilon from MVN(0,1)
        eps = tf.random_normal((self.batch_size,n_z),0,1,dtype=tf.float32)

        # And then define z = mu+sigma*epsilon
        self.z = tf.add(self.z_mean,tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)),eps),name='Z')


        #Finally, we define the mean of the output space as the output of the decoder.

        self.x_reconstr_mean = self._decoder(network_weights["weights_decoder"],
                                 network_weights["biases_decoder"])





    def _initialize_weights(self,n_hidden_encoder_1,n_hidden_encoder_2,
                           n_hidden_decoder_1,n_hidden_decoder_2,n_input,n_z):

        #Method that create a dictionary with all the weights and bias of the encoder and decoder's NN.
        all_weights = dict()

        # tf.random_uniform((-0.5, 0.5), minval=low, maxval=high,dtype=tf.float32)

        #Encoder Weights
        all_weights['weights_encoder'] = {
            'h1' : tf.Variable(xavier_init(n_input,n_hidden_encoder_1)),
            'h2' : tf.Variable(xavier_init(n_hidden_encoder_1,n_hidden_encoder_2)),

            # 'out_mean' : tf.Variable(xavier_init(n_hidden_encoder_2,n_z)),
            # 'out_log_sigma' : tf.Variable(xavier_init(n_hidden_encoder_2,n_z))}
            'out_mean': tf.Variable(xavier_init(n_hidden_encoder_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_encoder_2, n_z))}
            # 'out_mean': tf.Variable(tf.zeros([n_hidden_encoder_2,n_z], dtype=tf.float32)),
            # 'out_log_sigma': tf.Variable(tf.zeros([n_hidden_encoder_2,n_z], dtype=tf.float32))}
        #Encoder Bias
        all_weights['biases_encoder'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_encoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_encoder_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        #Decoder Weights
        all_weights['weights_decoder'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_decoder_1)),
            'h2': tf.Variable(xavier_init(n_hidden_decoder_1, n_hidden_decoder_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_decoder_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_decoder_2, n_input))}
        #Decoder Bias
        all_weights['biases_decoder'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_decoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_decoder_2], dtype=tf.float32)),
            'out_mean': 0.7*tf.Variable(tf.ones([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.ones([n_input], dtype=tf.float32))}
            # 'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            # 'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}

        return all_weights



    def _encoder(self, weights, biases):

        #Implementation of the encoder NN as a two layer neural network.
        layer_1 = self.transfer_function(tf.add(tf.matmul(self.x, weights['h1']),biases['b1']))
        layer_2 = self.transfer_function(tf.add(tf.matmul(layer_1, weights['h2']),biases['b2']))
        #We applied dropout
        layer_2_dp = tf.nn.dropout(layer_2,self.dp_prob_encoder)
        #Then as output we have a mean and a covariance vector
        z_mean = tf.add(tf.matmul(layer_2_dp, weights['out_mean']),biases['out_mean'],name='z_mean')
        z_log_sigma_sq = tf.add(tf.matmul(layer_2_dp, weights['out_log_sigma']),biases['out_log_sigma'],name='z_log_sigma_sq')

        return (z_mean,z_log_sigma_sq)



    def _decoder(self, weights, biases):

        #Implementation of the decoder NN as a two layer neural network.
        layer_1 = self.transfer_function(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.transfer_function(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        #We applied dropout
        layer_2_dp = tf.nn.dropout(layer_2,self.dp_prob_decoder)
        #Then as output we have a mean reconstruction vector
        #Sigmoid so that the bernoulli parameter is between 0 and 1.
        self.logits = tf.add(tf.matmul(layer_2_dp, weights['out_mean']), biases['out_mean'],'logitsReconstruction')
        x_reconstr_mean = tf.nn.sigmoid(self.logits,name='Reconstruction')

        return x_reconstr_mean



    def _create_loss_optimizer(self):

        #Method that creates the VAE loss and optimizer
        # E_Q(logP(X|z))

        # P = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.x), 1)
        #
        P = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                       + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)

        # KL divergence: KL[Q(z|X)||P(z)]
        KL = 0.1*-0.5 * tf.reduce_sum(1 + 2*self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(2*self.z_log_sigma_sq), 1)

        #N batch cost.
        self.cost = tf.reduce_mean(P+KL)

        #ADAM as optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


        #Some metrics for evaluation
        error_matrix = tf.reduce_sum((tf.square(self.x_reconstr_mean-self.x)))
        self.recon_error= error_matrix


    def train(self,n_epochs=10,learning_rate=0.001,display_step=5,transformRange=False):
        self.train_transform(n_epochs=n_epochs,learning_rate=learning_rate,display_step=display_step,transformRange=transformRange)

    def reconstruction(self, imageBatch):

        '''
        TRANSFORMATION STEP
        '''
        # First set of data
        return self.activeSession.run(self.x_reconstr_mean, feed_dict={self.input: imageBatch,
                                                              self.dp_prob_encoder: 1,self.dp_prob_decoder: 1})

    def getLatentRepresentation(self,x):
        return self.activeSession.run(self.z, feed_dict={self.input: x,
                                                              self.dp_prob_encoder: 1,self.dp_prob_decoder: 1})

    def __enter__(self):
        self.activeSession = tf.Session()
        #Initializate variables
        self.activeSession.run(self.init)

        self.train_writer = tf.summary.FileWriter(os.path.join('logs', 'VAEtest', 'train'),
                                                  self.activeSession.graph)



        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.activeSession.close()
        self.train_writer.close()

    def train_transform(self,n_epochs=10,learning_rate=0.001,display_step=5,transformRange=False):

        data_train = self.dataset

        #Define a list to append all the errors
        E=[]
        '''
        TRAIN STEP
        '''
        print()
        print ("Training of the VAE \n")
        init_time = time.time()
        #Number of data points
        n_samples = data_train.get_n_training_points()
        print("N de samples : {0} ".format(n_samples))
        #Number of iterations per epoch
        total_batch = int(n_samples*1.0 / self.batch_size)


        # Training cycle
        for epoch in range(n_epochs):

            # Loop over all batches
            for i in range(total_batch):

                batch_images,labels = self.dataset.next_batch(self.batchSize)

                # Batch images in -1 +1 range ? transform to 0 1 range
                if transformRange:
                    batch_images = (batch_images + 1) * 0.5
                    #batch_images = (batch_images / batch_images.sum(axis=(1,2)).reshape(-1,1,1,1))
                    # for i in range(batch_images.shape[0]):
                    #     batch_images[i] = exposure.equalize_adapthist(batch_images[i].reshape(96, 96),clip_limit=0.03).reshape(1,96,96,1)


                # Fit training using batch data, and calculate mean error of reconstruction
                opt,cost,error = self.activeSession.run((self.optimizer, self.cost,self.recon_error),
                                feed_dict={self.input: batch_images , self.learning_rate:learning_rate,
                                        self.dp_prob_encoder: self.dropout_encoder,
                                        self.dp_prob_decoder: self.dropout_decoder})

                # Compute average loss
                avg_cost = cost / self.batch_size
                E.append(error)



            # Display logs per epoch step
            if epoch % display_step == 0:

                print(( "Epoch: {0}  Cost: {1} MSE: {2} ".format((epoch+1), avg_cost, error)))
                pathFolder = os.path.join('VAE','model')
                if not os.path.exists(pathFolder):
                    os.makedirs(pathFolder)
                # self.evaluate(show=1)

        total_time = time.time()-init_time
        print()
        print(('Training ready!, Total time: ', total_time))
        print()

        # Loop over all batches
        for i in range(5):
            import matplotlib.pyplot as plt
            batch_images, labels = self.dataset.next_batch(self.batchSize)
            batch_images = (batch_images + 1) * 0.5
            r = self.activeSession.run(self.x_reconstr_mean,
                                       feed_dict={self.input: batch_images,
                                                  self.dp_prob_encoder: 1,
                                                  self.dp_prob_decoder: 1});
            plt.imshow(r[0].reshape(96, 96))

            plt.figure();

            plt.imshow(batch_images[0].reshape(96, 96));
            plt.show()

def main():
    batchSize = 30

    # dataset = DataFolder("cifar10", batchSize, testProp=0.3, validation_proportion=0.3, out_size=(32, 32, 1))
    # dataset = MnistDataset(outShape=(-1,28,28,1),batchSize=batchSize)
    dataset = DataFolder("data/MFPT96Scalograms", batchSize, testProp=0.3,validation_proportion=0.3, out_size=(96, 96, 1))
    transformRange = True # is the dataset in range -1 +1 ?


    print(dataset.shape)
    assert(dataset.shape[0] == dataset.shape[1]),'Dataset image shape must be square'

    # TRAINING OF THE VAE AND TRANSFORMATION OF DATASET
    # Define the Architecture for the VAE
    L1 = 200
    L2 = 100
    n_z = 1000
    epochs = 200
    learningRate = 0.0001
    displayAfter = 1

    flattenSize = reduce(lambda x,y : x*y,dataset.getImshape())


    network_architecture_vae = dict(n_hidden_encoder_1=500,  # 1st layer encoder neurons
                                    n_hidden_encoder_2=300,  # 2st layer encoder neurons
                                    n_hidden_decoder_1=300,  # 1st layer decoder neurons
                                    n_hidden_decoder_2=500,  # 2st layer decoder neurons
                                    n_input=flattenSize,  # data input (img shape)
                                    n_z=n_z)  # dimensionality of latent space
    # Define the parameters for the VAE
    vae_parameters = dict(batch_size=batchSize,
                          dropout_encoder=0.3,
                          dropout_decoder=0.3)

    with Vanilla_VAE(dataset,network_architecture_vae, vae_parameters, transf_function=tf.nn.relu) as vae:
        vae.train(n_epochs=epochs,learning_rate=learningRate,display_step=displayAfter,transformRange=transformRange)
        vae.evaluate('VAE')

if __name__ == '__main__':
    main()
