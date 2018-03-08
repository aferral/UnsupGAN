from infogan.misc.datasets import DataFolder
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os

from traditionalClusteringTests.MoreUnsupervisedAlgorithms.models import AutoencoderVanilla, ConvAutoencoder


def stringNow():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d-%H_%M_%S')

# definir datasets

batchSize = 20
imageShape = (32,32,3)

dataset = DataFolder("cifar10",batchSize,testProp=0.3, validation_proportion=0.3,out_size=imageShape)



# definir autoencoder o VAE


# Autoencoder 1 hidden layer X units
logsDir = 'logs/autoencoders'
learningRate = 0.01
units = 100
epochsTrain = 1000
inputSizeFlatten = imageShape[0] * imageShape[1] * imageShape[2]

#
# with AutoencoderVanilla(dataset,iterations=100,units=100,learningRate=0.01) as autoencoder:
#     autoencoder.train()
#     autoencoder.evaluate()


with ConvAutoencoder(dataset,iterations=100,units=50,learningRate=0.01,oneLayer=False) as autoencoder:
    autoencoder.train()
    autoencoder.evaluate()



# Definir modelo a cargar

# Tomar activaciones

# Mostrar labels reales con PCA 2

# Categorizar con K means o spectral

# Mostrar graficos predicted vs real

# Calcular NMI ARI PURITY

# Evaluar labeling
