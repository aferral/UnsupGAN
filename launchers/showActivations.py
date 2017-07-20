import json
import os
import sys
import tensorflow as tf
from sklearn.manifold import TSNE

from infogan.misc.dataset import Dataset
from infogan.misc.datasets import DataFolder
from launchers.discriminatorTest import trainsetTransform
from sklearn.preprocessing import normalize

from traditionalClusteringTests.dataUtils import showDimRed


def getActLayer(sess,layerName,d_in,norm=False):
    layer=sess.graph.get_tensor_by_name(layerName)
    if norm:
        return lambda x : normalize(sess.run(layer, {d_in: x}), axis=1, norm='l2')
    else:
        return lambda x : (sess.run(layer, {d_in: x}))


def main(configFile):
    # -----Cargar parametros
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

    batchSize = res['batch_size']
    imageSize = res['imageSize']


    discrInputName = res['discrInputName']


    # Crear lista de capas a usar
    capas = ['d_h3_conv_2/d_h3_conv/BiasAdd']

    useDataset = DataFolder(dataFolder, batchSize, testProp=0.9, validation_proportion=0.5, out_size=imageSize)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath+'.meta')
        new_saver.restore(sess, modelPath)

        d_in = sess.graph.get_tensor_by_name(discrInputName)

        for layerName in capas:

            #Define layer to use
            layerFunction = getActLayer(sess, layerName, d_in, norm=False)

            #Get layer activations of entire train set
            trainX, realLabels = trainsetTransform(layerFunction, useDataset)

            #DO TSNE
            model = TSNE(n_components=2)
            #Save image expName_layer_TSNE.png
            outFolder = os.path.join("ShowAct",exp_name)
            if not os.path.exists(outFolder):
                os.makedirs(outFolder)
            showDimRed(trainX[0:10], realLabels[0:10], layerName + str(' TSNE_Real'), model, outFolder)



if __name__ == '__main__':
    configFile = sys.argv[1]
    main(configFile)