import json
import os
import sys
import tensorflow as tf
from sklearn.manifold import TSNE

from infogan.misc.dataset import Dataset
from infogan.misc.datasets import DataFolder
from launchers.discriminatorTest import trainsetTransform, pool_features
from sklearn.preprocessing import normalize

from traditionalClusteringTests.dataUtils import showDimRed


#First get activations, then pool if needed (conv filters are reduced to 1x1 per filter doing a mean) then normalize is needed
def getActLayer(sess,layerName,d_in,norm=False):
    layer=sess.graph.get_tensor_by_name(layerName)
    if norm:
        return lambda x : normalize(pool_features(sess.run(layer, {d_in: x})), axis=1, norm='l2')
    else:
        return lambda x : pool_features((sess.run(layer, {d_in: x})))


def main(configFile,labelNames=None):
    # -----Cargar parametros
    print "Loading config file ", configFile

    limitPoints = 2000

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


    useDataset = DataFolder(dataFolder, batchSize, testProp=0.6, validation_proportion=0.5, out_size=imageSize)


    outFolder = os.path.join("ShowAct", exp_name)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath+'.meta')
        new_saver.restore(sess, modelPath)

        d_in = sess.graph.get_tensor_by_name(discrInputName)

        # Crear lista de capas a usar automaticamente mediante navegacion del grafo
        capas = []

        current = d_in.op.outputs[0]
        it = 100
        while not ('OutDiscriminator:0' in current.name):
            nextNames = [elem.name for elem in current.consumers()]
            if any([('batchnorm/mul_1' in name) for name in nextNames]):
                for ind, way in enumerate(current.consumers()):
                    if 'mul_1' in way.name:
                        current = current.consumers()[ind].outputs[0]
            else:
                current = current.consumers()[0].outputs[0]
            capas.append(current.name)
            it = it - 1
            if it == 0:
                raise Exception("Error en busqueda de grafo entro a ciclo maxima profundidad 100")

        #Add FC endoder (las FC before encoder)
        if (res['doEncoderLabel']) and (not (res['discrEncoderName'] is None)):
            finalEncoder = sess.graph.get_tensor_by_name(res['discrEncoderName'])
            lastFcEncoder = finalEncoder.op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0]
            capas.append(lastFcEncoder.name)

        #Filter capas like resize
        capas = filter(lambda x : not "Reshape" in x,capas)

        #Do the TSNE of the raw image
        layerFunction = lambda x : sess.run(d_in, {d_in: x}).reshape(batchSize,-1)
        trainX, realLabels = trainsetTransform(layerFunction, useDataset)
        model = TSNE(n_components=2)
        outName = "ImageRaw"+ str(' TSNE_Real')
        showDimRed(trainX[0:limitPoints], realLabels[0:limitPoints], outName, model, outFolder,labelsName=labelNames)

        for layerName in capas:
            if not ":0" in layerName:
                layerName = layerName + ":0"
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
            print trainX.shape
            outName = layerName.replace('/','_')+ str(' TSNE_Real')
            showDimRed(trainX[0:limitPoints], realLabels[0:limitPoints], outName , model, outFolder,labelsName=labelNames)



if __name__ == '__main__':
    configFile = sys.argv[1]
    labelNms = None
    if len(sys.argv) > 2:
        labelNms = sys.argv[2:]
    for ind,elem in enumerate(labelNms):
        print "Label ",ind," ",elem

    main(configFile,labelNms)