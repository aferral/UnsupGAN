import json
import os
import sys
import tensorflow as tf
from sklearn.manifold import TSNE
from scipy.io import savemat
from infogan.misc.dataset import Dataset
from infogan.misc.datasets import DataFolder
from launchers.discriminatorTest import trainsetTransform, pool_features
from sklearn.preprocessing import normalize

from traditionalClusteringTests.dataUtils import showDimRed

saveAct = True
doTSNE = True

#First get activations, then pool if needed (conv filters are reduced to 1x1 per filter doing a mean) then normalize is needed
def getActLayer(sess,layerName,d_in,norm=False):
    layer=sess.graph.get_tensor_by_name(layerName)
    if norm:
        return lambda x : normalize(pool_features(sess.run(layer, {d_in: x})), axis=1, norm='l2')
    else:
        return lambda x : pool_features((sess.run(layer, {d_in: x})))


def main(configFile,layerName,labelNames=None):
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

    with tf.Session(config=tf.GPUOptions(per_process_gpu_memory_fraction=0.05)) as sess:
        new_saver = tf.train.import_meta_graph(modelPath+'.meta')
        new_saver.restore(sess, modelPath)

        d_in = sess.graph.get_tensor_by_name(discrInputName)

        # Crear lista de capas a usar automaticamente mediante navegacion del grafo


        if not ":0" in layerName:
            layerName = layerName + ":0"


        #Define layer to use
        layerFunction = getActLayer(sess, layerName, d_in, norm=False)

        #Get layer activations of entire train set
        trainX, realLabels = trainsetTransform(layerFunction, useDataset)

        print "Procesando " + str(layerName)
        print trainX.shape

        outFolder = os.path.join("ShowAct", exp_name)

        if not os.path.exists(outFolder):
            os.makedirs(outFolder)


        if saveAct:
            #Export the activations to a .mat file save the layer name, the dataset folder and the number of points (data is a matrix N,D )
            data_dict = {'layerName': layerName, 'data': trainX, 'dataset' : useDataset.getFolderName(),'n_points' : trainX.shape[0]}
            savemat(os.path.join(outFolder,layerName.replace('/','_')+'.mat'),data_dict)

        #DO TSNE
        if doTSNE:
            model = TSNE(n_components=2)
            #Save image expName_layer_TSNE.png
            outName = layerName.replace('/','_')+ str(' TSNE_Real')
            showDimRed(trainX[0:limitPoints], realLabels[0:limitPoints], outName , model, outFolder,labelsName=labelNames)



if __name__ == '__main__':
    print sys.argv
    configFile = sys.argv[1]
    Layer = sys.argv[2]
    labelNms = None
    if len(sys.argv) < 3:
        print "Faltan parametros"
        exit(1)
    if len(sys.argv) > 3:
        labelNms = sys.argv[3:]
    if labelNms:
        for ind,elem in enumerate(labelNms):
            print "Label ",ind," ",elem

    main(configFile,Layer,labelNms)