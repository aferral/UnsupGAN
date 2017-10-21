import json
import os
import sys
import tensorflow as tf



def main(configFile):
    # -----Cargar parametros
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


    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(modelPath+'.meta')
        new_saver.restore(sess, modelPath)

        for n in tf.get_default_graph().as_graph_def().node:
            print n.name



if __name__ == '__main__':
    configFile = sys.argv[1]

    main(configFile)