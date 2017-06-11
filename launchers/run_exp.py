from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, MeanBernoulli

import tensorflow as tf
import os
import infogan.misc.datasets as datasets
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime
import sys
import json
#Load exp Json




def train(configPath):
    with open(configPath, 'r') as f:
        d = json.load(f)

        train_dataset = d['train_dataset']
        train_Folder = d['train_Folder']
        val_dataset = d['val_dataset']
        output_size = d['output_size']
        categories = d['categories']
        batch_size = d['batch_size']
        exp_name = d['exp_name']
        arch = d['arch']

        if d.has_key('semiSup'):
            semiSup = d['semiSup']
        else:
            semiSup = False
        if d.has_key('trainSplit'):
            trainSplit = d['trainSplit']
        else:
            trainSplit = 0.7

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/" + train_dataset
    root_checkpoint_dir = "ckt/" + train_dataset
    root_samples_dir = "samples/" + train_dataset
    updates_per_epoch = 100
    max_epoch = 50

    if exp_name is None:
        exp_name = "t-%s_v-%s_o-%d" % (train_dataset, val_dataset,output_size)
        if not (categories is None):
            exp_name = exp_name + "_c-%d" % (categories)
        exp_name = exp_name + "_%s" % (timestamp)

    print("Experiment Name: %s" % (exp_name))

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)
    samples_dir = os.path.join(root_samples_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(samples_dir)

    output_dist = None
    network_type = arch
    if train_dataset == "mnist":
        print("Creating train dataset ")
        dataset = datasets.MnistDataset()
        output_dist = MeanBernoulli(dataset.image_dim)
        print("CREATED train dataset ")
        network_type = 'mnist'

        print("Creating VAL dataset ")
        val_dataset = dataset
    elif train_dataset == "dataFolder":
        dataset = datasets.DataFolder(train_Folder,batch_size,out_size=output_size,validation_proportion=(1-trainSplit) )

        print("Folder datasets created ")
        val_dataset = dataset
    else:
        dataset = datasets.Dataset(name=train_dataset,
                                   batch_size=batch_size,
                                   output_size=output_size)
        val_dataset = datasets.Dataset(name= val_dataset,
                                       batch_size=batch_size,
                                       output_size= output_size)

    latent_spec = [
        (Uniform(100), False)
    ]
    if categories is not None:
        latent_spec.append((Categorical(categories), True))

    is_reg = False
    for x, y in latent_spec:
        if y:
            is_reg = True

    model = RegularizedGAN(
        output_dist=output_dist,
        latent_spec=latent_spec,
        is_reg=is_reg,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type=network_type,
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        isTrain=True,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        samples_dir=samples_dir,
        max_epoch=max_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=2e-3,
        discriminator_learning_rate=2e-3,
        semiSup=semiSup
    )

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    #config=tf.ConfigProto(gpu_options=gpu_options)
    algo.init_opt()
    with tf.Session() as sess:
        algo.train(sess)


if __name__ == "__main__":
    configPath = ''
    if len(sys.argv) < 2:
        print("Please specify path of config file")
    else:
        configPath = sys.argv[1]
        train(configPath)
