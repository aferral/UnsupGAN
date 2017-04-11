import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import sys
from infogan.misc.utils import get_image
from joblib import Parallel, delayed
import multiprocessing


class Dataset(object):
    def __init__(self, name=None, data_root=None, list_file=None, isVal=False,
                 batch_size=64, is_crop=True, is_grayscale=False,
                 output_size=64, images=None, labels=None):

        if name is None:
            print "Need to provide a dataset name"
            sys.exit(1)

        self.supported_datasets = ['celebA', 'imagenet', 'cifar', 'cifar100', 'stl10', "mnist"]
        if name not in self.supported_datasets:
            print "Dataset not supported"
            return NotImplementedError

        self.name = name
        self.batch_size = batch_size
        if data_root is not None:
            self.data_root = data_root
        else:
            self.data_root = './data/' + name

        self.isVal = isVal
        if self.name in ['cifar', 'cifar100', 'stl10']:
            self.isVal = True

        if self.isVal is True:
            keys = ['train', 'val']
        else:
            keys = ['train']

        self.list_file = dict.fromkeys(keys)
        self.image_list = {key: list() for key in keys}
        self.labels = {key: list() for key in keys}
        self.batch_idx = dict.fromkeys(keys)
        self.counter = dict.fromkeys(keys)

        if list_file is None:
            for key in self.list_file.keys():
                if self.name == 'cifar100':
                    suffix = '_coarse'
                else:
                    suffix = ''
                self.list_file[key] = os.path.join(self.data_root, key + suffix + '_shuffle.txt')
        else:
            self.list_file = list_file

        self.is_crop = is_crop
        self.output_size = output_size
        self.is_grayscale = is_grayscale
        self._images = images
        if self.is_grayscale:
            self.image_shape = (self.output_size, self.output_size, None)
        else:
            self.image_shape = (self.output_size, self.output_size, 3)

        if images is not None:
            self._images = images.reshape(images.shape[0], -1)
            self._labels = labels
            self._epochs_completed = -1
            self._num_examples = images.shape[0]
            # shuffle on first run
            self._index_in_epoch = self._num_examples

        for split in self.list_file.keys():
            if not os.path.exists(self.list_file[split]):
                print("List of %s set not found." % (split))
                sys.exit(1)
            with open(self.list_file[split], 'r') as ff:
                for line in ff.readlines():
                    if 'txt' not in line:
                        split_line = line.strip().split(' ')
                        self.image_list[split].append(split_line[0])  # path to the image
                        if len(split_line) > 1:
                            self.labels[split].append(int(split_line[1]))  # real label, if present
                        else:
                            self.labels[split].append(None)  # Junk label (None)

                self.batch_idx[split] = len(self.image_list[split]) // self.batch_size
                self.counter[split] = 0

        self.n_labels = len(set(self.labels['train']))

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, split="train"):
        self.batch_size = batch_size
        if self.images:
            """Return the next `batch_size` examples from this data set."""
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                if self._labels is not None:
                    self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            if self._labels is None:
                return self._images[start:end], None
            else:
                return self._images[start:end], self._labels[start:end]

        if self.list_file[split]:
            idx = self.counter[split]
            start_idx = idx*self.batch_size
            end_idx = (idx+1)*self.batch_size
            if end_idx > len(self.image_list[split]):
                extra = end_idx - len(self.image_list[split])
                end_idx = -1

            self.batch_files = self.image_list[split][start_idx:end_idx]
            batch_labels = self.labels[split][start_idx:end_idx]
            if end_idx == -1:
                rand_idx = np.random.randint(low=0, high=len(self.image_list[split]), size=extra)
                extra_files = [self.image_list[split][rand_idx[ii]] for ii in range(extra)]
                extra_labels = [self.labels[split][rand_idx[ii]] for ii in range(extra)]
                self.batch_files = self.batch_files + extra_files
                batch_labels = batch_labels + extra_labels
                assert len(self.batch_files) == self.batch_size

            with Parallel(n_jobs=1) as parallel:
                batch = parallel(delayed(get_image)
                                 (os.path.join(self.data_root, batch_file), is_crop=self.is_crop, resize_w=self.output_size)
                                 for batch_file in self.batch_files)

            self.counter[split] = (self.counter[split]+1) % self.batch_idx[split]

            if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)
            return batch_images, batch_labels

class DataFolder(object): #ALL THIS IMAGES ARE GRAYSCALE
    def __init__(self,folderName,batch_size):
        import random
        seed = int(100*random.random())
        self.dataObj = Dataset(folderName,batch_size=batch_size,seed=seed)

        self.name = folderName #TODO bypass if of chech name
        self.batch_idx = dict.fromkeys(['train','val'])


        self.batch_idx['train'] = self.dataObj.n_batches
        self.batch_idx['val'] = 1 #this is because we give the entire val dataset in 1 batch


        w = dataObj.imageSize

        self.image_dim = w * w
        self.image_shape = (w, w, 1)

        


        self.n_labels = self.dataObj.classes

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    def next_batch(self, batch_size, split="train"):
        if split == 'train':
            return self.dataObj.nextBatch()
        elif split == 'val': #TODO USE BATCH IN VAL ???
            return self.dataObj.getValidationSet()
class MnistDataset(object):
    def __init__(self):
        self.name = "mnist"
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        # self.supervised_train = Dataset(name=self.name,
        #     images=np.asarray(sup_images),
        #     labels=np.asarray(sup_labels),
        # )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

        self.batch_idx = dict.fromkeys(['train','val'])


        batch_size = 64
        self.batch_idx['train'] = (self.train.images.shape[0]) // batch_size
        self.batch_idx['val'] = (self.validation.images.shape[0]) // batch_size

        self.n_labels = len(set(sup_labels)) #TODO ESTAN EN ONE HOT ???
        output_size = 64

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    def next_batch(self, batch_size, split="train"):
        if split == 'train':
            return self.train.next_batch(batch_size)
        elif split == 'val':
            return self.validation.next_batch(batch_size)
