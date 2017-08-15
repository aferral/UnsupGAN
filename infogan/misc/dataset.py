import os
from collections import Counter
import scipy.io as sio
import numpy as np
import time
from skimage.color import rgb2gray
import skimage.io as io
from sklearn.model_selection import train_test_split
import cifar10

def grayscaleEq(rgbimage):
    cof = 1.0/3
    return rgbimage[:,:,0] * cof + rgbimage[:,:,1] * cof + rgbimage[:,:,2] * cof
def labels_to_one_hot(labels,n):
    ''' Converts list of integers to numpy 2D array with one-hot encoding'''
    N = len(labels)
    one_hot_labels = np.zeros([N, n], dtype=int)
    one_hot_labels[np.arange(N), labels] = 1
    return one_hot_labels


class DataDictionary:
    def __init__(self):
        pass
    @staticmethod
    def getDataset(datafolder,**others):
        if datafolder == 'cifar10': #TODO maybe a better way to do this?
            return cifar10Dataset(**others)
        else:
            return Dataset(datafolder,**others)

#Todo crear indice y dar batch en demand nomas
class Dataset:
    def __init__(self,dataFolder, batch_size=100,testProp=0.3, validation_proportion=0.3,seed = 1, normalize=True):
        # Training set 181 datos
        # 30 Clase 1 Train, 30 Clase 1 Test. 60 Clase 0 Train, 60 clase 0 Test
        #140      41
        self.classes = 0
        self.fileNames = []
        self.dataShape = None
        labels = []
        data = []

        #THIS MAKE THAT THE TRAIN VAL TEST REMAIN THE SAME IF YOU RE RUN (random number generator seed)
        np.random.seed(seed=seed)


        #Here open files in folder somehow
        all, allL, indAll = self.readMatrixData(dataFolder)
	

        #split trainVal -  test
        tvdata,test_data,tv_labs,test_labels,indTrainVal,indTest = train_test_split(all,allL,indAll, test_size=testProp,random_state=seed)

        #separar trainVal en val - train
        assert validation_proportion > 0. and validation_proportion < 1.
        tdata, vdata, tlabels, vlabels,indTrain,indVal = train_test_split(tvdata, tv_labs,indTrainVal, test_size=validation_proportion, random_state=seed)

        self.train_data = tdata
        self.train_labels = labels_to_one_hot(tlabels,self.classes)
        self.validation_data = vdata
        self.validation_labels = labels_to_one_hot(vlabels,self.classes)
        self.test_data = test_data
        self.test_labels = labels_to_one_hot(test_labels,self.classes)

        self.trainInd = indTrain
        self.valInd = indVal
        self.testInd = indTest


        # Normalize data
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis=0)

        if normalize:
            self.train_data = np.array([   (im-self.mean)/self.std for im in self.train_data])
            self.validation_data = np.array([   (im-self.mean)/self.std for im in self.validation_data])
            self.test_data =  np.array([   (im-self.mean)/self.std for im in self.test_data])

            #If we find image with no variation std = 0 and the data will have mean / 0 = nan. Replace nan with zero
            zeroStdTrain = np.where(self.std < 1e-10)  #I tried with the scipy zscore but the error in cords [0,73] keep happening (-1 const in Z score of constant value)

            self.train_data[:, zeroStdTrain] = 0
            self.train_data = np.nan_to_num(self.train_data)
            self.validation_data = np.nan_to_num(self.validation_data)
            self.test_data = np.nan_to_num(self.test_data)

        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels) // self.batch_size
        print "There are ",len(self.train_labels),' train points ',' so ',self.n_batches,' batches'
        print 'There are ',len(self.validation_labels),' val points '
        print 'There are ',len(self.test_labels),' test points'
        self.current_batch = 0
        self.current_epoch = 0

    def getNumberOfBatches(self):
        return self.n_batches
    def getDataShape(self):
        return self.dataShape
    def classDistribution(self):
        return "Train set "+str(Counter(np.where(self.train_labels == 1)[1]))+\
               " test set "+str(Counter(np.where(self.test_labels == 1)[1]))

    def readMatrixData(self,dataFolder):
        all = []
        allL = []
        fileList = enumerate(os.listdir(dataFolder))

        supImage = ["jpg", 'png']
        # Read all the images and labels
        for ind, f in fileList:
            if (f.split('.')[-1] in supImage) and (f.split('.')[0] != ''):
                image = io.imread(os.path.join(dataFolder, f))
                if len(image.shape) == 3 and image.shape[2] == 3:
                    ray_image = grayscaleEq(image)
                elif len(image.shape) == 2:
                    ray_image = image
                else:
                    raise Exception("That is neighter an RGB image or a Grayscale Image ",image.shape)
                self.dataShape = ray_image.shape
                if len(self.dataShape) == 2:
                    self.dataShape = (self.dataShape[0],self.dataShape[1],1)
                assert(len(self.dataShape) == 3)
                label = int(f.split("_")[-1].split('.')[0])
                all.append(ray_image)
                allL.append(label)
                self.fileNames.append(f)
        
        all = np.array(all)
        allL = np.array(allL)
        indAll = np.arange(all.shape[0])  # Get index to data to asociate filenames with data
        self.classes = len(set(allL)) 
        return all, allL, indAll

    def getTrainFilename(self,trainIndex):
        return self.fileNames[self.trainInd[trainIndex]]

    def getValFilename(self,valIndex):
        return self.fileNames[self.valInd[valIndex]]

    def getTestFilename(self,testIndex):
        return self.fileNames[self.testInd[testIndex]]

    def getNclasses(self):
        return self.classes

    def normalizeImage(self,image):
        val = (image-self.mean)/self.std
        return np.nan_to_num(val)

    def nextBatch(self):
        ''' Returns a tuple with batch and batch index '''
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_data = self.train_data[start_idx:end_idx].reshape((self.batch_size,) + self.dataShape)
        batch_labels = self.train_labels[start_idx:end_idx]
        batch_idx = self.current_batch

        # Update self.current_batch and self.current_epoch
        self.current_batch = (self.current_batch + 1) % self.n_batches
        if self.current_batch != batch_idx + 1:
            self.current_epoch += 1

        return ((batch_data, batch_labels), batch_idx)

    def getEpoch(self):
        return self.current_epoch


    def getSample(self,data,labels, asBatches = False):
        if asBatches:
            batches = []
            for i in range(max(1,len(data) // self.batch_size)):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                nElem = min(self.batch_size,data.shape[0])
                batch_data = data[start_idx:end_idx].reshape((nElem,) + self.dataShape)
                batch_labels = labels[start_idx:end_idx]
                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (data.reshape((data.shape[0],) + self.dataShape), labels)

    def getValidationSet(self, asBatches=False):
        return self.getSample(self.validation_data,self.validation_labels,asBatches)
    def getTestSet(self, asBatches=False):
        return self.getSample(self.test_data, self.test_labels, asBatches)


    def reset(self):
        self.current_batch = 0
        self.current_epoch = 0
#This class open .mat files instead of images NOTE THAT IS A SUBCLASS OF Dataset so  inherit all his functions
#The dataFolder should have inside N folder called like the labels. Example
#if dataFolder is data/CWRfeatures inside that folder should be 4 folders called
# label1 label2 label3 label4 . This class also save a log with the int-label used
class DatasetMat(Dataset):
    def readMatrixData(self,dataFolder):

        #for each subfolder a new label (just read folder)
        allFolders = filter(lambda x : os.path.isdir(os.path.join(dataFolder,x)), os.listdir(dataFolder))
        if len(allFolders) == 0:
            print "ERROR NOT FOUND FOLDERS ",dataFolder
            print os.getcwd()
            raise Exception('NOT FOUND FOLDERs')
        now = time.strftime('Day%Y-%m-%d-Time%H-%M')
        log = "\n\n Dataset label log "+str(dataFolder)+str(now)+" \n"

        all = []
        allL = []
        for label,folder in enumerate(allFolders):
            # Read all the mat files inside
            log += "Label: "+str(label)+" -- "+str(folder)+" \n"
            fileList = os.listdir(os.path.join(dataFolder,folder))
            for f in fileList:
                if f.split('.')[-1] == 'mat': #check that the file is mat
                    matpath = os.path.join(dataFolder,folder,f)
                    fileName = f.split('.')[-2] #Extract the .mat from the filename
                    matrixData = sio.loadmat(matpath)[fileName] #The .mat is a dictionary and the fileName is the key for the data
                    self.imageSize = matrixData.shape[0]
                    self.dataShape = matrixData.shape
                    all.append(matrixData)
                    allL.append(label)
                    self.fileNames.append(f)
            self.classes = len(set(allL))  # Get how many different classes are in the problem

        # Create a log with label - folder
        with open(os.path.join(dataFolder,"0_description"+os.path.split(dataFolder)[-1]+".txt"),'w') as f:
            f.write(log)

        outAll = np.zeros((len(all),all[0].shape[0],all[0].shape[1]))
        for i in range(len(all)):
            outAll[i] = all[i]
        allL = np.array(allL)
        indAll = np.arange(outAll.shape[0])  # Get index to data to asociate filenames with data

        return outAll, allL, indAll

class cifar10Dataset(Dataset):
    def __init__(self, batch_size=100,testProp=0.3, validation_proportion=0.3,seed = 1, normalize=True):
        
        self.cifarFolder = "data/cifar10"
        cifar10.data_path = self.cifarFolder
        cifar10.maybe_download_and_extract()
        Dataset.__init__(self,self.cifarFolder,batch_size=batch_size,testProp=testProp, validation_proportion=validation_proportion,seed = seed, normalize=normalize)
    def readMatrixData(self,dataFolder):
        images_train, cls_train, labels_train = cifar10.load_training_data()
        indAll = np.arange(images_train.shape[0])  # Get index to data to asociate filenames with data
        images_train = images_train * 255 #The data is asumed to have 0-255 range in this case is 0-1 range
        #TODO add test set data (we are using the original train data as the full dataset)
        images_train= images_train.astype(np.uint8)
        self.dataShape = images_train[0].shape
        self.classes = 10 #how many clases are there
        return images_train,cls_train,indAll

    def getTrainFilename(self,trainIndex): #Cifar 10 images are picked so no filename just a index and a label
        return "image_"+str(trainIndex)+"_"+str(self.train_labels[trainIndex])

    def getValFilename(self,valIndex): #Cifar 10 images are picked so no filename just a index and a label
        return "image "+str(valIndex)+"_"+str(self.validation_labels[valIndex])

    def getTestFilename(self,testIndex): #Cifar 10 images are picked so no filename just a index and a label
        return "image "+str(testIndex)+"_"+str(self.test_labels[testIndex])


if __name__ == '__main__':
    test=DatasetMat("data/CWRfeatures stride8")
    #test = Dataset("data/simulatedFault",batch_size=20,seed=1)
    #test2 = cifar10Dataset(batch_size=20,seed=1)
