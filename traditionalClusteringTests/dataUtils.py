from collections import Counter
import random
from scipy import spatial
from skimage.io import imsave
import numpy as np
import pylab as plt
from skimage.transform import rescale
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from launchers.plot import plotBlokeh

from infogan.misc.dataset import Dataset, DatasetMat
from infogan.misc.utilsTest import generate_new_color
import os

def OneHotToInt(labels):
    #If realLabels are in one hot pass to list of integer labels
    if labels[0].shape[0] > 1:
        labels = np.where(labels)[1].tolist()
    else:
        labels = labels.tolist()
    return labels

def flatInput(train_data,train_labels,test_data,test_labels):
    trainX = train_data
    trainY = train_labels

    tx = test_data
    ty = test_labels

    # Flatten data for test images as a vector and labels as numbers
    flattenDataset = np.nan_to_num(np.array([image.flatten() for image in trainX]))
    flatTest = np.nan_to_num(np.array([image.flatten() for image in tx]))
    fTrainLabels = np.where(trainY == 1)[1]
    fTestLabels = np.where(ty == 1)[1]
    return flattenDataset, flatTest, fTrainLabels, fTestLabels

def inverseNorm(image,dataset):
    imsize = dataset.getDataShape()

    im = image.reshape(imsize[0:2])
    return (im * dataset.std) + dataset.mean

def pca2Visua(pca,data,labels,nClases):
    # Show a few statistics of the data
    print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)
    print "PCA 2 comp of the data"

    transformed = pca.transform(data)

    plt.figure()
    allscatter = []
    for c in range(nClases):
        elements = np.where(labels == c)
        temp = plt.scatter(transformed[elements, 0], transformed[elements, 1],
                           facecolors='none', label='Class ' + str(c), c=np.random.rand(1, 3))
        allscatter.append(temp)
    plt.legend(tuple(allscatter),
               tuple(["class " + str(c) for c in range(nClases)]),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.show()


def showRandomImages(dataset,toShow=5):
    # REMEMBER THE IMAGES IN THE DATASET ARE NORMALIZED

    # if you want to see iamges as the files use
    # inverseNorm(img,dataset)
    nClases = dataset.getNclasses()

    dataX = dataset.train_data
    dataY = np.where(dataset.train_labels == 1)[1]

    for c in range(nClases):
        elements = np.where(dataY == c)


        name = str(c)

        # Show 5 random element of that class
        fig, grid = plt.subplots(1, toShow)
        print "Class ", c
        for j in range(toShow):
            r = random.randint(0, elements[0].shape[0])
            r = r if r <= elements[0].shape[0] else (elements[0].shape[0]-1)
            ind = elements[0][r]

            fileName = dataset.getTrainFilename(ind)
            image = dataX[ind, :,:]

            # If you want to see images as shown in files
            # image = inverseNorm(flattenDataset[ind,:],dataset)

            grid[j].imshow(image)
            grid[j].set_adjustable('box-forced')
            grid[j].autoscale(False)
            grid[j].set_title(str(fileName), fontsize=10)
            grid[j].axis('off')

def showMeanstd(dataset):
    nClases = dataset.getNclasses()
    # Show mean image of train per class
    # NOTE THIS IMAGE IS NOT THE MEAN IMAGE IN DATASET. This mean image is calculated with normalized images
    # THe original mean image was calculated with the original images in train set.
    imsize = dataset.getDataShape()[0:2]

    flattenDataset, flatTest, fTrainLabels, fTestLabels = \
        flatInput(dataset.train_data, dataset.train_labels, dataset.test_data, dataset.test_labels)


    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = np.mean(flattenDataset[elements, :], axis=1).reshape(imsize)
        grid[c].set_title('mean class ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    # SHOW IMAGES MEAN AS ORIGINAL IMAGE (INVERT NORM)
    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = inverseNorm(np.mean(flattenDataset[elements, :], axis=1), dataset)
        grid[c].set_title('mean Org ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = np.std(flattenDataset[elements, :], axis=1).reshape(imsize)
        grid[c].set_title('Std class ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    # SHOW IMAGES MEAN AS ORIGINAL IMAGE (INVERT NORM)
    fig, grid = plt.subplots(1, nClases)
    for c in range(nClases):
        elements = np.where(fTrainLabels == c)
        classMean = inverseNorm(np.std(flattenDataset[elements, :], axis=1), dataset)
        grid[c].set_title('Std Org ' + str(c))
        grid[c].imshow(classMean)
        grid[c].set_adjustable('box-forced')
        grid[c].autoscale(False)
        grid[c].axis('off')

    normMean = np.mean(flattenDataset, axis=0)
    plt.figure()
    plt.title('Mean full train set')
    plt.imshow(dataset.mean)


    plt.figure()
    plt.title('Mean norm full train set  (after invNorm)')
    plt.imshow(inverseNorm(normMean, dataset))


#MODEL TRAIN

def processPca(flattenDataset, flatTest, fTrainLabels, fTestLabels,nComp = 6):

    pca = PCA(n_components=nComp)
    pca.fit(flattenDataset)
    dataPcaTrainX = pca.transform(flattenDataset)
    dataPcaTrainy = fTrainLabels

    dataPcaTestX = pca.transform(flatTest)
    dataPcaTesty = fTestLabels
    return dataPcaTrainX, dataPcaTestX, dataPcaTrainy, dataPcaTesty


def showDataplots(dx,ltrain):
    import seaborn as sns
    data = pandas.DataFrame(data=dx, columns=['pca' + str(i) for i in range(dx.shape[1])])
    data['class'] = pandas.Series(ltrain, index=data.index)
    data.boxplot(by='class')
    sns.pairplot(data, hue="class")


# Use MLP
def runMLP(X, y, tX, ty, returnVal=False):
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(50, 3), random_state=1)

    clf.fit(X, y)
    pred = clf.predict(tX)

    if not returnVal:
        print classification_report(ty, pred)
    else:
        return accuracy_score(ty, pred)

# Save points,labels,name,fileNames TODO still here uses validation hardcoded
# TODO HERE FOR SOME REASON THE VAL DATA-LABELS ARE DESFASES BY 1 BATCH SIZE
# if you are in the points[0], realLabels[0] the corresponding saverls[128] iamgeSave[128]
def fixrealLabels(dataset,realLabels):
    iamgeSave = dataset.dataObj.validation_data  # TODO DONT USE THE PRIVATE VARIABLES
    saverls = np.where(dataset.dataObj.validation_labels)[1]  # HERE IS THE PROBLEM FOR THE VAL - TRAIN CASE. If you merge the you cant get the images back
    desfase = saverls.shape[0] - np.array(realLabels).shape[0]
    # THIS WILL MAKE THE saverls and realLabels coincide in values

    if desfase != 0:
        saverls = np.roll(saverls[:-1 * desfase], -dataset.dataObj.batch_size)

    names = np.array([dataset.dataObj.getValFilename(i) for i in
                 range(saverls.shape[0])])
    if desfase != 0:
        iamgeSave = np.roll(iamgeSave[:-1 * desfase],-dataset.dataObj.batch_size, axis=0)
        names = np.roll(names[:-1 * desfase], -dataset.dataObj.batch_size)


    realLabels = realLabels[0:names.shape[0]]
    saverls = saverls[0:names.shape[0]]
    return names,realLabels,iamgeSave,saverls

def plot3Dpca(dataset,points,labels,pickleName,outFolder):
    import pickle
    #Create 3d PCA and export pickle
    if type(labels) == list:
        labels = np.array(labels)

    fitted=PCA(n_components=3)
    fitted.fit(points)
    transformed = fitted.transform(points)

    names, labels, iamgeSave, saverls = fixrealLabels(dataset, labels)
    transformed = transformed[0:labels.shape[0]]
    assert (transformed.shape[0] == labels.shape[0])
    assert (transformed.shape[0] == names.shape[0])
    pickleName = os.path.join(outFolder,pickleName+" exp"+str(fitted.explained_variance_ratio_)+'.pkl')
    pickleName=pickleName.replace(" ","_")
    with open(pickleName, 'wb') as f:
        pickle.dump([transformed, labels, names, dataset.dataObj.getNclasses(), dataset.getFolderName()],f,-1)

def plotInteractive(transformed,realLabels,dataset,name,outputFolder): #HERE IT MUST USE THE VALIDATION SET

    savepoints = transformed

    names,realLabels,iamgeSave,saverls = fixrealLabels(dataset,realLabels)

    assert (np.array_equal(saverls, realLabels))

    plotBlokeh([savepoints, realLabels, iamgeSave, names],name,outputFolder, test=False)

def showResults(dataset,points,labels,realLabels,name,outFolder,ax=None,showBlokeh=False,show3d=False,labelNames=None):
    outNameFile = "Results for "+str(name)+'.txt'
    log = ""


    #SHOW PCA2 of data with PREDICTED labels

    pca = PCA(n_components=2)

    transformed = showDimRed(points, labels, name + str('PCA_Predicted'), pca,outFolder,labelsName=labelNames)
    print  "Pca with 2 components explained variance " + str(pca.explained_variance_ratio_)

    if show3d:
        #plot the 3pca of the transform
        plot3Dpca(dataset, points, labels, name, outFolder)

    if showBlokeh: #TODO showBlokeh is just true once in a transformation to be sure to plot just one real labels per transform
        plotInteractive(transformed, realLabels, dataset,name,outFolder)
        #plot the 3pca of the real labels. Just one per transform
        plot3Dpca(dataset, points, realLabels, name.split()[0]+"REAL_LABELS", outFolder)
        pass

    n_classes = len(set(labels))
    labels = np.array(labels)

    globalClassScore = []

    show = False
    if n_classes > 60:
        show = False
        log += ("The number of cluster is > 60 So no images will be shown "+ '\n')

    for i in range(n_classes):


        #Get all index of that class
        elements = np.where(labels == i)



        log += ("For class "+str(i)+" there are "+str(elements[0].shape[0])+'\n')
        #Get the real classes for those points
        rl = np.array(realLabels)[elements]
        #Show distribution
        dist = Counter(rl)
        log += ("Showing Real distribution for that generated Label "+str(dist)+'\n')
        totalPointsClass = sum(dist.values())
        pdist = [(elem,dist[elem]*1.0/ totalPointsClass) for elem in set(dist.elements())]
        log += ("%dist "+str(pdist)+'\n')

        #place the clasification score
        if len(pdist) > 0:
            classScore = max(pdist, key = lambda x : x[1])[1]
            classScore = classScore * ( totalPointsClass * 1.0 / labels.shape[0]) #Ponderate all classScore according to N of points
        else:
            classScore = 0
        log += ("Clasification score " + str(classScore) + '\n')
        globalClassScore.append(classScore)

        if show:
            # Make a KD-tree to seach nearest points
            tree = spatial.KDTree(points)
            tempFolder = name + ' Predicted ' + str(i)
            if not os.path.exists(os.path.join(outFolder, tempFolder)):
                os.makedirs(os.path.join(outFolder, tempFolder))
            #Calculate centroid
            selected = points[elements]
            centroid = np.mean(selected,axis=0)
            #Get X closest points from that centroid
            toShow =10
            distances,indexs = tree.query(centroid,k=toShow)
            #Get those points images
            for j in range(toShow):
                image = dataset.dataObj.validation_data[indexs[j]] #TODO DONT USE THE PRIVATE VARIABLES
                label = dataset.dataObj.validation_labels[indexs[j]] #HERE IS THE PROBLEM FOR THE VAL - TRAIN CASE. If you merge the you cant get the images back

                title = 'PredictedLabel '+str(i)+" Cls " + str(j) + " dist " + str(distances[j]) + " RealLabel " + str(label)
                toSave = rescale(image , 2)
                toSave = (toSave / 255).astype(np.float)
                imsave(os.path.join(outFolder,tempFolder,title+'.png'), toSave)

    #Log all the global information
    log += ('\n')
    log += ('Global metrics \n')
    log += ('\n')
    log += ("The ARI was " + str(metrics.adjusted_rand_score(realLabels, labels)) + '\n')
    log += ("The NMI was " + str(metrics.normalized_mutual_info_score(realLabels, labels)) + '\n')
    log += ("Predicted cluster number is " + str(n_classes) + '\n')
    log += ("The mean classificationScore is " + str(np.sum(np.array(globalClassScore))) + '\n')
    log += ("The std classificationScore is " + str(np.std(np.array(globalClassScore))) + '\n')
    log += ("PCA2 with predicted variance" + str(pca.explained_variance_ratio_)+ '\n')

    with open(os.path.join(outFolder,outNameFile),'w+') as f:
        f.write(log)
    return log

def showDimRed(points, labels, name, dimRalg, outF=None,labelsName=None):

    try:
        labels=OneHotToInt(labels)
    except:
        labels=labels



    transform_op = getattr(dimRalg, "transform", None)
    if callable(transform_op):
        dimRalg.fit(points)
        transformed = dimRalg.transform(points)
    else:
        transformed = dimRalg.fit_transform(points)

    #Save plot to pickle
    r3 = pandas.DataFrame(data=transformed)
    r3['c'] = np.array(labels)
    r3.to_pickle(os.path.join(outF, name + '.pkl'))



    objPlot = None

    plt.figure()
    plt.title(name)
    objPlot = plt

    allscatter = []
    n_classes = len(set(labels))

    colors = []
    for i in range(0, n_classes):
        colors.append(generate_new_color(colors, pastel_factor=0.9))

    for c in range(n_classes):
        elements = np.where(np.array(labels) == c)
        temp = objPlot.scatter(transformed[elements, 0], transformed[elements, 1],
                               facecolors='none', label='Class ' + str(c), c=colors[c])
        allscatter.append(temp)

    #Choose label names if labelsName is None use class X, else use labelsNames
    lnames = ["class " + str(c) for c in range(n_classes)] if labelsName is None else labelsName
    objPlot.legend(tuple(allscatter),
                   tuple(lnames),
                   scatterpoints=1,
                   loc='upper left',
                   ncol=3,
                   fontsize=8)
    if outF is None:
        plt.show()
    else:
        plt.savefig(os.path.join(outF, name + '.png'))

    return transformed

# Use Linear support machine
def useLinear(X, y, tX, ty, returnVal=False):
    clf = SGDClassifier()
    clf.fit(X, y)
    pred = clf.predict(tX)

    if not returnVal:
        print classification_report(ty, pred)
    else:
        return accuracy_score(ty, pred)


# USE SVM (uses rbf o gaussian kernel)
def useSVM(X, y, tX, ty, returnVal=False):
    clf = SVC()
    clf.fit(X, y)
    pred = clf.predict(tX)

    if not returnVal:
        print classification_report(ty, pred)
    else:
        return accuracy_score(ty, pred)

def getNewDataset(datasetFolder,seed=None,matDataset=False):
    if seed is None:
        seed = int(100 * random.random())
    #Changing the seed will give a new train-val-test split
    if not matDataset:
        dataset = Dataset(datasetFolder, batch_size=20,seed=seed)
    else:
        dataset = DatasetMat(datasetFolder, batch_size=20, seed=seed)
    flattenDataset, flatTest, fTrainLabels, fTestLabels = flatInput(dataset.train_data, dataset.train_labels, dataset.test_data, dataset.test_labels)
    return flattenDataset, flatTest, fTrainLabels, fTestLabels, dataset




def simplemethodsresults(datasetFolder,transform,limitPoints=None,isMat=False):
    xTimes = 3
    resultMLP = []
    resultLinear = []
    resultSVM = []


    # FOR the 3 alg repeat 10 times
    for i in range(xTimes):
        flattenDataset, flatTest, fTrainLabels, fTestLabels, _ = getNewDataset(datasetFolder,matDataset=isMat)
        if limitPoints is None:
            X, tX, y, ty = transform(flattenDataset, flatTest, fTrainLabels, fTestLabels)
        else:
            X, tX, y, ty = transform(flattenDataset, flatTest, fTrainLabels, fTestLabels)
            X = X[:limitPoints,:]
            y = y[:limitPoints]

        resultMLP.append(runMLP(X, y, tX, ty, True))
        resultLinear.append(useLinear(X, y, tX, ty, True))
        resultSVM.append(useSVM(X, y, tX, ty, True))

    meanMLP = np.mean(np.array(resultMLP))
    stdMLP = np.std(np.array(resultMLP))

    meanLSVM = np.mean(np.array(resultLinear))
    stdLSVM = np.std(np.array(resultLinear))

    meanSVM = np.mean(np.array(resultSVM))
    stdSVM = np.std(np.array(resultSVM))
    print "Results: "
    print "MLP accuracy mean ", meanMLP, " std ", stdMLP
    print "Linear SVM accuracy mean ", np.mean(np.array(resultLinear)), " std ", np.std(np.array(resultLinear))
    print "SVM accuracy mean ", np.mean(np.array(resultSVM)), " std ", np.std(np.array(resultSVM))

    return [[meanMLP,stdMLP],[meanLSVM,stdLSVM],[meanSVM,stdSVM]]

def count_number_trainable_params(tf):
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        print shape, " ", current_nb_params
        tot_nb_params = tot_nb_params + current_nb_params
    print "Total ", tot_nb_params
    return tot_nb_params


def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params * int(dim)
    return nb_params


