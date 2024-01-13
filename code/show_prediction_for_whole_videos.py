
from numpy.lib.npyio import load
from args import ArgReader
from args import str2bool
import os
import glob

import torch
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.manifold import TSNE
import sklearn
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as patches
import pims
import cv2
from PIL import Image

import load_data

import metrics
import utils
import formatData
import trainVal
import formatData
import scipy
import sys

import configparser

import matplotlib.patheffects as path_effects
import imageio
from skimage import img_as_ubyte

from scipy import stats
import math
from PIL import Image
from PIL import Image, ImageEnhance

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import torchvision
import pims
from skimage import exposure

def agregateTempAcc(acc,accParamDict):
    meanAccPerThres = acc.mean(axis=0)
    meanAccPerVid =  acc.mean(axis=1)
    return meanAccPerVid,meanAccPerThres

def getThresList(paramDict):
    tempDic = paramDict["Temp Accuracy"]
    thresList = np.arange(tempDic["minThres"],tempDic["maxThres"],tempDic["step"])
    return thresList

def saveTempAccPerThres(exp_id,model_id,meanAccPerThres):
    np.savetxt("../results/{}/{}_tempAcc.csv".format(exp_id,model_id),meanAccPerThres)


def formatMetr(mean,std,corr):
    if corr:
        return "$"+str(round(mean,4))+" \pm "+str(round(std,4))+"$"
    else:
        return "$"+str(round(mean,2))+" \pm "+str(round(std,2))+"$"

def plotScore(exp_id,model_ids,model_labels):
    ''' This function plots the scores given by a model to seral videos. '''

    fontSize = 35

    #This dictionnary returns a label using its index
    revLabelDict = formatData.getReversedLabels()
    labDict = formatData.getLabels()
    reverLabDict = formatData.getReversedLabels()
    cmap = cm.hsv(np.linspace(0, 1, len(revLabelDict.keys())))

    conf = configparser.ConfigParser()
    conf.read("../models/{}/{}.ini".format(exp_id,model_ids[0]))
    testPartBeg = int(float(conf["default"]["test_part_beg"]))
    testPartEnd = int(float(conf["default"]["test_part_end"]))
    trainPartBeg = int(float(conf["default"]["train_part_beg"]))
    trainPartEnd = int(float(conf["default"]["train_part_end"]))
    dataset = conf["default"]["dataset_test"]

    epochs = []
    for model_id in model_ids:
        bestWeightPath = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id,model_id))[0]
        bestEpoch = int(os.path.basename(bestWeightPath).split("epoch")[1])
        try:
            resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,bestEpoch))))
            videoDict = buildVideoNameDict(dataset,testPartBeg,testPartEnd,True,resFilePaths,raiseError=True)
        except ValueError:
            bestEpoch = findLastestEpoch(exp_id,model_id)
            resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,bestEpoch))))
            videoDict = buildVideoNameDict(dataset,testPartBeg,testPartEnd,True,resFilePaths,raiseError=True)
        epochs.append(bestEpoch)

    resFilePaths = []
    for i in range(len(model_ids)):
        resFilePaths.extend(glob.glob("../results/{}/{}_epoch{}*.csv".format(exp_id,model_ids[i],epochs[i])))
    resFilePaths = sorted(resFilePaths)

    videoPaths = load_data.findVideos(dataset,propStart=0,propEnd=1)
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    videoNameDict = buildVideoNameDict(dataset,testPartBeg,testPartEnd,propSetIntFormat=True,resFilePaths=resFilePaths,raiseError=True)
    revVideoNameDict = {}
    for resFilePath in videoNameDict.keys():
        if not videoNameDict[resFilePath] in revVideoNameDict.keys():
            revVideoNameDict[videoNameDict[resFilePath]] = [resFilePath]
        else:
            revVideoNameDict[videoNameDict[resFilePath]].append(resFilePath)

    for videoName in revVideoNameDict.keys():
        _, axList = plt.subplots(len(model_ids)*2+1, 1,figsize=(30,8),sharex=True)
        print(videoName)
        for i,model_id in enumerate(model_ids):

            bestWeightPath = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id,model_id))[0]
            bestEpoch = int(os.path.basename(bestWeightPath).split("epoch")[1])

            conf.read("../models/{}/{}.ini".format(exp_id,model_id))

            path = "../results/{}/{}_epoch{}_{}.csv".format(exp_id,model_ids[i],epochs[i],videoName)
            fileName = os.path.basename(os.path.splitext(path)[0])

            scores = np.genfromtxt(path,delimiter=" ")
            nbFrames = scores[-1,0]
            scores = scores[:,1:]

            legHandles = []

            #Plot the scores
            expVal = np.exp(scores)
            scores = expVal/expVal.sum(axis=-1,keepdims=True)

            #Plot the prediction only considering the scores and not the state transition matrix
            predSeq = scores.argmax(axis=-1)
            predSeq = labelIndList2FrameInd(predSeq,reverLabDict)
            legHandles = plotPhases(predSeq,legHandles,labDict,cmap,axList[i],model_labels[i],fontSize)

            #Plot the prediction with viterbi decoding
            transMat = torch.zeros((scores.shape[1],scores.shape[1]))
            priors = torch.zeros((scores.shape[1],))
            transMat,_ = trainVal.computeTransMat(dataset,transMat,priors,trainPartBeg,trainPartEnd,propSetIntFormat=True)
            predSeqs,_ = metrics.viterbi_decode(torch.log(torch.tensor(scores).float()),torch.log(transMat),top_k=1)
            predSeq = labelIndList2FrameInd(predSeqs[0],reverLabDict)
            legHandles = plotPhases(predSeq,legHandles,labDict,cmap,axList[i+len(model_ids)],model_labels[i]+" (Vit.)",fontSize)

            #break

        #Plot the ground truth phases
        foundGT = False
        for splitDataset in dataset.split("+"):
            if os.path.exists("../data/"+splitDataset+"/annotations/"+videoName+"_phases.csv"):
                gt = np.genfromtxt("../data/"+splitDataset+"/annotations/"+videoName+"_phases.csv",dtype=str,delimiter=",")
                foundGT = True
        if not foundGT:
            raise ValueError("Didn't find GT for {}".format(videoName))
        legHandles = plotPhases(gt,legHandles,labDict,cmap,axList[-1],"GT",fontSize)

        filteredLegHandles = []
        labelList = []
        for legHandle in legHandles:
            if not legHandle._label in labelList:
                filteredLegHandles.append(legHandle)
                labelList.append(legHandle._label)
        legHandles = filteredLegHandles

        labelList,legHandles = zip(*sorted(zip(labelList,legHandles),key=lambda x:formatData.getLabels()[x[0]]))
        labelList = list(map(lambda x:x.replace("t","p"),labelList))

        plt.xlabel("Time (image index)",fontsize=fontSize)
        halfL = len(legHandles) // 2 + 1
        leg1 = plt.legend(legHandles[:halfL],labelList[:halfL],bbox_to_anchor=(1.0, 8),prop={'size': fontSize})
        leg2 = plt.legend(legHandles[halfL:],labelList[halfL:],bbox_to_anchor=(1.35, 8),prop={'size': fontSize})
        plt.gca().add_artist(leg1)
        plt.gca().add_artist(leg2)        
        plt.subplots_adjust(hspace=0.1,right=0.775,bottom=0.145,left=0.16)
        plt.xticks(fontSize=fontSize)
        plt.savefig("../vis/{}/{}_epoch{}_video{}_scores.png".format(exp_id,"-".join(model_ids),"-".join([str(epoch) for epoch in epochs]),videoName))
        plt.close()

def plotPhases(phases,legHandles,labDict,cmap,ax,ylab,fontSize):
    phases = np.array(phases)
    #phases[:,1:] = 0.9*phases[:,1:].astype("float")
    phases[:,1:] = (phases[:,1:].astype("int")-phases[:,1:].astype("int").min())
    #print(ylab,phases)
    for i,phase in enumerate(phases):
        rect = patches.Rectangle((int(0.9*float(phase[1])),0),int(0.9*(float(phase[2])+1))-int(0.9*float(phase[1])),1,linewidth=1,\
                                    facecolor=cmap[labDict[phase[0]]],alpha=1,label=phase[0],edgecolor="black")
        legHandles += [ax.add_patch(rect)]
    ax.set_xlim(0,int(0.9*float(phases[-1][2])))
    ax.set_ylabel(ylab,rotation="horizontal",fontsize=fontSize,horizontalalignment="right",position=(0,0.3))
    ax.set_yticklabels(labels=[],fontdict=[])
    #if ylab != "GT":
    #    ax.set_xticklabels(labels=[],fontdict=[])
    return legHandles

def labelIndList2FrameInd(labelList,reverLabDict):

    currLabel = labelList[0]
    phases = []
    currStartFrame = 0
    for i in range(len(labelList)):

        if labelList[i] != currLabel:
            phases.append((reverLabDict[currLabel],currStartFrame,i-1))
            currStartFrame = i
            currLabel = labelList[i]

    phases.append((reverLabDict[currLabel],currStartFrame,i))
    return phases

def buildVideoNameDict(dataset,test_part_beg,test_part_end,propSetIntFormat,resFilePaths,raiseError=True):

    ''' Build a dictionnary associating a path to a video name (it can be the path to any file than contain the name of a video in its file name) '''
    videoPaths = load_data.findVideos(dataset,test_part_beg,test_part_end,propSetIntFormat)
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    videoNameDict = {}
    for path in resFilePaths:
        for videoName in videoNames:
            if videoName in path:
                videoNameDict[path] = videoName

    if len(videoNameDict.keys()) < len(videoNames):
        if raiseError:
            raise ValueError("Could not find result files corresponding to some videos. Files identified :",sorted(list(videoNameDict.keys())))

    return videoNameDict

def readConfFile(path,keyList):
    ''' Read a config file and get the value of desired argument

    Args:
        path (str): the path to the config file
        keyList (list): the list of argument to read name)
    Returns:
        the argument value, in the same order as in keyList
    '''

    conf = configparser.ConfigParser()
    conf.read(path)
    conf = conf["default"]
    resList = []
    for key in keyList:
        resList.append(conf[key])

    return ",".join(resList)

def findLastestEpoch(exp_id,model_id):
    allResFilesPath = sorted(glob.glob("../results/{}/{}_epoch*".format(exp_id,model_id)))
    allEpochs = list(map(lambda x:int(x[x.find("epoch")+5:].split("_")[0]),allResFilesPath))
    allEpochs = set(allEpochs)
    latestEpoch = sorted(list(allEpochs))[-1]
    return latestEpoch

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)
    argreader = load_data.addArgs(argreader)
    argreader.parser.add_argument('--names',type=str,nargs="*",metavar="NAME",help='The list of string to replace each key by during agregation.')
    argreader.parser.add_argument('--model_ids',type=str,nargs="*",metavar="NAME",help='The id of the models to process.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    plotScore(args.exp_id,args.model_ids,args.names)

if __name__ == "__main__":
    main()
