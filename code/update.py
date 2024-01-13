from torch.nn import functional as F
import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

def computeScore(model,allFeats,valLTemp):

    allOutput = {"allPred":None}
    splitSizes = [valLTemp for _ in range(allFeats.size(1)//valLTemp)]

    if allFeats.size(1)%valLTemp > 0:
        splitSizes.append(allFeats.size(1)%valLTemp)

    chunkList = torch.split(allFeats,split_size_or_sections=splitSizes,dim=1)

    sumSize = 0

    for i in range(len(chunkList)):
        output = model.secondModel(chunkList[i].squeeze(0),batchSize=1)

        for tensorName in output.keys():
            if not tensorName in allOutput.keys():
                allOutput[tensorName] = output[tensorName]
            else:
                allOutput[tensorName] = torch.cat((allOutput[tensorName],output[tensorName]),dim=1)

        sumSize += len(chunkList[i])

    return allOutput

def updateMetrics(args,model,allFeat,allTarget,precVidName,nbVideos,metrDict,outDict,targDict):

    allOutputDict = computeScore(model,allFeat,args.val_l_temp)

    allOutput = allOutputDict["pred"]

    if args.compute_metrics_during_eval:
        loss = F.cross_entropy(allOutput.squeeze(0),allTarget.squeeze(0)).data.item()

        metDictSample = metrics.binaryToMetrics(allOutput,allTarget,transition_matrix=model.transMat,allMetrics=False)
        metDictSample["Loss"] = loss
        metrDict = metrics.updateMetrDict(metrDict,metDictSample)

    outDict[precVidName] = allOutput
    targDict[precVidName] = allTarget

    nbVideos += 1

    return allOutput,nbVideos

def updateFrameDict(frameIndDict,frameInds,vidName):
    ''' Store the prediction of a model in a dictionnary with one entry per movie

    Args:
     - outDict (dict): the dictionnary where the scores will be stored
     - output (torch.tensor): the output batch of the model
     - frameIndDict (dict): a dictionnary collecting the index of each frame used
     - vidName (str): the name of the video from which the score are produced

    '''

    if vidName in frameIndDict.keys():
        reshFrInds = frameInds.view(len(frameInds),-1).clone()
        frameIndDict[vidName] = torch.cat((frameIndDict[vidName],reshFrInds),dim=0)

    else:
        frameIndDict[vidName] = frameInds.view(len(frameInds),-1).clone()

def updateBestModel(metricVal,bestMetricVal,exp_id,model_id,bestEpoch,epoch,net,isBetter,worseEpochNb):

    if isBetter(metricVal,bestMetricVal):
        if os.path.exists("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch)):
            os.remove("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch))

        torch.save(net.state_dict(), "../models/{}/model{}_best_epoch{}".format(exp_id,model_id, epoch))
        bestEpoch = epoch
        bestMetricVal = metricVal
        worseEpochNb = 0
    else:
        worseEpochNb += 1

    return bestEpoch,bestMetricVal,worseEpochNb
