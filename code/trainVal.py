import os,sys

import numpy as np
import torch
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import args
from args import ArgReader
from args import str2bool
import modelBuilder
import load_data
import metrics
import update
import glob 

def epochSeqTr(model,optim,log_interval,loader, epoch, args,**kwargs):
    model.train()

    print("Epoch",epoch," : train")

    metrDict = metrics.emptyMetrDict(allMetrics=False)
    validBatch = 0

    for batch_idx,batch in enumerate(loader):

        if (batch_idx % log_interval == 0):
            processedImgNb = batch_idx*len(batch[0])*len(batch[0][0])
            print("\t",processedImgNb,"/",len(loader.dataset))

        data,target = batch[0],batch[1]
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        resDict = model(data)
        output = resDict["pred"]
        loss = computeLoss(args.nll_weight,output,target)
        loss.backward()

        optim.step()
        optim.zero_grad()

        #Metrics
        metDictSample = metrics.binaryToMetrics(output,target,transition_matrix=model.transMat,allMetrics=False)
        metDictSample["Loss"] = loss.detach().data.item()
        metrDict = metrics.updateMetrDict(metrDict,metDictSample)

        validBatch += 1

        if args.debug and validBatch > 4:
            break

    torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id, epoch))
    metrDict = writeSummaries(metrDict,validBatch,epoch,"train",args.model_id,args.exp_id)

    printMetrDict(metrDict)

def computeLoss(nll_weight,output,target):
    output = output.view(output.size(0)*output.size(1),-1)
    target = target.view(-1)
    loss = nll_weight*F.cross_entropy(output, target)
    return loss

def epochSeqVal(model,log_interval,loader, epoch, args,metricEarlyStop,mode="val"):

    model.eval()

    print("Epoch",epoch," : ",mode)

    metrDict = metrics.emptyMetrDict(allMetrics=False)

    nbVideos = 0

    outDict,targDict = {},{}

    frameIndDict = {}

    precVidName = "None"
    videoBegining = True
    nbVideos = 0

    for batch_idx, (data,target,vidName,frameInds) in enumerate(loader):

        newVideo = (vidName != precVidName) or videoBegining

        if (batch_idx % log_interval == 0):
            print("\t",loader.sumL+1,"/",loader.nbImages)

        if args.cuda:
            data, target,frameInds = data.cuda(), target.cuda(),frameInds.cuda()

        visualDict = model.computeVisual(data)
        feat = visualDict["x"].data

        update.updateFrameDict(frameIndDict,frameInds,vidName)

        if newVideo and not videoBegining:  
            if mode == "test" and 'attMaps' in visualDict:
                saveMaps("attMaps",allAttMaps,precVidName,args.exp_id,args.model_id)   
                saveMaps("norm",allNorms,precVidName,args.exp_id,args.model_id)                                 
            allOutput,nbVideos = update.updateMetrics(args,model,allFeat,allTarget,precVidName,nbVideos,metrDict,outDict,targDict)

        if newVideo:
            allTarget = target
            allFeat = feat.unsqueeze(0)
            allAttMaps = visualDict["attMaps"].cpu() if mode == "test" and 'attMaps' in visualDict else None
            allNorms = visualDict["norm"].cpu() if mode == "test" and 'norm' in visualDict else None 
            videoBegining = False
        else:
            allTarget = torch.cat((allTarget,target),dim=1)
            allFeat = torch.cat((allFeat,feat.unsqueeze(0)),dim=1)
            allAttMaps = torch.cat((allAttMaps,visualDict["attMaps"].cpu()),dim=0) if mode == "test" and 'attMaps' in visualDict else None  
            allNorms = torch.cat((allNorms,visualDict["norm"].cpu()),dim=0) if mode == "test" and 'norm' in visualDict else None  
                 
        precVidName = vidName
    if mode == "test" and 'attMaps' in visualDict:
        saveMaps("attMaps",allAttMaps,precVidName,args.exp_id,args.model_id)  
        saveMaps("norm",allNorms,precVidName,args.exp_id,args.model_id)  

    allOutput,nbVideos = update.updateMetrics(args,model,allFeat,allTarget,precVidName,nbVideos,metrDict,outDict,targDict)

    if mode == "test":
        for videoName in outDict.keys():
            fullArr = torch.cat((frameIndDict[videoName].float(),outDict[videoName].squeeze(0).squeeze(1)),dim=1)
            np.savetxt("../results/{}/{}_{}.csv".format(args.exp_id,args.model_id,videoName),fullArr.cpu().detach().numpy())

    metrDict = writeSummaries(metrDict,nbVideos,epoch,mode,args.model_id,args.exp_id)

    printMetrDict(metrDict)

    return metrDict[metricEarlyStop]

def saveMaps(key,allMaps,vidName,exp_id,model_id):

    #100, 7, 3, 29, 29

    allMax = allMaps.view(allMaps.shape[0],-1).max(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    allMin = allMaps.view(allMaps.shape[0],-1).min(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    print(allMax.shape,allMin.shape,allMaps.shape)

    allMaps = (255*(allMaps-allMin)/(allMax-allMin)).numpy().astype("uint8")

    np.save(f"../results/{exp_id}/{key}_{vidName}_{model_id}.npy",allMaps)

def computeTransMat(transMat,priors,split,allVid=False):

    if allVid:
        videoPaths = load_data.findVideos("train",split)
        videoPaths += load_data.findVideos("val",split)
        videoPaths += load_data.findVideos("test",split)
    else:
        videoPaths = load_data.findVideos("train",split)

    for videoPath in videoPaths:
        videoName = os.path.splitext(os.path.basename(videoPath))[0]
        target = load_data.getGT(videoName)
        #Updating the transition matrix
        for i in range(len(target)-1):
            transMat[target[i],target[i+1]] += 1
            priors[target[i]] += 1

        #Taking the last target of the sequence into account only for prior
        priors[target[-1]] += 1

    #Just in case where propStart==propEnd, which is true when the training set is empty for example
    if len(videoPaths) > 0:
        return transMat/transMat.sum(dim=1,keepdim=True),priors/priors.sum()
    else:
        return transMat,priors

def writeSummaries(metrDict,sampleNb,epoch,mode,model_id,exp_id):

    for metric in metrDict.keys():
        metrDict[metric] = (metrDict[metric]/sampleNb)

        if torch.is_tensor(metrDict[metric]):
            metrDict[metric] = metrDict[metric].item()

    file_path = "../results/{}/{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode)

    if not os.path.exists(file_path):
        header = ",".join([metric.lower().replace(" ","_") for metric in metrDict.keys()])
    else:
        header = ""

    with open(file_path,"w") as text_file:
        print(header,file=text_file)
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]),file=text_file)

    return metrDict

def printMetrDict(metrDict):

    print("Loss",round(metrDict["Loss"],3),end=" ; ")
    for metric in metrDict:
        if metric.find("Accuracy") != -1:
            print(metric,round(100*metrDict[metric],1),end="% ; ")
    print("")

def get_OptimConstructor_And_Kwargs(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
        optimConst = getattr(torch.optim,optimStr)
        if optimStr == "SGD":
            kwargs= {'momentum': momentum}
        elif optimStr == "Adam":
            kwargs = {}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'amsgrad':True}

    return optimConst,kwargs

def addInitArgs(argreader):
    argreader.parser.add_argument('--start_mode', type=str,metavar='SM',
                help='The mode to use to initialise the model. Can be \'scratch\' or \'fine_tune\'.',default="auto")
    argreader.parser.add_argument('--init_path', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the network')
    argreader.parser.add_argument('--strict_init', type=str2bool,metavar='SM',
                help='Set to True to make torch.load_state_dict throw an error when not all keys match (to use with --init_path)')

    return argreader
def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=args.str2FloatList,metavar='LR',
                        help='learning rate (it can be a schedule : --lr 0.01,0.001,0.0001)')
    argreader.parser.add_argument('--momentum', type=args.str2float, metavar='M',
                        help='SGD momentum')
    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                        help='the optimizer to use (default: \'SGD\')')
    return argreader
def addValArgs(argreader):

    argreader.parser.add_argument('--val_l_temp', type=args.str2int,metavar='LMAX',help='Length of sequences for computation of scores when using a CNN temp model.')

    argreader.parser.add_argument('--metric_early_stop', type=str,metavar='METR',
                    help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_val_metric', type=args.str2bool,metavar='BOOL',
                    help='If true, The chosen metric for chosing the best model will be maximised')
    argreader.parser.add_argument('--max_worse_epoch_nb', type=args.str2int,metavar='NB',
                    help='The number of epochs to wait if the validation performance does not improve.')
    argreader.parser.add_argument('--run_test', type=args.str2bool,metavar='NB',
                    help='Evaluate the model on the test set')

    argreader.parser.add_argument('--compute_metrics_during_eval', type=args.str2bool,metavar='BOOL',
                    help='If false, the metrics will not be computed during validation, but the scores produced by the models will still be saved')

    return argreader
def addLossTermArgs(argreader):

    argreader.parser.add_argument('--nll_weight', type=args.str2float,metavar='FLOAT',
                    help='The weight of the negative log-likelihood term in the loss function.',default=1)

    return argreader

def run(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainLoader,_ = load_data.buildSeqTrainLoader(args)
    valLoader = load_data.buildSeqTestLoader(args,"val")

    #Building the net
    net = modelBuilder.netBuilder(args)

    trainFunc = epochSeqTr
    valFunc = epochSeqVal

    kwargsTr = {'log_interval':args.log_interval,'loader':trainLoader,'args':args}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader
    kwargsVal["metricEarlyStop"] = args.metric_early_stop

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargsOpti = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)
    kwargsOpti["lr"] = args.lr
    optim = optimConst(net.parameters(), **kwargsOpti)
    kwargsTr["optim"] = optim

    paths = glob.glob("../models/{}/model{}_epoch*".format(args.exp_id,args.model_id))
    if len(paths) > 0:
        lastWeightsPath = sorted(paths,key=lambda x:int(os.path.basename(x).split("epoch")[1]))[-1]
        net.load_state_dict(torch.load(lastWeightsPath))
        startEpoch = int(os.path.basename(lastWeightsPath).split("epoch")[1])+1
        bestEpochPaths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.model_id))

        if len(bestEpochPaths) ==0 or len(bestEpochPaths) > 1:
            raise ValueError("There is no or too many best epoch weight file. There should be only one.")

        bestEpoch = int(os.path.basename(bestEpochPaths[0]).split("epoch")[1])
        worseEpochNb = startEpoch - bestEpoch

        bestMetricFilePath = "../results/{}/{}_epoch{}_metrics_val.csv".format(args.exp_id,args.model_id,bestEpoch)
        bestMetricVal = float(np.genfromtxt(bestMetricFilePath,delimiter=",")[1,1])

        print("Re-starting at epoch {} with best metric {}".format(startEpoch,bestMetricVal))

    else:
        startEpoch = 1
        bestMetricVal = -np.inf if args.maximise_val_metric else np.inf
        bestEpoch,worseEpochNb = startEpoch,0

    transMat,priors = computeTransMat(net.transMat,net.priors,args.split)
    net.setTransMat(transMat)
    net.setPriors(priors)

    epoch = startEpoch
 
    if args.maximise_val_metric: 
        isBetter = lambda x,y:x>y
    else:
        isBetter = lambda x,y:x<y

    while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

        kwargsTr["epoch"],kwargsVal["epoch"] = epoch,epoch
        kwargsTr["model"],kwargsVal["model"] = net,net

        trainFunc(**kwargsTr)

        with torch.no_grad():
            metricVal = valFunc(**kwargsVal)

        bestEpoch,bestMetricVal,worseEpochNb = update.updateBestModel(metricVal,bestMetricVal,args.exp_id,args.model_id,bestEpoch,epoch,net,isBetter,worseEpochNb)

        epoch += 1
        
    testFunc = valFunc

    kwargsTest = kwargsVal
    kwargsTest["mode"] = "test"

    testLoader = load_data.buildSeqTestLoader(args,"test")

    kwargsTest['loader'] = testLoader

    if os.path.exists("../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id,bestEpoch)):        
        params = torch.load("../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id,bestEpoch),map_location="cpu" if not args.cuda else None)
    else:
        params = torch.load("../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id,bestEpoch),map_location="cpu" if not args.cuda else None)

    #Replace old names by new ones
    newParams = {}
    for key in params:
        if key.find("visualModel") !=-1:
            newKey = key.replace("visualModel","firstModel")
        elif key.find("tempModel") !=-1:
            newKey = key.replace("tempModel","secondModel")               
        else:
            newKey = key    
        newParams[newKey] = params[key]
    params = newParams

    #Add or remove 'module'

    param_has_module = list(params.keys())[0].find("module.") != -1
    net_has_module = list(net.state_dict().keys())[0].find("module.") != -1
    if not param_has_module and net_has_module:
        newParams = {}
        for key in params:           
            newParams["module."+key] = params[key]
        params = newParams
    elif param_has_module and not net_has_module:
        newParams = {}
        for key in params:           
            newParams[key.replace("module.","")] = params[key]
        params = newParams

    net.load_state_dict(params)
    kwargsTest["model"] = net
    kwargsTest["epoch"] = bestEpoch

    with torch.no_grad():
        testFunc(**kwargsTest)

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    #The folders where the experience file will be written
    os.makedirs("../vis/{}".format(args.exp_id),exist_ok=True)
    os.makedirs("../results/{}".format(args.exp_id),exist_ok=True)
    os.makedirs("../models/{}".format(args.exp_id),exist_ok=True)

    # Update the config args
    argreader.args = args

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id,args.model_id))

    print("Model :",args.model_id,"Experience :",args.exp_id)

    if os.path.exists("/home/E144069X/"):
        print("Working on laptop, lowressource mode")
        args.debug = True 
        args.feat = "resnet4"
        args.cuda = False 
        #args.batch_size = 1 
        args.val_batch_size = 1
        args.img_size = 114

    run(args)

if __name__ == "__main__":
    main()
