
from args import ArgReader
import configparser
import os,sys
import glob

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import load_data
import metrics
import utils
import trainVal

def evalModel(split,exp_id,model_id,nbClass):
 
    print("Evaluating model",model_id)

    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_*.csv".format(exp_id,model_id)),key=utils.findNumbers))
    resFilePaths = list(filter(lambda x:os.path.basename(x).find("metrics") == -1,resFilePaths))

    metricNameList = metrics.emptyMetrDict(allMetrics=True).keys()
    metEval={}

    for metricName in metricNameList:
        if metricName.find("Accuracy") != -1:
            metEval[metricName] = np.zeros(len(resFilePaths))

        if metricName == "Correlation":
            metEval[metricName] = []

    transMat,priors = torch.zeros((nbClass,nbClass)).float(),torch.zeros((nbClass)).float()
    transMat,_ = trainVal.computeTransMat(transMat,priors,split)

    totalFrameNb = 0

    for j,path in enumerate(resFilePaths):

        #6 ../results/cross_validation/resnet3D_split1_GS490-_6.csv
        path_split = path.split("/")[-1].split("_")
        
        if len(path_split) == 3:
            videoName = path_split[-1].replace(".csv","")
        elif len(path_split) == 4:
            videoName = path_split[-2]+"_"+path_split[-1].replace(".csv","")
        else:
            raise ValueError("Wrong path",path)
            sys.exit(0)
            
        print(videoName,path)

        metrDict,frameNb = computeMetrics(path,videoName,transMat)
        
        for metricName in metEval.keys():

            if metricName.find("Accuracy") != -1 and metricName.find("Temp") == -1:
                metEval[metricName][j] = metrDict[metricName]
                metEval[metricName][j] *= frameNb

            if metricName == "Correlation":
                metEval[metricName] += metrDict[metricName]

            if metricName == "Temp Accuracy":
                metEval[metricName][j] = metrDict[metricName]

        totalFrameNb += frameNb

    metEval["Correlation"] = np.array(metEval["Correlation"])
    metEval["Correlation"] = np.corrcoef(metEval["Correlation"][:,0],metEval["Correlation"][:,1])[0,1]
    metEval["Accuracy"] = metEval["Accuracy"].sum()/totalFrameNb
    metEval["Accuracy (Viterbi)"] = metEval["Accuracy (Viterbi)"].sum()/totalFrameNb
    metEval["Temp Accuracy"] = metEval["Temp Accuracy"].mean()
    
    for metric in metEval:
        metEval[metric] = str(metEval[metric])

    metricList = ["Correlation","Accuracy","Accuracy (Viterbi)","Temp Accuracy"]
    metric_file_path = "../results/{}/metrics.csv".format(exp_id)
    printHeader = not os.path.exists(metric_file_path)
    with open(metric_file_path,"a") as text_file:
        if printHeader:
            print("Model,"+",".join(metricList),file=text_file)

        print(model_id+","+",".join([metEval[metric] for metric in metricList]),file=text_file)
    
    print("Metrics have been written here : ",metric_file_path)

def computeMetrics(path,videoName,transMat):

    gt = load_data.getGT(videoName).astype(int)
    frameStart = (gt == -1).sum()
    gt = gt[frameStart:]

    scores = np.genfromtxt(path,delimiter=" ")[:,1:]

    gt = gt[:len(scores)]

    scores = torch.tensor(scores[np.newaxis,:]).float()
    target = torch.tensor(gt[np.newaxis,:])

    metr_dict = metrics.binaryToMetrics(scores,target,transMat,allMetrics=True,videoNames=[videoName])

    return metr_dict,len(scores)

def getLabels():
    return {"tPB2":0,"tPNa":1,"tPNf":2,"t2":3,"t3":4,"t4":5,"t5":6,"t6":7,"t7":8,"t8":9,"t9+":10,"tM":11,"tSB":12,"tB":13,"tEB":14,"tHB":15}

def plotData(nbClass=16):

    font = {'family' : 'normal',
            'weight' : 'regular',
            'size'   : 15}
    matplotlib.rc('font', **font)

    transMat = torch.zeros((nbClass,nbClass))
    priors = torch.zeros((nbClass,))
    transMat,priors = trainVal.computeTransMat(transMat,priors,1,allVid=True)

    labels = list(getLabels().keys())[:nbClass]
    labels = list(map(lambda x:x.replace("t","p"),labels))

    nbImages = len(glob.glob("../data/embryo_dataset/*/F0/*.*"))

    plt.figure()
    plt.bar(np.arange(nbClass),priors*nbImages,width=1,color='#8ebad9',edgecolor="black")
    plt.xticks(np.arange(nbClass),labels,rotation=45)
    plt.xlabel("Phases")
    plt.yticks(np.arange(1,7)*10000,[str(i*10)+"k" for i in np.arange(1,7)],rotation=45)
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig("../vis/prior.png")

def phaseNbHist():

    font = {'family' : 'normal',
            'weight' : 'regular',
            'size'   : 15}

    matplotlib.rc('font', **font)

    def countRows(x):
        x = np.genfromtxt(x,delimiter=",")
        return x.shape[0]

    #paths = glob.glob("../data/small/annotations/*phases.csv")
    #paths += glob.glob("../data/big/annotations/*phases.csv")

    vidNames = np.genfromtxt("../data/embryo_dataset_splits/split1.csv",delimiter=",",dtype=str)[:,0]
    paths = list(map(lambda x:"../data/embryo_dataset_annotations/"+x+"_phases.csv",vidNames))

    phases_nb_list = list(map(countRows,paths))

    min_val,max_val = min(phases_nb_list),max(phases_nb_list)

    plt.figure(1)
    plt.hist(phases_nb_list,alpha=0.5,density=False,bins=max_val,range=(0,max_val),edgecolor='black')
    plt.xlim(min_val,max_val)

    phaseDict = {}
    vidNb = 0
    for path in paths:
        annotation = np.genfromtxt(path,dtype=str,delimiter=",")

        if len(annotation.shape) == 2:
            vidNb += 1
            phaseList = annotation[:,0]

            for phase in phaseList:
                if phase in phaseDict:
                    phaseDict[phase] += 1
                else:
                    phaseDict[phase] = 1

    print("Total number of video",vidNb)
    phaseNames = getLabels()
    phaseNb = list(map(lambda x:phaseDict[x],phaseNames))
    phaseNames = {phase.replace("t","p"):phaseNames[phase] for phase in phaseNames.keys()}

    plt.figure(2)
    plt.bar(np.arange(len(phaseNb)),phaseNb,width=1,color='#8ebad9',edgecolor="black")

    plt.figure(1)
    plt.xticks(np.arange(min_val,max_val)+0.5,np.arange(min_val,max_val))
    plt.xlabel("Number of different phases per video")
    plt.ylabel("Number of videos")
    plt.tight_layout()
    plt.savefig("../vis/nbPhases.png")
    plt.close()

    plt.figure(2)
    plt.xticks(np.arange(16),phaseNames,rotation=45)
    plt.xlabel("Phases")
    plt.ylabel("Number of videos")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("../vis/priorsVid.png")
    plt.close()

def main(argv=None):
    argreader = ArgReader(argv)
    argreader.parser.add_argument('--model_ids',type=str,nargs="*",metavar="NAME",help='The ids of the models to process.')
    argreader.getRemainingArgs()
    args = argreader.args

    for model_id in args.model_ids:

        conf = configparser.ConfigParser()
        conf.read("../models/{}/{}.ini".format(args.exp_id,model_id))
        conf = conf["default"]

        evalModel(conf["split"],args.exp_id,model_id,nbClass=int(conf["class_nb"]))

if __name__ == "__main__":
    main()
