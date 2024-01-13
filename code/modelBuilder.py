import sys
import torch
from torch import nn
from torch.nn import functional as F
import resnet
import resnet3D
import args

def buildFeatModel(featModelName,pretrainedFeatMod,featMap=False,bigMaps=False,layerSizeReduce=False,stride=2,dilation=1,**kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("resnet") != -1:
        featModel = getattr(resnet,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,layerSizeReduce=layerSizeReduce,**kwargs)
    elif featModelName == "r2plus1d_18":
        featModel = getattr(resnet3D,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,bigMaps=bigMaps)
    else:
        raise ValueError("Unknown model type : ",featModelName)

    return featModel

#This class is just the class nn.DataParallel that allow running computation on multiple gpus
#but it adds the possibility to access the attribute of the model
class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super(DataParallelModel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(DataParallelModel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Model(nn.Module):

    def __init__(self,firstModel,secondModel,spatTransf=None):
        super(Model,self).__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel
        self.spatTransf = spatTransf

        self.transMat = torch.zeros((self.secondModel.nbClass,self.secondModel.nbClass))
        self.priors = torch.zeros((self.secondModel.nbClass))

    def forward(self,x):
        if self.spatTransf:
            x = self.spatTransf(x)["x"]

        visResDict = self.firstModel(x)
        x = visResDict["x"]

        resDict = self.secondModel(x,self.firstModel.batchSize)

        for key in visResDict.keys():
            resDict[key] = visResDict[key]

        return resDict

    def computeVisual(self,x):
        if self.spatTransf:
            resDict = self.spatTransf(x)
            x = resDict["x"]
            theta = resDict["theta"]

        resDict = self.firstModel(x)

        if self.spatTransf:
            resDict["theta"] = theta
        return resDict

    def setTransMat(self,transMat):
        self.transMat = transMat
    def setPriors(self,priors):
        self.priors = priors

################################# Visual Model ##########################

class FirstModel(nn.Module):

    def __init__(self,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False,**kwargs):
        super(FirstModel,self).__init__()

        self.featMod = buildFeatModel(featModelName,pretrainedFeatMod,featMap,bigMaps,**kwargs)
        self.featMap = featMap
        self.bigMaps = bigMaps

    def forward(self,x):
        raise NotImplementedError

class CNN2D(FirstModel):

    def __init__(self,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False,**kwargs):
        super(CNN2D,self).__init__(featModelName,pretrainedFeatMod,featMap,bigMaps,**kwargs)

    def forward(self,x):
        # N x T x P x C x H x L

        origSize = x.size()

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1)*x.size(2),x.size(3),x.size(4),x.size(5)).contiguous()
        # NTP x C x H x L
        feat = self.featMod(x)

        # NTP x D
        feat = feat.view(origSize[0]*origSize[1],origSize[2],feat.size(-1))
        # NT x P x D
        feat = feat.mean(dim=1)
        # NT x D

        return {'x':feat}

class CNN3D(FirstModel):

    def __init__(self,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False):
        super(CNN3D,self).__init__(featModelName,pretrainedFeatMod,featMap,bigMaps)

    def forward(self,x):
        # N x T x P x C x H x L
        self.batchSize = x.size(0)
        x = x.permute(0,2,3,1,4,5)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4),x.size(5))
        # NP x C x T x H x L
        x = self.featMod(x)

        if self.featMap:
            # NP x D x T x H x L
            x = x.permute(0,2,1,3,4)
            # NP x T x D x H x L
            x = x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
            # NPT x D x H x L
        else:
            # NP x D x T
            x = x.permute(0,2,1)
            # NP x T x D
            x = x.contiguous().view(x.size(0)*x.size(1),-1)
            # NPT x D
        return {'x':x}

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self,nbFeat,nbClass):
        super(SecondModel,self).__init__()
        self.nbFeat,self.nbClass = nbFeat,nbClass

    def forward(self,x):
        raise NotImplementedError

class LinearSecondModel(SecondModel):

    def __init__(self,nbFeat,nbClass,dropout,bias):
        super(LinearSecondModel,self).__init__(nbFeat,nbClass)
        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(nbFeat,self.nbClass,bias=bias)

    def forward(self,x,batchSize):

        # NPT x D
        x = self.dropout(x)
        x = self.linLay(x)
        # NPT x classNb
        x = x.view(batchSize,-1,self.nbClass)
        #NP x T x classNb
        return {"pred":x}

class LSTMSecondModel(SecondModel):

    def __init__(self,nbFeat,nbClass,dropout,nbLayers,nbHidden):
        super(LSTMSecondModel,self).__init__(nbFeat,nbClass)

        self.lstmTempMod = nn.LSTM(input_size=self.nbFeat,hidden_size=nbHidden,num_layers=nbLayers,batch_first=True,dropout=dropout,bidirectional=True)
        self.linTempMod = LinearSecondModel(nbFeat=nbHidden*2,nbClass=self.nbClass,dropout=dropout)

    def forward(self,x,batchSize):

        self.lstmTempMod.flatten_parameters()

        # NT x D
        x = x.view(batchSize,-1,x.size(-1))
        # N x T x D
        x,_ = self.lstmTempMod(x)
        # N x T x H
        x = x.contiguous().view(-1,x.size(-1))
        # NT x H
        x = self.linTempMod(x,batchSize)["pred"]
        # N x T x classNb
        return {"pred":x}

def getResnetFeat(backbone_name,backbone_inplanes):

    if backbone_name=="resnet50" or backbone_name=="resnet101" or backbone_name=="resnet151":
        nbFeat = backbone_inplanes*4*2**(4-1)
    elif backbone_name.find("resnet18") != -1:
        nbFeat = backbone_inplanes*2**(4-1)
    elif backbone_name.find("resnet14") != -1:
        nbFeat = backbone_inplanes*2**(3-1)
    elif backbone_name.find("resnet9") != -1:
        nbFeat = backbone_inplanes*2**(2-1)
    elif backbone_name.find("resnet4") != -1:
        nbFeat = backbone_inplanes*2**(1-1)
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def netBuilder(args):

    ############### Visual Model #######################
    if args.feat.find("resnet") != -1:
        nbFeat = getResnetFeat(args.feat,args.resnet_chan)

        CNNconst = CNN2D
        kwargs = {}

        kwargs.update({"preLayerSizeReduce":args.resnet_prelay_size_reduce,"layerSizeReduce":args.resnet_layer_size_reduce})

        firstModel = CNNconst(args.feat,args.pretrained_visual,chan=args.resnet_chan,stride=args.resnet_stride,dilation=args.resnet_dilation,\
                                    num_classes=args.class_nb,**kwargs)
    elif args.feat == "r2plus1d_18":
        nbFeat = 512
        firstModel = CNN3D(args.feat,args.pretrained_visual)
    else:
        raise ValueError("Unknown visual model type : ",args.feat)

    if args.freeze_visual:
        for param in firstModel.parameters():
            param.requires_grad = False

    ############### Temporal Model #######################
    if args.temp_mod == "lstm":
        secondModel = LSTMSecondModel(nbFeat,args.class_nb,args.dropout,args.lstm_lay,args.lstm_hid_size)
    elif args.temp_mod == "linear":
        secondModel = LinearSecondModel(nbFeat,args.class_nb,args.dropout,bias=True)
    else:
        raise ValueError("Unknown temporal model type : ",args.temp_mod)

    ############### Whole Model ##########################

    net = Model(firstModel,secondModel)

    if args.cuda:
        net.cuda()

    if args.multi_gpu:
        net = DataParallelModel(net)

    return net

def addArgs(argreader):

    argreader.parser.add_argument('--feat', type=str, metavar='MOD',
                        help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=args.str2float,metavar='D',
                        help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--temp_mod', type=str,metavar='MOD',
                        help='The temporal model. Can be "linear", "lstm" or "score_conv".')

    argreader.parser.add_argument('--lstm_lay', type=args.str2int,metavar='N',
                        help='Number of layers for the lstm temporal model')

    argreader.parser.add_argument('--lstm_hid_size', type=args.str2int,metavar='N',
                        help='Size of hidden layers for the lstm temporal model')

    argreader.parser.add_argument('--freeze_visual', type=args.str2bool, metavar='BOOL',
                        help='To freeze the weights of the visual model during training.',default=False)

    argreader.parser.add_argument('--pretrained_visual', type=args.str2bool, metavar='BOOL',
                        help='To have a visual feature extractor pretrained on ImageNet.')

    argreader.parser.add_argument('--resnet_chan', type=args.str2int, metavar='INT',
                        help='The channel number for the visual model when resnet is used',default=64)
    argreader.parser.add_argument('--resnet_stride', type=args.str2int, metavar='INT',
                        help='The stride for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_dilation', type=args.str2int, metavar='INT',
                        help='The dilation for the visual model when resnet is used')

    argreader.parser.add_argument('--resnet_layer_size_reduce', type=args.str2bool, metavar='INT',
                        help='To apply a stride of 2 in the layer 2,3 and 4 when the resnet model is used.')
    argreader.parser.add_argument('--resnet_prelay_size_reduce', type=args.str2bool, metavar='INT',
                        help='To apply a stride of 2 in the convolution and the maxpooling before the layer 1.')


    return argreader