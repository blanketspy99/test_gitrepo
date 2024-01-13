from multiprocessing import Value
import sys
import glob
import os

import numpy as np
import torch
from torchvision import transforms
import torchvision

import args
import warnings
warnings.filterwarnings('ignore',module=".*av.*")

import logging
logging.getLogger('libav').setLevel(logging.ERROR)

import utils

import torch.distributed as dist
from random import Random

from skimage import io
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

FOCAL_PLANS = ["_F-45","_F-30","_F-15","","_F15","_F30","_F45"]

class Sampler(torch.utils.data.sampler.Sampler):
    """ The sampler for the SeqTrDataset dataset
    """

    def __init__(self, nb_videos,nb_images,seqLen):
        self.nb_videos = nb_videos
        self.nb_images = nb_images
        self.seqLen = seqLen
    def __iter__(self):

        if self.nb_images > 0:
            return iter(torch.randint(0,self.nb_videos,size=(self.nb_images//self.seqLen,)))
        else:
            return iter([])

    def __len__(self):
        return self.nb_images

def collateSeq(batch):

    res = list(zip(*batch))

    res[0] = torch.cat(res[0],dim=0)

    if not res[1][0] is None:

        res[1] = torch.cat(res[1],dim=0)

    if torch.is_tensor(res[2][0]):
        res[2] = torch.cat(res[2],dim=0)

    if torch.is_tensor(res[-1][0]):
        res[-1] = torch.cat(res[-1],dim=0)

    return res

class SeqTrDataset(torch.utils.data.Dataset):
    '''
    The dataset to sample sequence of frames from videos

    When the method __getitem__(i) is called, the dataset randomly select a sequence from the video i

    Args:
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - trLen (int): the length of a sequence during training
    - imgSize (int): the size of each side of the image
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - exp_id (str): the name of the experience
    '''

    def __init__(self,dataset,trLen,imgSize,origImgSize,preCropSize,resizeImage,exp_id,augmentData,split,all_foc_plans):

        super(SeqTrDataset, self).__init__()

        self.dataset = dataset
        self.videoPaths = findVideos("train",split)

        print("Number of training videos :",len(self.videoPaths))
        self.imgSize = imgSize
        self.trLen = trLen
        self.nbImages = 0
        self.exp_id = exp_id
        self.origImgSize = origImgSize

        for videoPath in self.videoPaths:
            nbImg = utils.getVideoFrameNb(videoPath)
            self.nbImages += nbImg

        self.resizeImage = resizeImage

        if self.resizeImage:
            self.reSizeTorchFunc = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize(preCropSize)])
        else:
            self.reSizeTorchFunc = None

        self.preproc = PreProcess(self.origImgSize,imgSize,self.resizeImage,self.reSizeTorchFunc)
        self.augmentData = augmentData
        self.all_foc_plans = all_foc_plans
    def __len__(self):
        return self.nbImages

    def __getitem__(self,vidInd):

        vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

        videoPath = self.videoPaths[vidInd]
        dataset = videoPath.split("/")[-2]+"_api"
        fold = videoPath

        video = fold

        #Computes the label index of each frame
        gt = getGT(vidName)

        frameNb = min(len(gt),utils.getVideoFrameNb(videoPath))

        frameInds = np.arange(frameNb)

        ################# Frame selection ##################
        #The video are not systematically annotated from the begining
        frameStart = (gt == -1).sum()

        try:
            frameStart = torch.randint(int(frameStart),frameNb-self.trLen,size=(1,))
        except RuntimeError:
            print(vidName,frameStart,frameNb)
            sys.exit(0)

        frameInds_slice,gt_slice = frameInds[frameStart:frameStart+self.trLen],gt[frameStart:frameStart+self.trLen]

        frameInds = frameInds_slice
        gt = gt_slice

        return loadFrames_and_process(frameInds,gt,vidName,video,self.preproc,augmentData=self.augmentData,\
                                        all_foc_plans=self.all_foc_plans,train=True)

class PreProcess():

    def __init__(self,origImgSize,finalImgSize,resizeImage,resizeTorchFunc):

        self.origImgSize = origImgSize
        self.finalImgSize = finalImgSize
        self.resizeImage = resizeImage
        self.resizeTorchFunc = resizeTorchFunc
        self.toTensorFunc = torchvision.transforms.ToTensor()
        self.normalizeFunc = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def removeTopFunc(self,x):
        #Removing the top part where the name of the video is written
        x = x[:,x.shape[1]-self.origImgSize:,:]
        return x

    def resizeFunc(self,x):
        befSize = x.shape
        if self.resizeImage:
            x = np.asarray(self.resizeTorchFunc(x.astype("uint8")))
        return x[np.newaxis,:,:,0]

    def augmenDataFunc(self,frameSeq):

        h = torch.randint(frameSeq.shape[0]-self.finalImgSize,size=(1,))[0]
        w = torch.randint(frameSeq.shape[1]-self.finalImgSize,size=(1,))[0]

        frameSeq = frameSeq[h:h+self.finalImgSize,w:w+self.finalImgSize]

        if torch.rand((1,)) > 0.5:
            frameSeq = frameSeq[::-1].copy()
        if torch.rand((1,)) > 0.5:
            frameSeq = frameSeq[:,::-1].copy()      

        return frameSeq

    def centerCrop(self,frameSeq):
        h = (frameSeq.shape[0]-self.finalImgSize)//2
        w = (frameSeq.shape[1]-self.finalImgSize)//2

        frameSeq = frameSeq[h:h+self.finalImgSize,w:w+self.finalImgSize]

        return frameSeq

class TestLoader():
    '''
    The dataset to sample sequence of frames from videos. As the video contains a great number of frame,
    each video is processed through several batches and each batch contain only one sequence.

    Args:
    - evalL (int): the length of a sequence in a batch. A big value will reduce the number of batches necessary to process a whole video
    - dataset (str): the name of the dataset
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - imgSize (tuple): a tuple containing (in order) the width and size of the image
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - exp_id (str): the name of the experience
    '''

    def __init__(self,dataset,evalL,imgSize,origImgSize,preCropSize,resizeImage,exp_id,mode,split,debug,all_foc_plans):
        self.dataset = dataset
        self.evalL = evalL

        self.videoPaths = findVideos(mode,split)

        if debug:
            self.videoPaths = self.videoPaths[:3]

        self.exp_id = exp_id
        print("Number of eval videos :",len(self.videoPaths))

        self.origImgSize = origImgSize
        self.imgSize = imgSize
        self.resizeImage = resizeImage
        self.nbImages = 0
        for videoPath in self.videoPaths:
            self.nbImages += utils.getVideoFrameNb(videoPath)

        if self.resizeImage:
            self.reSizeTorchFunc = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize(preCropSize)])
        else:
            self.reSizeTorchFunc = None

        self.transf = None

        self.preproc = PreProcess(self.origImgSize,imgSize,self.resizeImage,self.reSizeTorchFunc)
        self.all_foc_plans = all_foc_plans
    def __iter__(self):
        self.videoInd = 0
        self.currFrameInd = None
        self.sumL = 0
        return self

    def __next__(self):

        if self.videoInd == len(self.videoPaths):
            raise StopIteration

        L = self.evalL
        self.sumL += L

        videoPath = self.videoPaths[self.videoInd]

        video = videoPath
       
        vidName = os.path.basename(os.path.splitext(videoPath)[0])

        frameNb = min(len(getGT(vidName)),utils.getVideoFrameNb(videoPath))

        if self.currFrameInd is None:
            #The video are not systematically annotated from the begining
            gt = getGT(vidName)
            frameStart = (gt == -1).sum()
            self.currFrameInd = int(frameStart)

        frameInds = np.arange(self.currFrameInd,min(self.currFrameInd+L,frameNb))

        gt = getGT(vidName)[self.currFrameInd:min(self.currFrameInd+L,frameNb)]

        if frameInds[-1] + 1 == frameNb:
            self.currFrameInd = None
            self.videoInd += 1
        else:
            self.currFrameInd += L

        return loadFrames_and_process(frameInds,gt,vidName,video,self.preproc,all_foc_plans=self.all_foc_plans,train=False)

def findFrameInd(path):
    fileName = os.path.splitext(os.path.basename(path))[0]
    frameInd = int(fileName.split("RUN")[1])
    return frameInd

def loadFrames_and_process(frameInds,gt,vidName,video,preproc,augmentData=False,all_foc_plans=False,train=False):

    #Building the frame sequence, remove the top of the video (if required)
    if all_foc_plans and train:
        focal_plans = [FOCAL_PLANS[np.random.randint(0,len(FOCAL_PLANS))]]
    elif all_foc_plans and not train:
        focal_plans = FOCAL_PLANS
    else:
        focal_plans = [""]

    allPlansImgs = []

    for plan in focal_plans:

        if plan != "":
            video_foc_plan = video.replace("embryo_dataset","embryo_dataset"+plan)
        else:
            video_foc_plan = video 

        allFrames = sorted(glob.glob(video_foc_plan+"/*.*"),key=findFrameInd)
        
        try:
            frameSeq = [allFrames[i] for i in frameInds]
        except IndexError:
            frameSeq = [allFrames[i] for i in range(frameInds[0],len(allFrames))]

        #frameSeq = np.concatenate(list(map(preproc.removeTopFunc,map(lambda x:cv2.imread(x)[np.newaxis],np.array(frameSeq)))),axis=0)
        frameSeq = np.concatenate(list(map(preproc.removeTopFunc,map(lambda x:io.imread(x)[np.newaxis,:,:,np.newaxis].repeat(3,-1),np.array(frameSeq)))),axis=0)

        #Resize the images (if required)
        frameSeq = np.concatenate(list(map(preproc.resizeFunc,frameSeq)),axis=0)

        #Those few lines of code convert the numpy array into a torch tensor, normalize them and apply transformations
        # Shape of tensor : T x H x W
        frameSeq = frameSeq.transpose((1,2,0))
        # H x W x T

        if augmentData:
            frameSeq = preproc.augmenDataFunc(frameSeq)
        else:
            frameSeq = preproc.centerCrop(frameSeq)

        frameSeq = preproc.toTensorFunc(frameSeq)
        # T x H x W
        frameSeq = frameSeq.unsqueeze(1)
        # T x 1 x H x W
        frameSeq = frameSeq.expand(frameSeq.size(0),3,frameSeq.size(2),frameSeq.size(3))
        # T x 3 x H x W
        frameSeq = frameSeq.unsqueeze(1)
        # T x 1 x 3 x H x W

        allPlansImgs.append(frameSeq)

    frameSeq = torch.cat(allPlansImgs,dim=1)
    # T x P x 3 x H x W

    return frameSeq.unsqueeze(0),torch.tensor(gt).unsqueeze(0),vidName,torch.tensor(frameInds).int()

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(dataset):

    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

def buildSeqTrainLoader(args):

    train_dataset = SeqTrDataset(args.dataset_train,args.tr_len,args.img_size,args.orig_img_size,args.pre_crop_size,\
                                    args.resize_image,args.exp_id,args.augment_data,\
                                    args.split,args.all_foc_plans)

    sampler = Sampler(len(train_dataset.videoPaths),train_dataset.nbImages,args.tr_len)
    collateFn = collateSeq
    kwargs = {"sampler":sampler,"collate_fn":collateFn}

    bsz = args.batch_size

    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=bsz, # use custom collate function here
                      pin_memory=False,num_workers=args.num_workers,**kwargs)

    return trainLoader,train_dataset

def buildSeqTestLoader(args,mode,normalize=True):

    datasetName = getattr(args,"dataset_{}".format(mode))

    testLoader = TestLoader(datasetName,args.val_l,\
                                    args.img_size,args.orig_img_size,args.pre_crop_size,args.resize_image,\
                                    args.exp_id,\
                                    mode,args.split,args.debug,args.all_foc_plans)

    return testLoader

def removeVid(videoPaths,videoToRemoveNames):
    #Removing videos with bad format
    vidsToRemove = []
    for vidPath in videoPaths:
        for vidName in videoToRemoveNames:
            if os.path.splitext(os.path.basename(vidPath))[0] == vidName:
                vidsToRemove.append(vidPath)
    for vidPath in vidsToRemove:
        videoPaths.remove(vidPath)

    return videoPaths

def findVideos(mode,split):
    videos_split = np.genfromtxt("../data/embryo_dataset_splits/split{}.csv".format(split),delimiter=",",dtype=str)
    videos_to_use = videos_split[:,0][videos_split[:,1] ==  mode]
    videoPaths = list(map(lambda x:"../data/embryo_dataset/"+x,videos_to_use))
    return videoPaths

def getLabels():
    return {"tPB2":0,"tPNa":1,"tPNf":2,"t2":3,"t3":4,"t4":5,"t5":6,"t6":7,"t7":8,"t8":9,"t9+":10,"tM":11,"tSB":12,"tB":13,"tEB":14,"tHB":15}

def getGT(vidName):
    ''' For one video, returns the label of each frame

    Args:
    - vidName (str): the video name. It is the name of the video file minus the extension.
    Returns:
    - gt (array): the list of labels corresponding to each image

    '''

    if not os.path.exists("../data/embryo_dataset_annotations/{}_targ.csv".format(vidName)):

        phases = np.genfromtxt("../data/embryo_dataset_annotations/{}_phases.csv".format(vidName),dtype=str,delimiter=",")

        gt = np.zeros((int(phases[-1,-1])+1))-1

        for phase in phases:
            #The dictionary called here convert the label into a integer

            gt[int(phase[1]):int(phase[2])+1] = getLabels()[phase[0]]

        np.savetxt("../data/embryo_dataset_annotations/{}_targ.csv".format(vidName),gt)
    else:
        gt = np.genfromtxt("../data/embryo_dataset_annotations/{}_targ.csv".format(vidName))

    return gt.astype(int)

def isVideo(name):

    i=0
    datasetPaths = ["../data/big/","../data/small"]
    isVideo=False

    while i < len(datasetPaths) and not isVideo:
        if os.path.exists(os.path.join(datasetPaths[i],name+".avi")):
            isVideo = True
        i+=1
    
    return isVideo

def getDataset(videoName):


    videoDatasetPath = None

    i=0
    datasetPaths = ["../data/big/","../data/small/"]
    datasetFound=False

    while i < len(datasetPaths) and not datasetFound:
        if os.path.exists(os.path.join(datasetPaths[i],videoName+".avi")):
            videoDatasetPath = datasetPaths[i]
            datasetFound = True
        i+=1

    if videoDatasetPath is None:
        raise ValueError("No dataset found for ",videoName)

    datasetName = videoDatasetPath.split("/")[-2]

    return datasetName

def addArgs(argreader):

    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                        help='The network producing the features can only be pretrained on \'imageNet\'. This argument must be \
                            set to \'imageNet\' datasets.')
    argreader.parser.add_argument('--batch_size', type=args.str2int,metavar='BS',
                        help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=args.str2int,metavar='BS',
                        help='The batchsize to use for validation')

    argreader.parser.add_argument('--tr_len', type=args.str2int,metavar='LMAX',
                        help='The maximum length of a training sequence')
    argreader.parser.add_argument('--val_l', type=args.str2int,metavar='LMAX',
                        help='Length of sequences for validation.')

    argreader.parser.add_argument('--img_size', type=args.str2int,metavar='EDGE_SIZE',
                        help='The size of each edge of the images after resizing, if resizing is desired. Else is should be equal to --orig_img_size')
    argreader.parser.add_argument('--orig_img_size', type=args.str2int,metavar='EDGE_SIZE',
                        help='The size of each edge of the images before preprocessing.')
    argreader.parser.add_argument('--pre_crop_size', type=args.str2int,metavar='EDGE_SIZE',
                        help='Images will be resized to this size before a random/center crop of size --img_size is extracted')

    argreader.parser.add_argument('--train_part_beg', type=args.str2float,metavar='START',
                        help='The start position of the train set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.\
                        Ignored when video_mode is False.')
    argreader.parser.add_argument('--train_part_end', type=args.str2float,metavar='END',
                        help='The end position of the train set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.\
                        Ignored when video_mode is False.')
    argreader.parser.add_argument('--val_part_beg', type=args.str2float,metavar='START',
                        help='The start position of the validation set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.\
                        Ignored when video_mode is False.')
    argreader.parser.add_argument('--val_part_end', type=args.str2float,metavar='END',
                        help='The end position of the validation set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.\
                        Ignored when video_mode is False.')
    argreader.parser.add_argument('--test_part_beg', type=args.str2float,metavar='START',
                        help='The start position of the test set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.\
                        Ignored when video_mode is False.')
    argreader.parser.add_argument('--test_part_end', type=args.str2float,metavar='END',
                        help='The end position of the test set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.\
                        Ignored when video_mode is False.')

    argreader.parser.add_argument('--prop_set_int_fmt', type=args.str2bool,metavar='BOOL',
                        help='Set to True to set the sets (train, validation and test) proportions\
                            using int between 0 and 100 instead of float between 0 and 1.')

    argreader.parser.add_argument('--dataset_train', type=str,metavar='DATASET',
                        help='The dataset for training. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_val', type=str,metavar='DATASET',
                        help='The dataset for validation. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_test', type=str,metavar='DATASET',
                        help='The dataset for testing. Can be "big" or "small"')

    argreader.parser.add_argument('--resize_image', type=args.str2bool, metavar='S',
                        help='to resize the image to the size indicated by the img_width and img_heigth arguments.')

    argreader.parser.add_argument('--class_nb', type=args.str2int, metavar='S',
                        help='The number of class of to model')

    argreader.parser.add_argument('--augment_data', type=args.str2bool, metavar='S',
                        help='Set to True to augment the training data with transformations')

    argreader.parser.add_argument('--split', type=int, metavar='S',
                        help='The split index for cross validation. Should be between 1 and 8 included.')

    argreader.parser.add_argument('--debug', action="store_true")

    argreader.parser.add_argument('--all_foc_plans', action="store_true")

    return argreader
