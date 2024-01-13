
from args import ArgReader
import configparser
import os,sys
import glob

import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import load_data
import metrics
import utils
import trainVal

from PIL import Image

FOCAL_PLANS = ["F-45","F-30","F-15","","F15","F30","F45"]

def get_plan_path(path,plan):

    if plan == 0:
        return path 
    else:
        plan = FOCAL_PLANS[plan+3]
        plan_path = path.replace("/embryo_dataset/",f"/embryo_dataset_{plan}/")
        
        if not os.path.exists(plan_path):
            plan_path = path

        return plan_path

def selectAndNorm(maps,ind,plan):
    map = torch.tensor(maps[ind][plan])
    map = normFunc(map)
    map = map.unsqueeze(0)  
    return map

def normFunc(map):
    return (map-map.min())/(map.max()-map.min())

def main(argv=None):
    argreader = ArgReader(argv)
    argreader.parser.add_argument('--img_nb',type=int,default=5)
    argreader.getRemainingArgs()
    args = argreader.args

    paths = sorted(glob.glob(f"../results/cross_validation/attMaps_*_{args.model_id}.npy"))

    for path in paths:

        attMaps = np.load(path,mmap_mode="r")

        if os.path.exists(path.replace("attMaps","norm")):
            norms = np.load(path.replace("attMaps","norm"),mmap_mode="r")
        else:
            norms = attMaps

        vidName = os.path.basename(path).split("attMaps_")[1].split("_"+args.model_id)[0]
        framePaths = sorted(glob.glob(f"../data/embryo_dataset/{vidName}/*.*"),key=load_data.findFrameInd)

        print(vidName)

        grid = []
        grid_img = []

        frameInds = [int(len(attMaps)*i*1.0/args.img_nb) for i in range(args.img_nb)]

        for frameInd in frameInds:

            for plan in range(-3,4):

                framePath = get_plan_path(framePaths[frameInd],plan)

                frame = torch.tensor(np.asarray(Image.open(framePath)))
                frame = (frame-frame.min())/(frame.max()-frame.min())
                frame = frame.unsqueeze(0).expand(3,-1,-1)
                frame = frame.unsqueeze(0)
                frame = torch.nn.functional.interpolate(frame,size=(224,224),mode="bicubic",align_corners=False)
                grid.append(frame)
                grid_img.append(frame)    

                if attMaps.shape[1] == 1:
                    plan = 0
                else:
                    plan += 3
                
                attMap = selectAndNorm(attMaps,frameInd,plan)
                norm = selectAndNorm(norms,frameInd,plan)
                attMap = normFunc(attMap*norm)

                attMap = torch.nn.functional.interpolate(attMap,size=(224,224),mode="bicubic",align_corners=False)
                
                attMap = 0.8*attMap+0.2*frame
                
                grid.append(attMap)

        grid = torch.cat(grid,dim=0)
        torchvision.utils.save_image(grid,f"../vis/cross_validation/attMaps_{vidName}_{args.model_id}.png",nrow=len(FOCAL_PLANS*2))

        grid_img = torch.cat(grid_img,dim=0)
        torchvision.utils.save_image(grid_img,f"../vis/cross_validation/imgs_{vidName}_{args.model_id}.png",nrow=len(FOCAL_PLANS*2))

        sys.exit(0)

if __name__ == "__main__":
    main()
