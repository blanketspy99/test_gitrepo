import numpy as np
import torch
import torch
import torch.nn.functional as F
import load_data

phasesSTDs = {"tPNa": 1.13,"tPNf":0.50,"t2":0.91,"t3":1.81,"t4":1.34,"t5":1.49,"t6":1.61,"t7":2.93,"t8":5.36,"t9+":4.42,"tM": 5.46,"tSB":3.78,"tB":3.29,"tEB":4.85,"tHB":15}
labelDict = {"tPB2":0,"tPNa":1,"tPNf":2,"t2":3,"t3":4,"t4":5,"t5":6,"t6":7,"t7":8,"t8":9,"t9+":10,"tM":11,"tSB":12,"tB":13,"tEB":14,"tHB":15}
revLabDict = {labelDict[key]:key for key in labelDict.keys()}
# Code taken from https://gist.github.com/PetrochukM/afaa3613a99a8e7213d2efdd02ae4762#file-top_k_viterbi-py-L5
# Credits to AllenNLP for the base implementation and base tests:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L174

# Modified AllenNLP `viterbi_decode` to support `top_k` sequences efficiently.
def viterbi_decode(tag_sequence,transition_matrix,top_k=1):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    top_k : int, required.
        Integer defining the top number of paths to decode.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : float
        The score of the viterbi path.
    """

    transition_matrix = transition_matrix.to(tag_sequence.device)

    sequence_length, num_tags = list(tag_sequence.size())

    path_scores = []
    path_indices = []
    # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
    # to allow for 1 permutation.
    path_scores.append(tag_sequence[0, :].unsqueeze(0))
    # assert path_scores[0].size() == (n_permutations, num_tags)

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        # assert scores.size() == (n_permutations, num_tags)
        # assert paths.size() == (n_permutations, num_tags)

        scores = tag_sequence[timestep, :] + scores
        # assert scores.size() == (n_permutations, num_tags)
        path_scores.append(scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores = path_scores[-1].view(-1)
    max_k = min(path_scores.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    return viterbi_paths, viterbi_scores

def emptyMetrDict(allMetrics=True):
    if allMetrics:
        return {"Loss":0,"Accuracy":0,"Accuracy (Viterbi)":0,"Correlation":0,"Temp Accuracy":0}
    else:
        return {"Loss":0,"Accuracy":0,"Accuracy (Viterbi)":0}

def updateMetrDict(metrDict,metrDictSample):

    for metric in metrDict.keys():

        if metric in list(metrDictSample.keys()):
            if metric.find("Entropy") == -1:
                metrDict[metric] += metrDictSample[metric]
            else:
                if metrDict[metric] is None:
                    metrDict[metric] = metrDictSample[metric]
                else:
                    metrDict[metric] = torch.cat((metrDict[metric],metrDictSample[metric]),dim=0)

    return metrDict

def binaryToMetrics(output,target,transition_matrix=None,allMetrics=False,videoNames=None):

    pred = output.argmax(dim=-1)
    acc = accuracy(pred,target)

    pred_vit = viterbi(output,target,transition_matrix)
    
    accViterb = accuracy(pred_vit,target)

    metDict = {"Accuracy":acc,'Accuracy (Viterbi)':accViterb}

    if allMetrics:
        metDict["Correlation"],metDict["Temp Accuracy"] = correlation(pred,target,videoNames)

    return metDict

def accuracy(pred,target):
    return (pred == target).float().sum()/(pred.numel())

def viterbi(output,target,transition_matrix):
    pred = []
    for outSeq in output:
        outSeq = torch.nn.functional.softmax(outSeq, dim=-1)
        predSeqs,_ = viterbi_decode(torch.log(outSeq),torch.log(transition_matrix),top_k=1)
        pred.append(torch.tensor(predSeqs[0]).unsqueeze(0))
    pred = torch.cat(pred,dim=0).to(target.device)
    return pred

def correlation(predBatch,target,videoNames,threshold=1):
    ''' Computes the times at which the model predicts the developpement phase is changing and
    compare it to the real times where the phase is changing. Computes a correlation between those
    two list of numbers.

    '''

    for i,pred in enumerate(predBatch):

        timeElapsedTensor = np.genfromtxt("../data/embryo_dataset_time_elapsed/{}_timeElapsed.csv".format(videoNames[i]),delimiter=",")[1:,1]

        phasesPredDict = phaseToTime(pred,timeElapsedTensor)
        phasesTargDict = phaseToTime(target[i],timeElapsedTensor)

        commonPhases = list(set(list(phasesPredDict.keys())).intersection(set(list(phasesTargDict.keys()))))

        timePairs = []
        accuracy = 0
        for phase in commonPhases:
            if phase != 0:
                timePairs.append((phasesPredDict[phase],phasesTargDict[phase]))
                if np.abs(phasesPredDict[phase]-phasesTargDict[phase]) <= threshold*phasesSTDs[revLabDict[phase]]:
                    accuracy +=1
        accuracy /= len(phasesTargDict.keys()) -1
  
        return timePairs,accuracy

def phaseToTime(phaseList,timeElapsedTensor):
    changingPhaseFrame = np.concatenate(([1],(phaseList[1:]-phaseList[:-1]) > 0),axis=0)
    phases = phaseList[np.argwhere(changingPhaseFrame)[:,0]]

    changingPhaseFrame = np.argwhere(changingPhaseFrame)[:,0]
    changingPhaseTime = timeElapsedTensor[changingPhaseFrame]

    phaseToFrameDict = {phases[i].item():changingPhaseTime[i] for i in range(len(phases))}

    return phaseToFrameDict
