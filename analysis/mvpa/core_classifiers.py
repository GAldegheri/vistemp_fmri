import numpy as np
from mvpa2.clfs.warehouse import SVM
from mvpa2.mappers.zscore import zscore
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner
from sklearn import svm
from scipy.stats import pearsonr
import pandas as pd
import pdb

# =================================================================================================
# Core decoding functions:
# =================================================================================================

def trainandtest(trainDS, testDS, zscore_data=True):
    
    if zscore_data:
        zscore(trainDS, chunks_attr=None)
        zscore(testDS, chunks_attr=None)
    
    clf = SVM()
    clf.train(trainDS)
    
    res = clf(testDS)
    
    #return np.column_stack((res.samples, res.targets))
    accuracy = (res.samples == res.targets).astype(int)
    outputs = (res.samples == np.unique(res.targets)[0]).astype(int)
    
    return pd.DataFrame({'correct': accuracy, 'output': outputs})

# -------------------------------------------------------------------------------------------------

def trainandtest_sklearn(trainDS, testDS, zscore_data=True):
    
    if zscore_data:
        zscore(trainDS, chunks_attr=None)
        zscore(testDS, chunks_attr=None)
    
    clf = svm.SVC(kernel='linear')
    wrapped_clf = SKLLearnerAdapter(clf)
    wrapped_clf.train(trainDS)
    
    res = wrapped_clf(testDS)
    
    accuracy = (res.samples.flatten() == res.targets.flatten()).astype(int)
    outputs = (res.samples.flatten() == np.unique(res.targets)[0]).astype(int)
    
    # distance from bound
    y = clf.decision_function(testDS)
    w_norm = np.linalg.norm(clf.coef_)
    dist = y / w_norm
    
    # normalize distance and multiply by correct direction
    zscoredist = (dist - np.mean(dist))/np.std(dist)
    neglabel = np.unique(res.samples[np.sign(zscoredist)==-1])[0]
    zscoredist[res.targets==neglabel] *= -1
    
    res = pd.DataFrame({'correct': accuracy, 'output': outputs,
                        'rawdistance': dist, 'distance': zscoredist, 
                        'runno': testDS.sa.chunks})
    
    if hasattr(testDS.sa, 'trialno'): # for trial betas
        
        res['trialno'] = testDS.sa.trialno
        
    if hasattr(testDS.sa, 'TRno'):
        
        res['TRno'] = testDS.sa.TRno
        
    if hasattr(testDS.sa, 'delay'):
        
        res['delay'] = testDS.sa.delay
        
    return res
    
# -------------------------------------------------------------------------------------------------

def CV_leaveoneout(DS, zscore_data=True):
    
    if zscore_data:
        DS = Demean_chunkwise(DS, inplace=True)
        
    CV_res = []
    for c in DS.sa.chunks:
        thistrainDS = DS[DS.sa.chunks!=c]
        thistestDS = DS[DS.sa.chunks==c]
        thisres = trainandtest_sklearn(thistrainDS,
                                       thistestDS,
                                       zscore_data=False)
        CV_res.append(thisres)
    
    CV_res = pd.concat(CV_res)
    CV_res = CV_res.sort_values(by=['runno'], ascending=True)
    
    
    return CV_res

# -------------------------------------------------------------------------------------------------

def splithalfcorr(DS):
    
    t = np.unique(DS.sa.targets)
    assert(len(t)==2)
    cond1_odd = np.mean(DS[(DS.sa.chunks%2==1) & (DS.sa.targets==t[0])], axis=0)
    cond2_odd = np.mean(DS[(DS.sa.chunks%2==1) & (DS.sa.targets==t[1])], axis=0)
    cond1_even = np.mean(DS[(DS.sa.chunks%2==0) & (DS.sa.targets==t[0])], axis=0)
    cond2_even = np.mean(DS[(DS.sa.chunks%2==0) & (DS.sa.targets==t[1])], axis=0)
    
    match_corr = pearsonr(cond1_odd, cond1_even)[0] + pearsonr(cond2_odd, cond2_even)[0]
    mismatch_corr = pearsonr(cond1_odd, cond2_even)[0] + pearsonr(cond2_odd, cond1_even)[0]
    
    return match_corr - mismatch_corr

# =================================================================================================
# Utils
# =================================================================================================

def Demean_chunkwise(DS, inplace=True):
    if inplace:
        demeanDS = DS
    else:
        demeanDS = DS.copy()
        
    for ch in np.unique(demeanDS.sa.chunks):
        demeanDS.samples[demeanDS.sa.chunks==ch] -= np.nanmean(demeanDS.samples[demeanDS.sa.chunks==ch], axis=0)
    
    return demeanDS