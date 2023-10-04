from mvpa2.base import dataset
import numpy as np
from .core_classifiers import (
    trainandtest_sklearn, CV_leaveoneout,
    splithalfcorr
)
from .mvpa_utils import split_views
from .classify_models import (
    isWideNarrow, isAllViews, is30or90
)
import pandas as pd

# =================================================================================================
# Decoding wrapper functions
# =================================================================================================

def decode_viewspecific(trainDS, testDS, trainopt, testopt, split=None):
    '''
    split = 'train'/'test'/'both'
    thirds = 'train'/'test'/'none'
    '''
    
    if split=='train':
        
        trainDS = list(split_views(trainDS, trainopt))
        testDS = [testDS]
        
    elif split=='test':
        
        trainDS = [trainDS]
        testDS = list(split_views(testDS, testopt))
        
    elif split=='both':
       
        trainDS = list(split_views(trainDS, trainopt))
        testDS = list(split_views(testDS, testopt))
        
    else:
        raise Exception('Must specify which dataset to split! (train, test, or both)')
    
    res = []
    
    for i, tr in enumerate(trainDS):
        for j, te in enumerate(testDS):
            if i==j or split!='both': # match same view in train and test (e.g. A with A, B with B)
                
                thisres = trainandtest_sklearn(tr, te, zscore_data=True)
                
                thisres['view'] = i+1 if split=='train' else j+1 
                res.append(thisres)
    
    return pd.concat(res)

# -------------------------------------------------------------------------------------------------

def decode_traintest(trainDS, testDS, trainopt, testopt):
    
    '''
    No need to consider test model 9 here. 
    At the moment, not possible to train in a view-invariant model
    and test on a view-specific.
    '''
    
    if isAllViews(trainopt) and isAllViews(testopt):
            
        res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                    split='both')
        
    elif isAllViews(trainopt) and not isAllViews(testopt): 
        
        res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                    split='train')
        
    elif not isAllViews(trainopt) and isAllViews(testopt):
        
        res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                    split='test')
        
    else:
        
        res = trainandtest_sklearn(trainDS, testDS, zscore_data=True)
    
    return res
    
# -------------------------------------------------------------------------------------------------

def decode_CV(DS, opt):
    
    if isAllViews(opt):
        
        (DS_1, DS_2) = split_views(DS, opt)
        res_1 = CV_leaveoneout(DS_1, zscore_data=True)
        res_2 = CV_leaveoneout(DS_2, zscore_data=True)
        
        return pd.concat([res_1, res_2])
        
    else: # pure vanilla crossval
        
        res = CV_leaveoneout(DS, zscore_data=True)
        return res
    
# -------------------------------------------------------------------------------------------------

def decode_splithalf(DS, opt):
    
    res = splithalfcorr(DS)
    return res