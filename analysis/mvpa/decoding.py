from mvpa2.base import dataset
import numpy as np
from scipy.stats import pearsonr
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

def decode_viewspecific(trainDS, testDS, trainopt, testopt, split=None, thirds='none'):
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
                                                    split='both', thirds='none')
        
    elif isAllViews(trainopt) and not isAllViews(testopt): 
        
        res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                    split='train', thirds='none')
        
    elif not isAllViews(trainopt) and isAllViews(testopt):
        
        res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                    split='test', thirds='none')
        
    else:
        
        res = trainandtest_sklearn(trainDS, testDS, zscore_data=True)
    
    return res
    
# -------------------------------------------------------------------------------------------------
