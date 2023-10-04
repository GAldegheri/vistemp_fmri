from mvpa2.base import dataset
import numpy as np
from .loading import load_betas, load_TRs
from .classify_models import (
    is30or90, isWideNarrow, isAllViews
)

# -------------------------------------------------------------------------------------------------

def correct_labels(DS, opt):
    '''
    Fix labels if necessary
    '''
    
    # -----------------------------------------
    if isWideNarrow(opt):
        
        wideind = [i for i, item in enumerate(DS.sa.targets) if 'wide' in item]
        narrind = [i for i, item in enumerate(DS.sa.targets) if 'narrow' in item]
        DS.sa.targets[wideind] = 'wide'
        DS.sa.targets[narrind] = 'narrow'
        
    elif is30or90(opt):
        
        rot30ind = [i for i, item in enumerate(DS.sa.targets) if '30' in item or 'near' in item]
        rot90ind = [i for i, item in enumerate(DS.sa.targets) if '90' in item or 'far' in item]
        DS.sa.targets = np.empty(DS.sa.targets.shape, dtype='<U21')
        DS.sa.targets[rot30ind] = 'rot30'
        DS.sa.targets[rot90ind] = 'rot90'
        
    elif isAllViews(opt):
        
        A30ind = [i for i, item in enumerate(DS.sa.targets) if 'A' in item and '30' in item]
        A90ind = [i for i, item in enumerate(DS.sa.targets) if 'A' in item and '90' in item]
        B30ind = [i for i, item in enumerate(DS.sa.targets) if 'B' in item and '30' in item]
        B90ind = [i for i, item in enumerate(DS.sa.targets) if 'B' in item and '90' in item]
        DS.sa.targets = np.empty(DS.sa.targets.shape, dtype='<U21')
        DS.sa.targets[A30ind] = 'A30'
        DS.sa.targets[A90ind] = 'A90'
        DS.sa.targets[B30ind] = 'B30'
        DS.sa.targets[B90ind] = 'B90'
        
    return DS

# -------------------------------------------------------------------------------------------------

def split_views(DS, opt):
    
    if (opt.task=='train' and opt.model in [7, 9]) or (opt.task=='test' and opt.model in [6, 8]):
        # decode 30 vs. 90 separately on A and B
       
        DS_A = DS[np.core.defchararray.find(DS.sa.targets,'A')!=-1]
        DS_B = DS[np.core.defchararray.find(DS.sa.targets,'B')!=-1]
        return (DS_A, DS_B)
    
    elif (opt.task=='train' and opt.model in [8, 10]) or (opt.task=='test' and opt.model in [7]):
        # decode A vs. B separately on 30 and 90
        
        DS_30 = DS[np.core.defchararray.find(DS.sa.targets,'30')!=-1]
        DS_90 = DS[np.core.defchararray.find(DS.sa.targets,'90')!=-1]
        return (DS_30, DS_90)
        
    else:
        raise Exception(f'Task: {opt.task}, Model: {opt.model} does not support separate view decoding!')

# -------------------------------------------------------------------------------------------------

def assign_loadfun(dataformat):
    if dataformat == 'TRs':
        loadfun = load_TRs
    elif dataformat == 'betas':
        loadfun = load_betas
    elif dataformat == 'trialbetas':
        raise NotImplementedError('Trial betas are not implemented yet!')
    else:
        raise ValueError(f'Data format {dataformat} unknown!')
    return loadfun