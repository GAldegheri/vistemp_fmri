from mvpa2.base import dataset
import numpy as np

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
  