from mvpa2.base import dataset
from mvpa2.datasets import mri
import numpy as np
from glob import glob
import re
import pandas as pd
from tqdm import tqdm
from collections.abc import Iterable
import random
import os
import sys
sys.path.append('..')
from utils import loadmat
from glm.modelspec import (
    specify_model_funcloc,
    specify_model_train,
    specify_model_test
)
import configs as cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =================================================================================================
# Loading functions
# =================================================================================================

def load_betas(opt, mask_templ=None, 
               bids_dir=cfg.bids_dir, fir=False, 
               max_delay=float('inf')):
        
    betas_dir = bids_dir+'derivatives/spm-preproc/derivatives/spm-stats/betas/'
        
    datamodel = get_correct_model(opt)
    data_dir = os.path.join(betas_dir, f'{opt.sub}/{opt.task}/model_{datamodel:g}/')
    if fir:
        data_dir = os.path.join(data_dir, 'FIR')
        if not os.path.isdir(data_dir):
            raise Exception('FIR not found for this model!')
    
    SPM = loadmat(os.path.join(data_dir, 'SPM.mat'))
    if fir:
        regr_names = [n[6:] for n in SPM['SPM']['xX']['name']]
    else:
        regr_names = [n[6:-6] if '*bf(1)' in n else n[6:] for n in SPM['SPM']['xX']['name']]
    file_names = [os.path.join(data_dir, b.fname) for b in SPM['SPM']['Vbeta']]
    
    exclude = ['buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    chunk_count = {}
    for f in regr_names:
        if not f in exclude:
            chunk_count[f] = 1
    
    if mask_templ is None:
        mask_templ = os.path.join(cfg.project_dir, 'anat_roi_masks', 
                                  'wholebrain.nii')
    
    if '{:s}' in mask_templ:
        mask = mask_templ.format(opt.sub, opt.sub)
    else:
        mask = mask_templ
    
    if os.path.exists(mask):
    
        AllDS = []

        for i, f in enumerate(tqdm(file_names)):
            regr = regr_names[i]
            if not regr in exclude:
                if fir:
                    bf_n = int(re.search(r'.*bf\((\d+)\)', regr).group(1)) - 1
                    if bf_n <= max_delay:
                        # only append delays up to max_delay
                        thisDS = mri.fmri_dataset(f, chunks=chunk_count[regr], 
                                                targets=regr[:regr.find('*bf')], mask=mask)
                        thisDS.sa['delay'] = [bf_n]
                        AllDS.append(thisDS)
                else:
                    AllDS.append(mri.fmri_dataset(f, chunks=chunk_count[regr], targets=regr, mask=mask))
                chunk_count[regr] += 1

        AllDS = dataset.vstack(AllDS, a=0)

        return AllDS

    else:
        return
 
 # -------------------------------------------------------------------------------------------------

def load_TRs(opt, TR_delay=None, mask_templ=None, bids_dir=cfg.bids_dir):
    
    if not TR_delay:
        TR_delay = [6]
    if not isinstance(TR_delay, Iterable):
        TR_delay = [TR_delay]
    
    preproc = 'smooth'
    
    # MRI data:
    data_dir = bids_dir + 'derivatives/spm-preproc/{:s}/{:s}/'.format(opt.sub, preproc)
    
    allruns = glob(data_dir+'*_task-{:s}_*_bold.nii'.format(opt.task))
    
    exclude = ['buttonpress']
    
    if mask_templ is None:
        mask_templ = os.path.join(cfg.project_dir, 'anat_roi_masks', 
                                  'wholebrain.nii')
    
    if '{:s}' in mask_templ:
        mask = mask_templ.format(opt.sub, opt.sub)
    else:
        mask = mask_templ
    
    if os.path.exists(mask):
        
        AllDS = []
        
        for i, run in enumerate(allruns):
            runno = int(run.split('run-')[1][0])
            evfile = os.path.join(bids_dir,
                                  f'{opt.sub}/func/{opt.sub}_task-{opt.task}_run-{runno:g}_events.tsv')
            
            datamodel = get_correct_model(opt)
            
            if opt.task == 'funcloc':
                events = specify_model_funcloc(evfile, datamodel)
            elif opt.task == 'train':
                events = specify_model_train(evfile, datamodel)
            elif opt.task=='test':
                behav = pd.read_csv(os.path.join(bids_dir, opt.sub, 'func',
                                                 f'{opt.sub}_task-{opt.task}_beh.tsv'), 
                                    sep='\t')
                events = specify_model_test(evfile, datamodel, behav)
            
            fullrunDS = mri.fmri_dataset(run, chunks=runno, targets='placeholder', mask=mask)

            for i, cond in enumerate(events.conditions):
                if cond not in exclude:
                    # find TR indices
                    TR_indices = []
                    TR_dels = [] # to store at which delay each sample occurred (relative to onset)
                    for j in range(len(events.onsets[i])): # onsets is a list of arrays
                        for tr in TR_delay:
                            thisonset = np.floor(events.onsets[i][j]) + tr
                            TR_indices.append(int(thisonset))
                            TR_dels.append(tr)

                    # slice dataset
                    thisDS = fullrunDS[TR_indices, :]
                    thisDS.targets = np.full(thisDS.targets.shape, cond, dtype='<U21')
                    thisDS.sa['delay'] = np.array(TR_dels)
                    thisDS.sa['TRno'] = np.array(TR_indices)
                    AllDS.append(thisDS)

        AllDS = dataset.vstack(AllDS, a=0)
        AllDS.samples = AllDS.samples.astype(float)
        return AllDS  
    
    else:
        return
 
 # -------------------------------------------------------------------------------------------------

def get_correct_model(opt):
    """
    Some models are just different labelings
    of data estimated from other models.
    """
    if opt.task=='train' and opt.model==8:
        return 7
    elif opt.task=='train' and opt.model==10:
        return 9
    elif opt.task=='test' and opt.model==7:
        return 6
    else:
        return opt.model