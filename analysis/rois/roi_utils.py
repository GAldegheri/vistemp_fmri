import nibabel as nb
from nilearn.image import new_img_like
import numpy as np
import sys
sys.path.append('..')
from configs import project_dir, bids_dir
import os
import copy
from nipype.interfaces.matlab import MatlabCommand
from configs import project_dir, spm_dir, matlab_cmd, bids_dir
MatlabCommand.set_default_matlab_cmd(matlab_cmd)
MatlabCommand.set_default_paths(spm_dir)
from nipype.interfaces.spm.utils import Reslice
from collections.abc import Iterable
from tqdm import tqdm
import pdb

# Useful models/contrasts in functional localizer:
# - Model 1, contrast 1: object vs. scrambled
# - Model 2, contrast 1: stimulus vs. baseline
# - Model 3, contrast 1: object/scrambled vs. baseline

roidir = os.path.join(project_dir, 'anat_roi_masks')

roi_dict = {'ba-17': {'anat_mask': os.path.join(roidir, 'ba-17.nii'), 
                      'task': 'funcloc',
                      'model': 3,
                      'contrast': 1,
                      'out_file_templ': '{:s}_ba-17_contr-objscrvsbas_top-{:g}.nii'},
            
            'ba-18': {'anat_mask': os.path.join(roidir, 'ba-18.nii'),
                      'task': 'funcloc',
                      'model': 3,
                      'contrast': 1,
                      'out_file_templ': '{:s}_ba-18_contr-objscrvsbas_top-{:g}.nii'},
            
            'ba-19': {'anat_mask': os.path.join(roidir, 'ba-19.nii'),
                      'task': 'funcloc',
                      'model': 3,
                      'contrast': 1,
                      'out_file_templ': '{:s}_ba-19_contr-objscrvsbas_top-{:g}.nii'},
            
            'ba-17-18': {'anat_mask': os.path.join(roidir, 'ba-17-18.nii'),
                        'task': 'funcloc',
                        'model': 3,
                        'contrast': 1,
                        'out_file_templ': '{:s}_ba-17-18_contr-objscrvsbas_top-{:g}.nii'},
            
            'LO': {'anat_mask': os.path.join(roidir, 'LO.nii'),
                        'task': 'funcloc',
                        'model': 1,
                        'contrast': 1,
                        'out_file_templ': '{:s}_LO_contr-objvscr_top-{:g}.nii'},
            
            'ba-19-37': {'anat_mask': os.path.join(roidir, 'ba-19-37.nii'),
                        'task': 'funcloc',
                        'model': 3,
                        'contrast': 1,
                        'out_file_templ': '{:s}_ba-19-37_contr-objscrvsbas_top-{:g}.nii'},
            
            'PPA': {'anat_mask': os.path.join(roidir, 'PPA.nii'),
                    'task': 'funcloc',
                    'model': 1,
                    'contrast': 6,
                    'out_file_templ': '{:s}_PPA_contr-scenevsobj_top-{:g}.nii'}}

def create_functional_roi(sub, roiname, nvoxels='all', 
                          tthresh=1.96, split_lr=False, out_dir=None):
    """
    Create ROI based on functional (t) contrast.
    """
    if not (isinstance(nvoxels, str) and nvoxels == 'all') and not \
        isinstance(nvoxels, Iterable):
        
        nvoxels = [nvoxels]
        
    task = roi_dict[roiname]['task']
    model = roi_dict[roiname]['model']
    contrast = roi_dict[roiname]['contrast']
    
    if out_dir is None:
        out_dir = os.path.join(bids_dir, 'derivatives', 'spm-preproc',
                               'derivatives', 'roi-masks', sub)
    
    out_file_templ = os.path.join(out_dir, roi_dict[roiname]['out_file_templ'])
    if (isinstance(nvoxels, str) and nvoxels == 'all'):
        out_file_templ = out_file_templ.replace('_top-{:g}', '_allsignif')
    
    if split_lr:
        out_file_templ = out_file_templ.replace('_contr', '_{:s}_contr')
    
    anat_mask = roi_dict[roiname]['anat_mask']
    tmap_file = os.path.join(bids_dir, 'derivatives', 'spm-preproc',
                             'derivatives', 'spm-stats',
                             'contrasts', sub, task, f'model_{model}',
                             f'spmT_{contrast:04d}.nii')
    brodmann_mask = nb.load(anat_mask).get_fdata()
    
    contr_vol = nb.load(tmap_file)
    contr_data = contr_vol.get_fdata()
    
    contr_data[brodmann_mask==0] = 0
    contr_vol = new_img_like(contr_vol, contr_data)
    
    if not (isinstance(nvoxels, str) and nvoxels == 'all'):
        if split_lr:
            contr_L, contr_R = split_hemispheres(contr_vol)
            top_voxels = [(cl, cr) for cl, cr in zip(
                pick_top_voxels(contr_L, nvoxels), 
                pick_top_voxels(contr_R, nvoxels))]
        else:
            top_voxels = pick_top_voxels(contr_vol, nvoxels)
        
        # save files 
        for tv, nv in zip(top_voxels, nvoxels):
            if split_lr:
                for side, hemi in zip(tv, ['L', 'R']):
                    min_value = np.min(side.get_fdata()[np.nonzero(side.get_fdata())])
                    if min_value >= tthresh:
                        # save file
                        out_file = out_file_templ.format(sub, hemi, nv)
                        side = binarize(side)
                        side.to_filename(out_file)
                    else:
                        print(f'Less than {nv} significant voxels for '+\
                            f'{sub}, {roiname} - {hemi}.')
            else:
                min_value = np.min(tv.get_fdata()[np.nonzero(tv.get_fdata())])
                if min_value >= tthresh:
                    # save file
                    out_file = out_file_templ.format(sub, nv)
                    tv = binarize(tv)
                    tv.to_filename(out_file)
                else:
                    print(f'Less than {nv} significant voxels for '+\
                        f'{sub}, {roiname} (both hemispheres).')
    else:
        if split_lr:
            contr_L, contr_R = (above_threshold(c) for c in
                split_hemispheres(contr_vol, tthresh))
            contr_L.to_filename(out_file_templ.format(sub, 'L'))
            contr_R.to_filename(out_file_templ.format(sub, 'R'))
        else:
            contr_vol = above_threshold(contr_vol, tthresh)
            contr_vol.to_filename(out_file_templ.format(sub))

def binarize(vol):
    return new_img_like(vol, vol.get_fdata() > 0)

def above_threshold(vol, thresh):
    return new_img_like(vol, vol.get_fdata() >= thresh)        
        
def pick_top_voxels(vol, nvoxels, binary=False):
    
    voldata = vol.get_fdata()
    voldata_sorted = -np.sort(-voldata.flatten()) # descending order
    
    top_voxels = []
    for n in nvoxels:
        topnthresh = voldata_sorted[n]
        this_thresh = voldata.copy()
        this_thresh[this_thresh<topnthresh] = 0
        this_thresh = new_img_like(vol, this_thresh)
        if binary:
            this_thresh = binarize(this_thresh)
        top_voxels.append(this_thresh)
    
    return top_voxels
        

def create_brodmann_roi(ba_number):
    """
    Create ROI based only on anatomical (Brodmann) atlas.
    """
    if not isinstance(ba_number, list):
        ba_number = [ba_number]
    
    vol = nb.load(os.path.join(roidir, 'brodmann.nii'))
    vol_data = vol.get_fdata()
    
    ba_img = new_img_like(vol, np.isin(vol_data, ba_number))
    
    return ba_img


def split_hemispheres(vol):
    
    if isinstance(vol, str):
        vol = nb.load(vol)
        
    vol_data = vol.get_fdata()
    
    mask_L = np.zeros(vol_data.shape)
    mask_L[41:, :, :] = 1
    mask_R = np.zeros(vol_data.shape)
    mask_R[:40, :, :] = 1
    
    vol_L = copy.deepcopy(vol_data)
    vol_L[mask_L == 0] = 0 
    vol_R = copy.deepcopy(vol_data)
    vol_R[mask_R == 0] = 0
    
    vol_L = new_img_like(vol, vol_L)
    vol_R = new_img_like(vol, vol_R)
    
    return vol_L, vol_R
    

def reslice_spm(in_file, out_file=None):        
    
    resl = Reslice()
    resl.inputs.in_file = in_file
    resl.inputs.space_defining = os.path.join(roidir, 'ba-17-18.nii')
    reslicedfile = resl.run()
    
    # Replace original file unless otherwise specified
    if out_file is None:
        os.rename(reslicedfile, in_file) 
    else:
        os.rename(reslicedfile, out_file)
    
if __name__=="__main__":
    nvoxels = np.arange(100, 1100, 100)
    allsubjs = [f'sub-{i:03d}' for i in range(1, 35)]
    
    roimap = create_brodmann_roi([17, 18, 19, 37])
    roimap_L, roimap_R = split_hemispheres(roimap)
    nb.save(roimap_L, os.path.join(roidir, 'ba-17-18-19-37_L.nii'))
    nb.save(roimap_R, os.path.join(roidir, 'ba-17-18-19-37_R.nii'))
    # for s in tqdm(allsubjs):
    #     create_functional_roi(s, 'ba-17', nvoxels=nvoxels,
    #                         split_lr=True)