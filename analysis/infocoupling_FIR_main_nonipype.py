import argparse

def decode_FIR_timecourses(sub, roi, task, model, approach):
    """
    """
    import os
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_betas
    from mvpa.decoding import decode_CV, decode_traintest
    from mvpa.mvpa_utils import correct_labels
    from configs import project_dir, bids_dir
    from utils import Options, split_options
    
    opt = Options(
        sub=sub,
        roi=roi,
        task=task,
        model=model
    )
    
    if 'contr' in opt.roi: # functional contrast
        roi_basedir = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 
                                   'derivatives', 'roi-masks')
        mask_templ = os.path.join(roi_basedir, '{:s}/{:s}_' + opt.roi + '.nii')
    else: # only anatomical map
        roi_basedir = os.path.join(project_dir, 'anat_roi_masks')
        mask_templ = os.path.join(roi_basedir, opt.roi + '.nii')
        
    max_delay = 9
    
    if approach=='CV':
        
        DS = load_betas(opt, mask_templ=mask_templ, fir=True,
                        max_delay=max_delay)
        
        if DS is not None:
            
            DS = correct_labels(DS, opt)
            DS = DS.remove_nonfinite_features()
            
            allres = []
            
            for d in range(max_delay):
                thisDS = DS[DS.sa.delay==d]
                allres.append(decode_CV(thisDS, opt))
            
            allres = pd.concat(allres)
            
        else:
            
            allres = None
            
    elif approach=='traintest':
        
        train_opt, test_opt = split_options(opt)
        
        # only option for now, maybe implement more later
        assert train_opt.task=='train' and test_opt.task=='test'
        
        # Just normal betas for training
        trainDS = load_betas(train_opt, mask_templ=mask_templ,
                                fir=False)
        
        if trainDS is not None:
            
            trainDS = correct_labels(trainDS, train_opt)
            
            testDS = load_betas(test_opt, mask_templ=mask_templ, 
                                fir=True, max_delay=max_delay)
            testDS = correct_labels(testDS, test_opt)
            
            nanmask = np.logical_and(np.all(np.isfinite(trainDS.samples), axis=0), \
                np.all(np.isfinite(testDS.samples), axis=0))
            trainDS = trainDS[:, nanmask]
            testDS = testDS[:, nanmask]
            
            allres = []
            
            for d in range(max_delay+1):
                thistestDS = testDS[testDS.sa.delay==d]
                thisres = decode_traintest(trainDS, thistestDS, \
                    train_opt, test_opt)
                allres.append(thisres)
            
            allres = pd.concat(allres)
        
        else:
            
            allres = None
    
    #pdb.set_trace()    
    if allres is not None:
        allres['subject'] = opt.sub
        allres['roi'] = opt.roi
        allres['approach'] = approach
        if approach == 'traintest':
            allres['traintask'] = train_opt.task
            allres['testtask'] = test_opt.task
            allres['trainmodel'] = train_opt.model
            allres['testmodel'] = test_opt.model
        elif approach == 'CV':
            allres['traintask'] = opt.task
            allres['testtask'] = opt.task
            allres['trainmodel'] = opt.model
            allres['testmodel'] = opt.model
        
        return allres, sub, roi
    
    else:
        return np.nan, sub, roi

# ---------------------------------------------------------------------------------

def correlate_timeseqs(tc, sub, roi):
    import pandas as pd
    import numpy as np
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_betas
    from mvpa.mvpa_utils import split_expunexp
    from utils import Options
    from nilearn.image import new_img_like
    
    
    n_timepoints = tc.delay.nunique()
    print(pd.__version__)
    print(tc.dtypes)
    print(tc)
    tc = tc.groupby(['delay', 'expected']).mean().reset_index()
    
    
    # load FIR timecourses
    univar_opt = Options(
        sub=sub, 
        task='test',
        model=27
    )
    
    wholebrainDS = load_betas(univar_opt, mask_templ=None, 
                             fir=True)
    wholebrainDS = split_expunexp(wholebrainDS)
    nanmask = np.all(np.isfinite(wholebrainDS.samples), axis=0)
    wholebrainDS = wholebrainDS[:, nanmask]
    
    univar_df = pd.DataFrame(
        {'delay': list(wholebrainDS.sa.delay),
         'expected': list(wholebrainDS.sa.expected),
         'samples': list(wholebrainDS.samples)}
    )
    print(univar_df.dtypes)
    univar_df = univar_df.groupby(['delay', 'expected']).mean().reset_index()
    
    # Get (n. voxels x n. timepoints) arrays for exp and unexp
    exp_univar_array = np.vstack(univar_df[univar_df.expected==1].samples).T
    unexp_univar_array = np.vstack(univar_df[univar_df.expected==0].samples).T
    # Normalize
    exp_univar_array = (exp_univar_array - np.mean(exp_univar_array, axis=1, keepdims=True))/np.std(exp_univar_array, axis=1, keepdims=True)
    unexp_univar_array = (unexp_univar_array - np.mean(unexp_univar_array, axis=1, keepdims=True))/np.std(unexp_univar_array, axis=1, keepdims=True)
    
    # Same thing for multivariate sequence
    exp_multivar_array = np.hstack(tc[tc.expected==True].distance).reshape(1, n_timepoints)
    unexp_multivar_array = np.hstack(tc[tc.expected==False].distance).reshape(1, n_timepoints)
    exp_multivar_array = (exp_multivar_array - np.mean(exp_multivar_array, axis=1, keepdims=True))/np.std(exp_multivar_array, axis=1, keepdims=True)
    unexp_multivar_array = (unexp_multivar_array - np.mean(unexp_multivar_array, axis=1, keepdims=True))/np.std(unexp_multivar_array, axis=1, keepdims=True)
    
    # Compute Pearsons correlations
    exp_corrs = np.dot(exp_univar_array, exp_multivar_array.T)/(n_timepoints-1)
    unexp_corrs = np.dot(unexp_univar_array, unexp_multivar_array.T)/(n_timepoints-1)
    
    # Convert into brain maps
    i, j, k = wholebrainDS.fa.voxel_indices.T
    
    exp_map = np.full(wholebrainDS.a.voxel_dim, np.nan)
    exp_map[i, j, k] = exp_corrs.flatten()
    exp_map = new_img_like('/project/3018040.05/anat_roi_masks/wholebrain.nii', exp_map)
    
    unexp_map = np.full(wholebrainDS.a.voxel_dim, np.nan)
    unexp_map[i, j, k] = unexp_corrs.flatten()
    unexp_map = new_img_like('/project/3018040.05/anat_roi_masks/wholebrain.nii', unexp_map)
    
    return exp_map, unexp_map, sub, roi
            
# ---------------------------------------------------------------------------------

def save_corrmaps(exp_map, unexp_map, sub, roi):
    import nibabel as nb
    import os
    
    outdir = os.path.join('/project/3018040.05/',
                          'FIR_correlations', roi)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    nb.save(exp_map, os.path.join(outdir,
                                f'{sub}_exp.nii'))
    nb.save(unexp_map, os.path.join(outdir,
                                f'{sub}_unexp.nii'))
    
    return    

# ---------------------------------------------------------------------------------

def main(sub, roi):
    print('--------------------------------')
    print(f'Subject: {sub}, ROI: {roi}')
    print('--------------------------------')
    
    print('Starting decoding...')
    allres, sub, roi = decode_FIR_timecourses(sub, roi, 
                                              ('train', 'test'),
                                              (5, 24), 'traintest')
    print('Done! Computing correlations...')
    exp_map, unexp_map, sub, roi = correlate_timeseqs(allres, sub, roi)
    print('Done!')
    save_corrmaps(exp_map, unexp_map, sub, roi)
    print('Saved files.')
    return

# ---------------------------------------------------------------------------------

if __name__=="__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--sub", required=True, type=str, help="Subject")
    parser.add_argument("--roi", required=True, type=str, help="ROI")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the args namespace
    main(args.sub, args.roi)