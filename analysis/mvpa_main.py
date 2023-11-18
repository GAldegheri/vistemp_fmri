import numpy as np
from nipype import Node, JoinNode, Workflow, \
    IdentityInterface, Function
from configs import project_dir

# ---------------------------------------------------------------------------------

def decoding_approaches(sub, roi, approach, task, model, dataformat):
    """
    Calls the specific loading functions and decoding
    functions needed for a given approach (CV, train-test),
    data format (betas, TRs, trial betas), task and model (e.g.
    split into three partitions, into separate views)
    
    - opt: Options object, containing task, model, sub, roi
    - approach: str, one of 'CV', 'splithalf', 'traintest', 'corr_traintest'
    - dataformat: str, one of 'TRs', 'betas', 'trialbetas' or a tuple 
        in the case of train/test splits
    """
    import numpy as np
    import sys
    sys.path.append('/project/3018040.07/Scripts/vistemp_fmri/analysis/')
    #import pdb; pdb.set_trace()
    from mvpa.mvpa_utils import correct_labels, assign_loadfun
    from mvpa.decoding import decode_traintest, decode_CV, \
        decode_splithalf
    from utils import Options, split_options
    from configs import project_dir, bids_dir
    import os
    
    opt = Options(
        sub=sub,
        roi=roi,
        task=task,
        model=model
    )
    
    if isinstance(dataformat, tuple):
        if approach != 'traintest':
            raise ValueError('Data format was provided as a tuple, but approach is not train/test!')
        traindataformat, testdataformat = dataformat
        loadfun_train = assign_loadfun(traindataformat)
        loadfun_test = assign_loadfun(testdataformat)
    else:
        traindataformat = testdataformat = dataformat
        loadfun = assign_loadfun(dataformat)
        
    if 'contr' in opt.roi: # functional contrast
        roi_basedir = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 
                                   'derivatives', 'roi-masks')
        mask_templ = os.path.join(roi_basedir, '{:s}/{:s}_' + opt.roi + '.nii')
    else: # only anatomical map
        roi_basedir = os.path.join(project_dir, 'anat_roi_masks')
        mask_templ = os.path.join(roi_basedir, opt.roi + '.nii')
        
    if approach in ['CV', 'splithalf']:
        # Only need one task and model
        
        DS = loadfun(opt, mask_templ=mask_templ)
        if DS != None:
            DS = correct_labels(DS, opt)
            DS = DS.remove_nonfinite_features()
            if approach == 'CV':
                res = decode_CV(DS, opt)
            else:
                res = decode_splithalf(DS, opt)
            
        else:
            res = None
            
    elif approach == 'traintest':
        
        assert isinstance(opt.task, tuple) and isinstance(opt.model, tuple), \
            'For train/test, 2 tasks and 2 models must be provided as a tuple!'
        
        train_opt, test_opt = split_options(opt)
        
        try:
            trainDS = loadfun(train_opt, mask_templ=mask_templ)
        except:
            trainDS = loadfun_train(train_opt, mask_templ=mask_templ)
        
        if trainDS is not None:
            
            trainDS = correct_labels(trainDS, train_opt)
            
            try:
                testDS = loadfun(test_opt, mask_templ=mask_templ)
            except:
                testDS = loadfun_test(test_opt, mask_templ=mask_templ)
            
            testDS = correct_labels(testDS, test_opt)
            
            nanmask = np.logical_and(np.all(np.isfinite(trainDS.samples), axis=0), \
                np.all(np.isfinite(testDS.samples), axis=0))
            trainDS = trainDS[:, nanmask]
            testDS = testDS[:, nanmask]
            
            res = decode_traintest(trainDS, testDS, train_opt, test_opt)
        
        else:
            
            res = None
            
    if res is not None:
        res['subject'] = sub
        res['roi'] = roi
        res['approach'] = approach
        res['traindataformat'] = traindataformat
        res['testdataformat'] = testdataformat
        if isinstance(task, tuple):
            res['traintask'] = task[0]
            res['testtask'] = task[1]
            res['trainmodel'] = model[0]
            res['testmodel'] = model[1]
        else:
            res['traintask'] = task
            res['testtask'] = task
            res['trainmodel'] = model
            res['testmodel'] = model
        
        return res
    else:
        return np.nan

# ---------------------------------------------------------------------------------

# Stupid function to gather results
def partial_join(this_reslist):
    return this_reslist

# ---------------------------------------------------------------------------------
 
def save_allres(res_list, out_file):
    """
    Collects results from the different Nipype jobs,
    and saves them into a single file.
    """
    import os
    import sys
    sys.path.append('/project/3018040.07/Scripts/vistemp_fmri/analysis')
    from configs import mvpa_outdir as data_dir
    import pandas as pd
    import numpy as np
    
    flat_list = [] # input is a list of lists, flatten into single list
    for r in res_list:
        flat_list.extend(r)
    
    allres = [r for r in flat_list if not (isinstance(r, float) and pd.isna(r))]
    
    #pdb.set_trace()
    
    if len(allres) > 0:
        allres = pd.concat(allres)
        allres = allres.replace(np.nan, 'none')
        fpath = os.path.join(data_dir, out_file)
        allres.to_csv(fpath, index=False)
        print('Saving to ', fpath)
        
    else:
        print('No results to save.')
        
    return

# ---------------------------------------------------------------------------------

def main():
    
    # Subject and ROI list
    subjlist = [f'sub-{i:03d}' for i in range(1, 35)] #1, 35
    
    #rois_to_use = ['ba-17-18-19-37_{:s}_contr-expwidenarr']
    rois_to_use = ['ba-17-18_{:s}_contr-objscrvsbas',
                   'ba-19-37_{:s}_contr-objscrvsbas']
    nothresh = True

    roilist = []

    voxelnos_evc = np.arange(100, 6100, 100)
    voxelnos_loc = np.arange(100, 1100, 100)
    voxelnos_ppa = np.arange(100, 500, 100)
    voxelnos_1937 = np.arange(100, 6100, 100)
    voxelnos_17 = np.arange(100, 1100, 100)
    voxelnos_18 = np.arange(100, 1100, 100)
    voxelnos_19 = np.arange(100, 1100, 100)
    voxelnos_37 = np.arange(100, 1100, 100)
    voxelnos_17181937 = np.arange(100, 1100, 100)
    
    for r in rois_to_use:
        if 'LO' in r:
            voxelnos = voxelnos_loc
        elif 'PPA' in r:
            voxelnos = voxelnos_ppa
        elif '17-18' in r:
            voxelnos = voxelnos_evc
        elif '17' in r:
            voxelnos = voxelnos_17
        elif '18' in r:
            voxelnos = voxelnos_18
        elif '19-37' in r:
            voxelnos = voxelnos_1937
        elif '19' in r:
            voxelnos = voxelnos_19
        elif '37' in r:
            voxelnos = voxelnos_37
        for vn in voxelnos:
            if '_{:s}' in r:
                for s in ['L', 'R']:
                    thisroiname = r.format(s) + '_top-{:g}'.format(vn)
                    if nothresh:
                        thisroiname += '_nothresh'
                    roilist.append(thisroiname)
            else:
                thisroiname = r + '_top-{:g}'.format(vn)
                if nothresh:
                    thisroiname += '_nothresh'
                roilist.append(thisroiname)
                
    allsignif_rois = []
    
    for r in allsignif_rois:
        if '_{:s}' in r:
            for s in ['L', 'R']:
                roilist.append(r.format(s) + '_allsignif')
        else:
            roilist.append(r + '_allsignif')

    # full_rois = [
    #              'ba-17-18_{:s}', 'ba-19-37_{:s}',
    #              'ba-17-18-19-37_{:s}',
    #              'ba-17_{:s}', 'ba-18_{:s}', 
    #              'ba-19_{:s}', 'ba-37_{:s}',
    #              'LO_{:s}']
    
    # for r in full_rois:
    #     if '{:s}' in r:
    #         for s in ['L', 'R']:
    #             roilist.append(r.format(s))
    #     else:
    #         roilist.append(r)   
    
    roilist = ['ba-19-37-infocoupling']
            
    print('------------------- ROI list: -------------------')
    print(roilist)
    print('-------------------------------------------------')
    
    # Identity interface
    idint = Node(IdentityInterface(fields=['sub', 'roi']), name='idint')
    idint.iterables = [('sub', subjlist), ('roi', roilist)]
    
    # Main decoding node
    decodingnode = Node(Function(input_names=['sub', 'roi', 'approach', 'task', 'model', 'dataformat'],
                              output_names=['res'],
                              function=decoding_approaches),
                     name='decodingnode', overwrite=True)
    decodingnode.iterables = [('dataformat', ['betas']*2),
                              ('approach', ['traintest']*2),
                              ('task', [('train', 'test'), ('test', 'train')]),
                              ('model', [(6, 4), (4, 6)])]
    decodingnode.synchronize = True
    
    # Gather results
    partialjoinnode = JoinNode(Function(input_names=['this_reslist'], output_names=['this_reslist'],
                                    function=partial_join), joinsource='decodingnode',
                                    joinfield='this_reslist', name='partialjoinnode')
    
    # Save data
    savingnode = JoinNode(Function(input_names=['res_list', 'out_file'],
                               output_names=[],
                               function=save_allres),
                      joinsource='idint',
                      joinfield='res_list',
                      name='savingnode', overwrite=True)
    
    # --------------------------------------
    savingnode.inputs.out_file = 'results_infocouplROI.csv'
    print('Output file:', savingnode.inputs.out_file)
    # --------------------------------------
    
    # Create workflow
    MVPA_wf = Workflow(name='MVPA_wf')
    MVPA_wf.base_dir = project_dir
    MVPA_wf.connect([(idint, decodingnode, [('sub', 'sub'), ('roi', 'roi')]),
                     (decodingnode, partialjoinnode, [('res', 'this_reslist')]),
                     (partialjoinnode, savingnode, [('this_reslist', 'res_list')])])
    
    MVPA_wf.config['execution']['poll_sleep_duration'] = 1
    MVPA_wf.config['execution']['job_finished_timeout'] = 120
    MVPA_wf.config['execution']['remove_unnecessary_outputs'] = True
    MVPA_wf.config['execution']['stop_on_first_crash'] = True

    MVPA_wf.config['logging'] = {
            'log_directory': MVPA_wf.base_dir+'/'+MVPA_wf.name,
            'log_to_file': False}
    
    # Run workflow
    #MVPA_wf.run()
    MVPA_wf.run('PBS', plugin_args={'max_jobs' : 300, 'qsub_args': '-l walltime=02:00:00,mem=16g', 
                                  'max_tries':3,'retry_timeout': 5, 'max_jobname_len': 15})

# ---------------------------------------------------------------------------------

if __name__=="__main__":
    
    main()