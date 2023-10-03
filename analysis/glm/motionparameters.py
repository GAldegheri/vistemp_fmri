import numpy as np

################################# Add motion parameters #####################################

def read_motion_par(mp_files, task, run):
    '''
    mp_files is a list of .txt motion parameter files
    (one per run)
    '''
    
    motpar = [m for m in mp_files if 'task-{:s}_run-{:g}'.format(task, run+1) in m][0]
    motionarray = np.loadtxt(motpar)
    
    return [list(col) for col in motionarray.T] # this *should* be what Nipype wants

# ---------------------------------------------------------------------------------

def add_motion_regressors(subj_info, task, use_motion_reg, motpar):
    '''
    - subj_info: output of ModelSpecify
    - use_motion_reg: bool
    - motpar: list of .txt files
    '''
    import sys
    sys.path.append('/project/3018040.07/vistemp_fmri/analysis/')
    from glm.motionparameters import read_motion_par
    
    if use_motion_reg:
        for run, _ in enumerate(subj_info):
            subj_info[run].regressor_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
            subj_info[run].regressors = read_motion_par(motpar, task, run) # list of 6 columns
    
    return subj_info