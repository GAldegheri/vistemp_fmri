import numpy as np
from nipype import Node, JoinNode, Workflow, \
    IdentityInterface, Function
from configs import project_dir
import pandas as pd

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
    sys.path.append('/project/3018040.07/vistemp_fmri/analysis')
    from mvpa.mvpa_utils import correct_labels, assign_loadfun
    from mvpa.decoding import decode_traintest, decode_CV, \
        decode_SplitHalf, traintest_dist
    from general