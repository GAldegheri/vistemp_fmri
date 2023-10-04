import scipy.io as spio

class Options(object):
    def __init__(self, sub=None, 
                 task=None, model=None,
                 roi=None):
        self.sub = sub
        self.task = task
        self.model = model
        self.roi = roi
        
def split_options(opt):
    """
    Helper function to split options into train and test
    """
    assert isinstance(opt.task, tuple) and isinstance(opt.model, tuple), \
        "Task and model should both be tuples!"
    train_opt = Options(sub=opt.sub, task=opt.task[0],
                        model=opt.model[0], roi=opt.roi)
    test_opt = Options(sub=opt.sub, task=opt.task[1],
                       model=opt.model[1], roi=opt.roi)
    return train_opt, test_opt

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict