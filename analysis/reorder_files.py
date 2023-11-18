import os
from glob import glob
import shutil

basedir = '/project/3018040.07/bids/derivatives/spm-preproc/derivatives/spm-stats/betas'

if __name__=="__main__":
    subjlist = [f'sub-{i:03d}' for i in range(1, 35)]
    for s in subjlist:
        thissubjdir = os.path.join(basedir, s, 'test',
                                   'model_10')
        allfiles = glob(os.path.join(thissubjdir, '*'))
        outdir = os.path.join(thissubjdir, 'FIR')
        os.makedirs(outdir)
        for f in allfiles:
            shutil.move(f, outdir)