# =================================================================================================
# Helper functions to determine which models are split in wide/narrow, exp/unexp, thirds
# =================================================================================================
    
def isWideNarrow(opt):
    # Little helper function to determine
    # which tasks/models should be labeled as 
    # wide vs. narrow
    
    if (opt.task=='train' and opt.model in [4, 6]) or (opt.task=='test' and opt.model in [3, 4, 5]):
        return True
    else:
        return False
    
# -------------------------------------------------------------------------------------------------

def is30or90(opt):
    # Little helper function to determine
    # which tasks/models should be labeled as 
    # rot. 30 vs. 90 degrees
    
    if (opt.task=='train' and opt.model in [3, 5]) or (opt.task=='test' and opt.model==2):
        return True
    else:
        return False

# -------------------------------------------------------------------------------------------------

def isAllViews(opt):
    # Which tasks/models contain both object view (A/B)
    # and scene rotation (30/90).
    
    if (opt.task=='train' and opt.model in [7, 8, 9, 10]) or (opt.task=='test' and opt.model in [6, 7, 8]):
        return True
    else:
        return False
    
# -------------------------------------------------------------------------------------------------

def isInitAndFinal(opt):
    # Models (for now only one) which include both initial view and final
    return (opt.task=='test' and opt.model==22)