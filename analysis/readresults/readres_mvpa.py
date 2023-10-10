import pandas as pd
import warnings
import numpy as np
import sys
sys.path.append('..')

def quick_get_results(res_list, combine_thirds=True):
    """
    Combines all relevant data
    reading functions for quick access
    to the data.
    """
    results = merge_results(res_list)
    #if combine_thirds:
    #    varcombs = get_varcombs(results)
    results = parse_roi_info(results)
    results = exclude_participants(results)
    #results = get_subj_avg(results, avg_decodedirs=True)
    #results = fill_in_nvoxels(results)
    return results

def exclude_participants(results):
    res = results[results['subject'].isin([f'sub-{s:03d}' for s in [9, 22, 19, 26]])]
    return res

def merge_results(res_list):
    """
    Given list of results files,
    returns a dataframe merging all of them.
    """
    if not isinstance(res_list, list):
        res_list = [res_list]
        
    all_results = []
    for r in res_list:
        all_results.append(pd.read_csv(r))
    all_results = pd.concat(all_results)
    all_results = all_results.replace(np.nan, 'none')
    return all_results

def get_subj_avg(results, avg_decodedirs=False):
    results = results.drop(['runno'], axis=1)
    ind_vars = ['subject', 'roi', 'approach', 
                'traindataformat', 'testdataformat', 'traintask',
                'testtask', 'trainmodel', 'testmodel', 
                'hemi', 'contrast', 'nvoxels', 'split']
    ind_vars = [i for i in ind_vars if i in results.columns]
    
    if avg_decodedirs:
        # remove traintask, trainmodel, testtask, testmodel...
        ind_vars = [i for i in ind_vars if 'train' not in i and 'test' not in i]
        groupedres = []
        taskmodelpairs = list(results[['traintask', 'trainmodel',
                                      'testtask', 'testmodel']].drop_duplicates().itertuples(index=False, name=None))
        taskmodelpairs = [((a, b), (c, d)) for a, b, c, d in taskmodelpairs]
        taskmodelpairs = {frozenset(x) for x in taskmodelpairs}
        taskmodelpairs = [tuple(x) for x in taskmodelpairs]
        for (trtask, trm), (tetask, tem) in taskmodelpairs:
            thistm_fwd = results[(results['traintask']==trtask)&(results['trainmodel']==trm)&\
                                 (results['testtask']==tetask)&(results['testmodel']==tem)]
            thistm_back = results[(results['traintask']==tetask)&(results['trainmodel']==tem)&\
                                 (results['testtask']==trtask)&(results['testmodel']==trm)]
            thistm = pd.concat([thistm_fwd, thistm_back])
            thesemodels = sorted(list(thistm.trainmodel.unique()))
            thesetasks = sorted(list(thistm.traintask.unique()), reverse=True)
            assert len(thesemodels)==2 and len(thesetasks)==2
            thistm = thistm.groupby(ind_vars, dropna=False).mean().reset_index()
            thistm['traintask'] = thesetasks[0]+'_'+thesetasks[1]
            thistm['testtask'] = thesetasks[0]+'_'+thesetasks[1]
            thistm['trainmodel'] = str(thesemodels[0])+'_'+str(thesemodels[1])
            thistm['testmodel'] = str(thesemodels[0])+'_'+str(thesemodels[1])
            groupedres.append(thistm)
        results = pd.concat(groupedres)
    else:
        results = results.groupby(ind_vars, dropna=False).mean().reset_index()
    
    if 'view' in results.columns:
        results = results.drop(['view'], axis=1)
    
    return results 

def parse_roi_info(results):
    """
    Info to be extracted from ROI string:
    - ROI name (e.g. BA 17, LOC, PPA)
    - Hemisphere (L/R/LR)
    - Contrast (e.g. Object vs. Scrambled)
    - N voxels (e.g. 100, 1000, 'all' [all significant],
                None [no selection])
    """
    if 'roi' not in results.columns:
        warnings.warn('No ROI information in this dataframe!')
        return results
    
    roinames = []
    hemispheres = []
    contrasts = []
    nvoxels = []
    
    for r in results.roi:
        allinfo = r.split('_')
        roinames.append(allinfo[0])
        if len(allinfo) > 1:
            if allinfo[1] in ['L', 'R']:
                hemispheres.append(allinfo[1])
                contrindx = 2
            else:
                hemispheres.append('LR')
                contrindx = 1
            if len(allinfo) > contrindx:
                contrasts.append(allinfo[contrindx].split('contr-')[1])
                if 'allsignif' in allinfo[contrindx+1]:
                    nvoxels.append('all')
                elif 'top-' in allinfo[contrindx+1]:
                    nvoxels.append(int(allinfo[contrindx+1].split('top-')[1]))
            else:
                contrasts.append('none')
                nvoxels.append('none')
        else:
            hemispheres.append('LR')
            contrasts.append('none')
            nvoxels.append('none')
    
    results['roi'] = roinames
    results['hemi'] = hemispheres
    results['contrast'] = contrasts
    results['nvoxels'] = nvoxels
    
    return results

def fill_in_nvoxels(results):
    
    
    # Get existing voxels numbers per each ROI
    nvox = {
        r: sorted([n for n in results[results['roi']==r].nvoxels.unique() if n != 'none'])  for 
            r in results.roi.unique()
    }
    
    def voxel_filling(theseres, nvox_dict=nvox):
       
       # maximum number of voxels in this set of results
       max_nv = 0
       for nv in nvox_dict[theseres.roi.unique()[0]]:
            if len(theseres[theseres['nvoxels']==nv])>0:
                max_nv = nv
            elif max_nv != 0:
                maxnvoxres = theseres[theseres['nvoxels']==max_nv].copy()
                maxnvoxres['nvoxels'] = nv
                theseres = pd.concat([theseres, maxnvoxres])
       return theseres
    
    fillin_fn = lambda res: voxel_filling(res, nvox_dict=nvox)
    
    filledinres = apply_fn_to_varcombs(results, fillin_fn)
    
    return filledinres

def combine_splits(res):
    """
    Given a results dataframe divided in three splits (of trials),
    combine them in the appropriate way for each column.
    """
    for s in sorted(res.split.unique()):
        thissplitlength = len(res[res['split']==s])
        res.loc[res['split']==s, 'sample'] = list(range(thissplitlength))
    
    return res.groupby('sample').mean().reset_index().drop(
        ['sample'], axis=1)

def combine_splits_all(results):
    
    third_results = results[~pd.isnull(results['split'])]
    nonthird_results = results[pd.isnull(results['split'])]

    combined_res = apply_fn_to_varcombs(third_results, combine_splits)
    
    if len(nonthird_results) > 0:
        combined_res = pd.concat([combined_res, nonthird_results])
    
    return combined_res.reset_index()
 
                
def apply_fn_to_varcombs(results, func):
    """
    - results: a results dataframe.
    - func: a lambda function taking results df as input and output 
        (any extra input arguments need to be defined outside).
    """
    allres = []
    varcombs = get_varcombs(results)
    for vc in varcombs:
        for i, v in enumerate(vc):
            # Get results corresponding to this specific combination.
            if i==0:
                thismask = np.array(results[v]==vc[v])
            else:
                thismask &= np.array(results[v]==vc[v])
        theseresults = results[thismask]
        theseresults = func(theseresults)
        allres.append(theseresults)
        
    allres = pd.concat(allres)
    
    return allres
        

def get_varcombs(results):
    
    ind_vars = ['subject', 'roi', 'approach', 
                'traindataformat', 'testdataformat', 'traintask',
                'testtask', 'trainmodel', 'testmodel', 
                'hemi', 'contrast', 'runno', 'view']
    ind_vars = [i for i in ind_vars if i in results.columns]
    
    # get unique combinations of independent variables:
    varcombinations = [r.to_dict() for _, r in
                       results[ind_vars].drop_duplicates().iterrows()]
    
    return varcombinations