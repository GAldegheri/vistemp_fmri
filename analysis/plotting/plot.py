import numpy as np
import pingouin as pg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append('..')
from mvpa.classify_models import isExpUnexp
from utils import Options
from mne.stats import permutation_cluster_1samp_test
import plotting.PtitPrince as pt
import ipdb


def plot_by_nvoxels(data, measure='distance', tfce_pvals=None):
    """
    - data: pandas dataframe containing the data
    - tfce_pvals are provided if they have been precomputed,
        else they're computed here
    """
    assert data.roi.nunique()==1
    
    avgdata = data.copy()
    nsubjs = avgdata.subject.nunique()
    maxvoxels = avgdata.nvoxels.max()
    
    # sort n. voxels and make categorical
    avgdata.sort_values('nvoxels', inplace=True, ascending=True)
    avgdata.loc[:, 'nvoxels'] = avgdata.loc[:, 'nvoxels'].astype(str)
    avgdata.loc[:, 'nvoxels'] = pd.Categorical(avgdata.loc[:, 'nvoxels'], 
                                               categories=avgdata.nvoxels.unique(), ordered=True)
    
    # should this be provided already averaged?
    #avgdata = data.groupby(['subject', 'nvoxels', 'expected', 'hemi']).mean().reset_index()
    
    avgdiffs = accs_to_diffs(avgdata).groupby(['subject', 'hemi']).mean().reset_index()
    
    if tfce_pvals is None:
        if 'ba-17-18' in data.roi.unique():
            tfce_pvals = [0.188,  0.005, 0.0067, 0.0078, 0.0088, 
                          0.0106, 0.0109, 0.0115, 0.0127, 0.0148,
                          0.0201, 0.0424, 0.0224, 0.0237, 0.0244, 
                          0.0273, 0.0321, 0.0357, 0.0391, 0.0424,
                          0.0435, 0.0462, 0.0119, 0.0435, 0.0039,
                          0.0055, 0.0062, 0.003,  0.003,  0.0039]
        elif 'LO' in data.roi.unique():
            tfce_pvals = [0.8016, 0.5093, 0.5406, 0.5151, 0.5151,
                          0.4857, 0.4423, 0.4658, 0.4912, 0.4541,
                          0.4191, 0.4912, 0.4307, 0.7372, 0.5799,
                          0.4968, 0.4597, 0.5747, 0.4968, 0.5747]
        else:
            subxvoxels = df_to_array_tfce(avgdata.groupby(['subject','nvoxels','expected']).mean().reset_index(),
                                          measure=measure)
            threshold_tfce = dict(start=0, step=0.01)
            _, _, tfce_pvals, _ = permutation_cluster_1samp_test(
                subxvoxels, n_jobs=1, threshold=threshold_tfce, adjacency=None,
                n_permutations=10000, out_type='mask') #10000
            
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(1, 4, figure=fig)
    with sns.axes_style('white'):
        ax0 = fig.add_subplot(gs[0, :-1])
        sns.lineplot(data=avgdata.groupby(['subject', 'nvoxels', 'expected']).mean().reset_index(), 
                     x='nvoxels', y=measure,
                     hue='expected', hue_order=[True, False], #linewidth=1,
                     palette='Set2', ci=68, marker='o', mec='none', markersize=15) #plot_kws=dict(edgecolor="none")) #markersize=10
        plt.yticks(fontsize=20)
        #ax0.set(ylim=(0.05, 0.35), xticks=['100']+[str(x) for x in np.arange(500, 3500, 500)])
        ax0.set(ylim=(0.0, 0.45), xticks=['100']+[str(x) for x in np.arange(500, maxvoxels+500, 500)])
        ax0.set_xlabel('Number of Voxels', fontsize=24)
        ax0.set_ylabel('Classifier information (a.u.)', fontsize=24)
        plt.xticks(fontsize=20)
        plt.margins(0.02)
        ax0.legend_.set_title(None)
        ax0.legend(['Congruent', 'Incongruent'], prop={'size': 26}, frameon=False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        for x in np.arange(0, len(tfce_pvals)):
            if tfce_pvals[x] < 0.01:
                ax0.scatter(x, 0.02, marker=(6, 2, 0), s=200, color='k', linewidths=1.)
                ax0.scatter(x, 0.04, marker=(6, 2, 0), s=200, color='k', linewidths=1.)
            elif tfce_pvals[x] < 0.05:
                ax0.scatter(x, 0.02, marker=(6, 2, 0), s=200, color='k', linewidths=1.)
    with sns.axes_style('white'):
        ax1 = fig.add_subplot(gs[0, -1])
        _, suppL, densL = pt.half_violinplot(y='difference', data=avgdiffs[avgdiffs['hemi']=='L'], color='.8', 
                                            width=.3, inner=None, bw=.4, flip=False, CI=True, offset=0.04)
        _, suppR, densR = pt.half_violinplot(y='difference', data=avgdiffs[avgdiffs['hemi']=='R'], color='.8', 
                                            width=.3, inner=None, bw=.4, flip=True, CI=True, offset=0.04)
        
        densities_left = []
        for d in avgdiffs[avgdiffs['hemi']=='L']['difference']:
            ix, _ = find_nearest(suppL[0], d)
            densities_left.append(densL[0][ix])
        densities_left = np.array(densities_left).reshape(nsubjs,1)
        scatter_left = -0.04-np.random.uniform(size=(nsubjs,1))*densities_left*0.15
        plt.scatter(scatter_left, avgdiffs[avgdiffs['hemi']=='L']['difference'], color='black', alpha=.3)
        densities_right = []
        for d in avgdiffs[avgdiffs['hemi']=='R']['difference']:
            ix, _ = find_nearest(suppR[0], d)
            densities_right.append(densR[0][ix])
        densities_right = np.array(densities_right).reshape(nsubjs,1)
        scatter_right = 0.04+np.random.uniform(size=(nsubjs,1))*densities_right*0.15
        plt.scatter(scatter_right, avgdiffs[avgdiffs['hemi']=='R']['difference'], color='black', alpha=.3)
        
        # Get mean and 95% CI:
        meandiff = avgdiffs['difference'].mean()
        tstats = pg.ttest(avgdiffs.groupby(['subject']).mean().reset_index()['difference'], 0.0)
        ci95 = tstats['CI95%'][0]
        #ax1.axis("equal")
        #ax1.set_aspect('equal')
        for tick in ax1.get_xticks():
            #ax1.plot([tick-0.1, tick+0.1], [meandiff, meandiff],
            #            lw=4, color='k')
            ax1.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
            ax1.plot([tick-0.01, tick+0.01], [ci95[0], ci95[0]], lw=3, color='k')
            ax1.plot([tick-0.01, tick+0.01], [ci95[1], ci95[1]], lw=3, color='k')
            #circlemarker = plt.Circle((tick, meandiff), 0.015, color='k')
            #ax1.add_patch(circlemarker)
            ax1.plot(tick,meandiff, 'o', markersize=15, color='black')
        ax1.axhline(0.0, linestyle='--', color='black')
        plt.yticks(fontsize=20) 
        ax1.set_xlabel('Average', fontsize=24)
        ax1.set_ylabel('Δ Classifier information (a.u.)', fontsize=24)
        ax1.set(ylim=(-0.4, 0.4))
        ax1.axes_style = 'white'
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        #ax1.spines['left'].set_visible(False)
    plt.tight_layout()
    #plt.savefig('Plots/EVC_nvox_distance.pdf')
    plt.show()



def accs_to_diffs(df, measure='correct'):
    diffs = []
    for nv in df.nvoxels.unique():
        for hemi in df.hemi.unique():
            for sub in df[(df['nvoxels']==nv)&(df['hemi']==hemi)].subject.unique():
                thissub = df[(df['nvoxels']==nv)&(df['hemi']==hemi)&(df['subject']==sub)]
                #assert(len(thissub)==2) # should only be one expected, and one unexpected value
                thisdiff = thissub[thissub['expected']==True][measure].values[0] - \
                           thissub[thissub['expected']==False][measure].values[0]
                diffs.append({'subject': sub, 'nvoxels': nv, 'hemi': hemi, 'difference': thisdiff})
    return pd.DataFrame(diffs)


    
def df_to_array_tfce(df, measure='correct'):
    """
    """
    subxvoxels = np.zeros((df.subject.nunique(), df.nvoxels.nunique()))
    for i, sub in enumerate(np.sort(df.subject.unique())):
        for j, nv in enumerate(np.sort(df.nvoxels.unique())):
            thisdata = df[(df['subject']==sub)&(df['nvoxels']==nv)]
            try:
                subxvoxels[i, j] = thisdata[thisdata['expected']==True][measure].values - \
                    thisdata[thisdata['expected']==False][measure].values
            except:
                ipdb.set_trace()
    return subxvoxels



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
