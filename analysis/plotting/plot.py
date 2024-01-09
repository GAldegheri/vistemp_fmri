import numpy as np
import pingouin as pg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append('..')
from mne.stats import permutation_cluster_1samp_test
import plotting.PtitPrince as pt


def plot_by_nvoxels(data, measure='distance', tfce_pvals=None, right_part=False, saveimg=False):
    """
    - data: pandas dataframe containing the data
    - tfce_pvals are provided if they have been precomputed,
        else they're computed here
    """
    if right_part:
        assert 'hemi' in data.columns
    assert data.roi.nunique()==1
    
    avgdata = data.copy()
    nsubjs = avgdata.subject.nunique()
    maxvoxels = avgdata.nvoxels.max()
    
    # sort n. voxels and make categorical
    avgdata.sort_values('nvoxels', inplace=True, ascending=True)
    avgdata.loc[:, 'nvoxels'] = avgdata.loc[:, 'nvoxels'].astype(str)
    avgdata.loc[:, 'nvoxels'] = pd.Categorical(avgdata.loc[:, 'nvoxels'], 
                                               categories=avgdata.nvoxels.unique(), ordered=True)
    
    
    if tfce_pvals is None:
        _, _, tfce_pvals, _ = get_tfce_stats(avgdata.groupby(['subject','nvoxels']).mean().reset_index(),
                                             measure=measure, n_perms=1000)
            
    fig = plt.figure(figsize=(20,10), facecolor='white')
    gs = GridSpec(1, 4, figure=fig)
    with sns.axes_style('white'):
        if right_part:
            ax0 = fig.add_subplot(gs[0, :-1])
        else:
            ax0 = fig.add_subplot(gs[0, :])
        sns.lineplot(data=avgdata.groupby(['subject', 'nvoxels']).mean().reset_index(), 
                     x='nvoxels', y=measure,
                     palette='Set2', ci=95, marker='o', mec='none', markersize=10) #plot_kws=dict(edgecolor="none")) #markersize=10
        plt.yticks(fontsize=20)
        #ax0.set(ylim=(0.05, 0.35), xticks=['100']+[str(x) for x in np.arange(500, 3500, 500)])
        ax0.set(ylim=(-0.05, 0.2), xticks=['100']+[str(x) for x in np.arange(500, maxvoxels+500, 500)])
        ax0.set_xlabel('Number of Voxels', fontsize=24)
        ax0.set_ylabel('Classifier information (a.u.)', fontsize=24)
        plt.xticks(fontsize=20)
        plt.margins(0.02)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        for x in np.arange(0, len(tfce_pvals)):
            if tfce_pvals[x] < 0.01:
                ax0.scatter(x, -0.040, marker=(6, 2, 0), s=180, color='k', linewidths=1.)
                ax0.scatter(x, -0.033, marker=(6, 2, 0), s=180, color='k', linewidths=1.)
            elif tfce_pvals[x] < 0.05:
                ax0.scatter(x, -0.040, marker=(6, 2, 0), s=180, color='k', linewidths=1.)
        ax0.axhline(0.0, color='k', linestyle='--', linewidth=2.)
    if right_part:
        avgdata = avgdata.groupby(['subject', 'hemi']).mean().reset_index()
        with sns.axes_style('white'):
            ax1 = fig.add_subplot(gs[0, -1])
            _, suppL, densL = pt.half_violinplot(y=measure, data=avgdata[avgdata['hemi']=='L'], color='.8', 
                                                width=.3, inner=None, bw=.4, flip=False, CI=True, offset=0.04)
            _, suppR, densR = pt.half_violinplot(y=measure, data=avgdata[avgdata['hemi']=='R'], color='.8', 
                                                width=.3, inner=None, bw=.4, flip=True, CI=True, offset=0.04)
            
            densities_left = []
            for d in avgdata[avgdata['hemi']=='L'][measure]:
                ix, _ = find_nearest(suppL[0], d)
                densities_left.append(densL[0][ix])
            densities_left = np.array(densities_left).reshape(nsubjs,1)
            scatter_left = -0.04-np.random.uniform(size=(nsubjs,1))*densities_left*0.15
            plt.scatter(scatter_left, avgdata[avgdata['hemi']=='L'][measure], color='black', alpha=.3)
            densities_right = []
            for d in avgdata[avgdata['hemi']=='R'][measure]:
                ix, _ = find_nearest(suppR[0], d)
                densities_right.append(densR[0][ix])
            densities_right = np.array(densities_right).reshape(nsubjs,1)
            scatter_right = 0.04+np.random.uniform(size=(nsubjs,1))*densities_right*0.15
            plt.scatter(scatter_right, avgdata[avgdata['hemi']=='R'][measure], color='black', alpha=.3)
            
            # Get mean and 95% CI:
            meanacc = avgdata[measure].mean()
            tstats = pg.ttest(avgdata.groupby(['subject']).mean().reset_index()[measure], 0.0)
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
                ax1.plot(tick,meanacc, 'o', markersize=15, color='black')
            ax1.axhline(0.0, linestyle='--', color='black')
            plt.yticks(fontsize=20) 
            ax1.set_xlabel('Average', fontsize=24)
            ax1.set_ylabel('Classifier information (a.u.)', fontsize=24)
            ax1.set(ylim=(-0.8, 0.8))
            ax1.axes_style = 'white'
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            #ax1.spines['left'].set_visible(False)
    plt.tight_layout()
    # if saveimg:
    #     plt.savefig('results_plots/EVC_nvox_distance.pdf')
    #plt.show()

def get_tfce_stats(data, measure='distance', n_perms=10000):
    subxvoxels = df_to_array_tfce(data.groupby(['subject','nvoxels']).mean().reset_index(),
                                  measure=measure)
    threshold_tfce = dict(start=0, step=0.01)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        subxvoxels, n_jobs=1, threshold=threshold_tfce, adjacency=None,
        n_permutations=n_perms, out_type='mask')
    return t_obs, clusters, cluster_pv, H0


def df_to_array_tfce(df, measure='correct'):
    """
    """
    subxvoxels = np.zeros((df.subject.nunique(), df.nvoxels.nunique()))
    for i, sub in enumerate(np.sort(df.subject.unique())):
        for j, nv in enumerate(np.sort(df.nvoxels.unique())):
            thisdata = df[(df['subject']==sub)&(df['nvoxels']==nv)]
            subxvoxels[i, j] = thisdata[measure].values
    return subxvoxels




def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
