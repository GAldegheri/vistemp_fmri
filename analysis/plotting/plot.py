import numpy as np
import pingouin as pg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from pathlib import Path
import sys
sys.path.append('..')
from mne.stats import permutation_cluster_1samp_test
import plotting.PtitPrince as pt


def plot_by_nvoxels(data, measure='distance', tfce_pvals=None, right_part=False, n_perms=10000):
    """
    - data: pandas dataframe containing the data
    - tfce_pvals are provided if they have been precomputed,
        else they're computed here
    """
    fpath = Path("./fonts/HelveticaWorld-Regular.ttf")
    fontprop = FontProperties(fname=fpath)
    
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
                                             measure=measure, n_perms=n_perms)
            
    fig = plt.figure(figsize=(20,10), facecolor='white')
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 4, 4, 1])
    with sns.axes_style('white'):
        if right_part:
            ax0 = fig.add_subplot(gs[1:3, 1:])
        else:
            ax0 = fig.add_subplot(gs[1:3, :])
        sns.lineplot(data=avgdata.groupby(['subject', 'nvoxels']).mean().reset_index(), 
                     x='nvoxels', y=measure,
                     palette='Dark2', ci=95, marker='o', mec='none', markersize=10)
        if measure == 'distance':
            ylabel = 'Classifier Information (a.u.)'
            ylimits_left = (-0.05, 0.2)
            ylimits_right = (-0.8, 0.8)
            yticks = list(np.arange(-0.05, 0.2, 0.05))
            marker_bottom = -0.04
            marker_mid = -0.03
            marker_top = -0.01
            chancelevel = 0.0
        elif measure == 'correct':
            ylabel = 'Decoding Accuracy (a.u.)'
            ylimits_left = (0.45, 0.6)
            ylimits_right = (0.1, 0.9)
            yticks = list(np.arange(0.45, 0.65, 0.05))
            marker_bottom = 0.46
            marker_mid = 0.467
            marker_top = 0.474
            chancelevel = 0.5
        plt.yticks(font=fpath, fontsize=28, ticks=yticks)
        #ax0.set(ylim=(0.05, 0.35), xticks=['100']+[str(x) for x in np.arange(500, 3500, 500)])
        ax0.set(ylim=ylimits_left, xticks=['100', '500']+[str(x) for x in np.arange(1000, maxvoxels+1000, 1000)])
        ax0.set_xlabel('Number of Voxels', font=fpath, fontsize=32)
        ax0.set_ylabel(ylabel, font=fpath, fontsize=32)
        plt.xticks(font=fpath, fontsize=28)
        plt.margins(0.02)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_linewidth(2)
        ax0.spines['bottom'].set_linewidth(2)
        for x in np.arange(0, len(tfce_pvals)):
            if tfce_pvals[x] < 0.001:
                ax0.scatter(x, marker_bottom, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
                ax0.scatter(x, marker_mid, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
                ax0.scatter(x, marker_top, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
            elif tfce_pvals[x] < 0.01:
                ax0.scatter(x, marker_bottom, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
                ax0.scatter(x, marker_mid, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
            elif tfce_pvals[x] < 0.05:
                ax0.scatter(x, marker_bottom, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
        ax0.axhline(chancelevel, color='k', linestyle='--', linewidth=2.)
    if right_part:
        avgdata = avgdata.groupby(['subject', 'hemi']).mean().reset_index()
        with sns.axes_style('white'):
            ax1 = fig.add_subplot(gs[:, 0])
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
            ax1.axhline(chancelevel, linestyle='--', color='black', linewidth=2)
            plt.yticks(font=fpath, fontsize=32) 
            ax1.set_xlabel('Average', font=fpath, fontsize=32)
            ax1.set_ylabel(ylabel, font=fpath, fontsize=32)
            ax1.set(ylim=ylimits_right)
            ax1.axes_style = 'white'
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_linewidth(2)
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    # if saveimg:
    #     plt.savefig('results_plots/EVC_nvox_distance.pdf')
    #plt.show()

def get_tfce_stats(data, measure='distance', n_perms=10000):
    subxvoxels = df_to_array_tfce(data.groupby(['subject','nvoxels']).mean().reset_index(),
                                  measure=measure)
    if measure == 'correct':
        subxvoxels -= 0.5
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
