import pandas as pd
from nipype.interfaces.base import Bunch
import numpy as np

# ---------------------------------------------------------------------------------------------------

def specify_model_funcloc(eventsfile, model):
    '''
    Functional localizer:
    1 - faces, objects, scenes, scrambled, buttonpress
    2 - stimulus, baseline, buttonpress
    3 - object or scrambled, baseline, buttonpress
    '''
    
    events = pd.read_csv(eventsfile, sep='\t')
    
    onsets = []
    durations = []
    
    if model==1:
        
        conditions = ['faces', 'objects', 'scenes', 'scrambled', 'buttonpress']
    
        for r in conditions:
            onsets.append(list(events.loc[events['trial_type'] == r]['onset']))
            durations.append(list(events.loc[events['trial_type'] == r]['duration']))

    elif model==2:
        
        conditions = ['stimulus', 'baseline', 'buttonpress']
        
        # Stimulus:
        onsets.append(list(events.loc[events['trial_type'].isin(['faces', 'objects', 'scenes', 'scrambled'])]['onset']))
        durations.append(list(events.loc[events['trial_type'].isin(['faces', 'objects', 'scenes', 'scrambled'])]['duration']))
        
        # Baseline:
        onsets.append(list(events.loc[events['trial_type']=='blank']['onset']))
        durations.append(list(events.loc[events['trial_type']=='blank']['duration']))
        
        # Buttonpress:
        onsets.append(list(events.loc[events['trial_type']=='buttonpress']['onset']))
        durations.append(list(events.loc[events['trial_type']=='buttonpress']['duration']))
        
    elif model==3:
        
        conditions = ['objscr', 'baseline', 'buttonpress']
        
        # Stimulus:
        onsets.append(list(events.loc[events['trial_type'].isin(['objects', 'scrambled'])]['onset']))
        durations.append(list(events.loc[events['trial_type'].isin(['objects', 'scrambled'])]['duration']))
        
        # Baseline:
        onsets.append(list(events.loc[events['trial_type']=='blank']['onset']))
        durations.append(list(events.loc[events['trial_type']=='blank']['duration']))
        
        # Buttonpress:
        onsets.append(list(events.loc[events['trial_type']=='buttonpress']['onset']))
        durations.append(list(events.loc[events['trial_type']=='buttonpress']['duration']))
    
    else:
        raise Exception('"{:d}" is not a known model.'.format(model))
        
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)
    
    return evs

# ---------------------------------------------------------------------------------------------------

def specify_model_train(eventsfile, model):
    '''
    Models:
    1 - objects, scenes
    2 - bed, couch
    3 - near, far
    4 - wide, narrow
    5 - near, far miniblocks
    6 - wide, narrow miniblocks
    '''
    
    events = pd.read_csv(eventsfile, sep='\t')
    events['rotation'] = events['rotation'].apply(str)
    events['exemplar'] = events['exemplar'].apply(str)
    
    # --------------------------------------
    # Divide in miniblocks
    # --------------------------------------
    stimonly = events[events['trial_type']!='buttonpress'].reset_index() # Get only stimuli (no button presses)
    mb_length = 9 # Number of events in each miniblock
    mblock_indx = {tt: 1 for tt in stimonly.trial_type.unique()}
    
    miniblocks = []
    
    for i, row in stimonly.iterrows():
        
        if i==0 or row.trial_type!=prev_type: # beginning of a new miniblock
            thisblock_end = stimonly.iloc[i+mb_length-1].onset + stimonly.iloc[i+mb_length-1].duration
        
            if row['trial_type']=='object':
                if (row['view']=='A' and row['rotation']=='30') or (row['view']=='B' and row['rotation']=='90'):
                    widenarr = 'wide'
                elif (row['view']=='B' and row['rotation']=='30') or (row['view']=='A' and row['rotation']=='90'):
                    widenarr = 'narrow'
                view = 'None'
                rotation = 'None'
            else: # scene
                widenarr = 'None'
                view = row['view']
                rotation = row['rotation']

            miniblocks.append({'onset': row['onset'], 'duration': thisblock_end - row['onset'], 
                               'trial_type': row['trial_type'],
                               'view': view, 'rotation': rotation, 
                               'exemplar': row['exemplar'], 'widenarr': widenarr,
                               'miniblock': mblock_indx[row['trial_type']]})
            
            mblock_indx[row['trial_type']] += 1
            
        prev_type = row.trial_type
    
    miniblocks = pd.DataFrame(miniblocks)
    
    # --------------------------------------
    # Assign events to miniblocks
    # --------------------------------------
    mblock_indx = {tt: 1 for tt in events.trial_type.unique() if tt != 'buttonpress'}
    events['miniblock'] = ' '
    prev_type = ''
    for i, row in events.iterrows():
        if row.trial_type!=prev_type and i!=0 and row.trial_type!='buttonpress':
            mblock_indx[thistype] += 1
        if row.trial_type=='buttonpress':
            thistype = prev_type
        else:
            thistype = row['trial_type']
        events.loc[i, 'miniblock'] = mblock_indx[thistype]
        prev_type = thistype
    # --------------------------------------
    
    bedscenes = ['2', '5', '6', '7', '9', '13', '14', '17', '18', '20']
    
    onsets = []
    durations = []
    
    if model==1:
        
        conditions = ['objects', 'scenes']
        
        # ------------- Objects: ---------------
        onsets.append(list(miniblocks.loc[miniblocks['trial_type']=='object']['onset']))
        durations.append(list(miniblocks.loc[miniblocks['trial_type']=='object']['duration']))
        # ------------- Scenes: ----------------
        onsets.append(list(miniblocks.loc[miniblocks['trial_type']=='scene']['onset']))
        durations.append(list(miniblocks.loc[miniblocks['trial_type']=='scene']['duration']))
        
    elif model==2:
        
        conditions = ['bed', 'couch']
        
        # ------------- Beds: ---------------
        onsets.append(list(miniblocks.loc[(miniblocks['trial_type']=='object')&(miniblocks['exemplar'].isin(bedscenes))]['onset']))
        durations.append(list(miniblocks.loc[(miniblocks['trial_type']=='object')&(miniblocks['exemplar'].isin(bedscenes))]['duration']))
        # ------------- Couches: ------------
        onsets.append(list(miniblocks.loc[(miniblocks['trial_type']=='object')& ~(miniblocks['exemplar'].isin(bedscenes))]['onset']))
        durations.append(list(miniblocks.loc[(miniblocks['trial_type']=='object')& ~(miniblocks['exemplar'].isin(bedscenes))]['duration']))
        
    elif model==3:
        
        conditions = ['near', 'far']
        
        # ------------ Near: ----------------
        onsets.append(list(miniblocks.loc[(miniblocks['trial_type']=='scene')&(miniblocks['rotation']=='30')]['onset']))
        durations.append(list(miniblocks.loc[(miniblocks['trial_type']=='scene')&(miniblocks['rotation']=='30')]['duration']))
        # ------------- Far: ----------------
        onsets.append(list(miniblocks.loc[(miniblocks['trial_type']=='scene')&(miniblocks['rotation']=='90')]['onset']))
        durations.append(list(miniblocks.loc[(miniblocks['trial_type']=='scene')&(miniblocks['rotation']=='90')]['duration']))
        
    elif model==4:
        
        conditions = ['wide', 'narrow']
        
        # ------------ Wide: -----------------
        onsets.append(list(miniblocks.loc[miniblocks['widenarr']=='wide']['onset']))
        durations.append(list(miniblocks.loc[miniblocks['widenarr']=='wide']['duration']))
        # ------------ Narrow: ---------------
        onsets.append(list(miniblocks.loc[miniblocks['widenarr']=='narrow']['onset']))
        durations.append(list(miniblocks.loc[miniblocks['widenarr']=='narrow']['duration']))
        
    elif model==5:
        
        mblocks_near = miniblocks.loc[(miniblocks['trial_type']=='scene')&(miniblocks['rotation']=='30')]
        mblocks_far = miniblocks.loc[(miniblocks['trial_type']=='scene')&(miniblocks['rotation']=='90')]
        assert(len(mblocks_near)==10 and len(mblocks_far)==10)
        
        n_miniblocks = 10 # per condition
        conditions = []
        for mb in range(1, n_miniblocks+1):
            conditions.append('near_{:d}'.format(mb))
            onsets.append([mblocks_near.iloc[mb-1]['onset']])
            durations.append([mblocks_near.iloc[mb-1]['duration']])
        for mb in range(1, n_miniblocks+1):
            conditions.append('far_{:d}'.format(mb))
            onsets.append([mblocks_far.iloc[mb-1]['onset']])
            durations.append([mblocks_far.iloc[mb-1]['duration']])
    
    elif model==6:
        
        mblocks_wide = miniblocks.loc[(miniblocks['trial_type']=='object')&(miniblocks['widenarr']=='wide')]
        mblocks_narr = miniblocks.loc[(miniblocks['trial_type']=='object')&(miniblocks['widenarr']=='narrow')]
        assert(len(mblocks_wide)==10 and len(mblocks_narr)==10)
        
        n_miniblocks = 10
        conditions = []
        for mb in range(1, n_miniblocks+1):
            conditions.append('wide_{:d}'.format(mb))
            onsets.append(list(mblocks_wide.iloc[mb-1]['onset']))
            durations.append(list(mblocks_wide.iloc[mb-1]['duration']))
        for mb in range(1, n_miniblocks+1):
            conditions.append('narrow_{:d}'.format(mb))
            onsets.append(list(mblocks_narr.iloc[mb-1]['onset']))
            durations.append(list(mblocks_narr.iloc[mb-1]['duration']))
            
    elif model==7:
        # Use 'raw' events instead of miniblocks here
        
        mblocks_A30 = events.loc[(events['view']=='A')&(events['rotation']=='30')]
        mblocks_A90 = events.loc[(events['view']=='A')&(events['rotation']=='90')]
        mblocks_B30 = events.loc[(events['view']=='B')&(events['rotation']=='30')]
        mblocks_B90 = events.loc[(events['view']=='B')&(events['rotation']=='90')]
        
        conditions = []
        
        for i, mb in enumerate(mblocks_A30.miniblock.unique()): # for each individual miniblock where that trial type has occurred
            conditions.append('A30_{:d}'.format(i+1))
            onsets.append(list(mblocks_A30[mblocks_A30['miniblock']==mb]['onset']))
            durations.append([0]*len(mblocks_A30[mblocks_A30['miniblock']==mb]))
        
        for i, mb in enumerate(mblocks_A90.miniblock.unique()):
            conditions.append('A90_{:d}'.format(i+1))
            onsets.append(list(mblocks_A90[mblocks_A90['miniblock']==mb]['onset']))
            durations.append([0]*len(mblocks_A90[mblocks_A90['miniblock']==mb]))
        
        for i, mb in enumerate(mblocks_B30.miniblock.unique()):
            conditions.append('B30_{:d}'.format(i+1))
            onsets.append(list(mblocks_B30[mblocks_B30['miniblock']==mb]['onset']))
            durations.append([0]*len(mblocks_B30[mblocks_B30['miniblock']==mb]))
        
        for i, mb in enumerate(mblocks_B90.miniblock.unique()):
            conditions.append('B90_{:d}'.format(i+1))
            onsets.append(list(mblocks_B90[mblocks_B90['miniblock']==mb]['onset']))
            durations.append([0]*len(mblocks_B90[mblocks_B90['miniblock']==mb]))
    
    elif model==9:
        A30 = events.loc[(events['view']=='A')&(events['rotation']=='30')]
        A90 = events.loc[(events['view']=='A')&(events['rotation']=='90')]
        B30 = events.loc[(events['view']=='B')&(events['rotation']=='30')]
        B90 = events.loc[(events['view']=='B')&(events['rotation']=='90')]
        
        conditions = ['A30', 'A90', 'B30', 'B90']
        
        onsets.append(list(A30['onset']))
        durations.append([0]*len(A30))
        
        onsets.append(list(A90['onset']))
        durations.append([0]*len(A90))
        
        onsets.append(list(B30['onset']))
        durations.append([0]*len(B30))
        
        onsets.append(list(B90['onset']))
        durations.append([0]*len(B90))
    
    else:
        raise Exception('"{:d}" is not a known model.'.format(model))
    
    if events['trial_type'].str.contains('buttonpress').any(): # if there's at least one response
        conditions.append('buttonpress')
        onsets.append(list(events.loc[events['trial_type']=='buttonpress']['onset']))
        durations.append(list(events.loc[events['trial_type']=='buttonpress']['duration']))
    
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)
        
    return evs
# ---------------------------------------------------------------------------------------------------

def specify_model_test(eventsfile, model, behav):
    '''
    Models:
    1 - bed, couch
    2 - near, far
    3 - initial wide, narrow
    4 - imagined wide, narrow
    5 - final wide, narrow (few stimulus appearances)
    6 - A30, A90, B30, B90
    
    Events:
    1 - Fixation (1.5 sec)
    2 - Scene view 1 (2 sec)
    3 - Scene view 2 (0.5 sec)
    4 - Scene view 3 (0.5 sec) **
    5 - Scene view 4 (0.5 sec) **
    6 - (Final) view 5 (1.5-2 sec) **
    (( 7 - Target object (0.2 sec) ))
    8 - Scene offset
    
    ** = occluder
    
    '''
    
    events = pd.read_csv(eventsfile, sep='\t')

    bedscenes = [2, 5, 6, 7, 9, 13, 14, 17, 18, 20]
    
    onsets = []
    durations = []
    
    if model==1:
        
        conditions = ['bed', 'couch']
        bedindx = behav.index[behav['scene'].isin(bedscenes)]
        bedtrials = events[events['trial_no'].isin(bedindx)]
        couchtrials = events[~events['trial_no'].isin(bedindx)]
        onsets.append(list(bedtrials[bedtrials['event_no']==2].onset))
        onsets.append(list(couchtrials[couchtrials['event_no']==2].onset))
        durations.append(list(bedtrials[bedtrials['event_no']==4].onset.values - bedtrials[bedtrials['event_no']==2].onset.values))
        durations.append(list(couchtrials[couchtrials['event_no']==4].onset.values - couchtrials[couchtrials['event_no']==2].onset.values))
        
    elif model==2:
        
        conditions = ['near', 'far']
        nearindx = behav.index[behav['finalview']==30]
        neartrials = events[events['trial_no'].isin(nearindx)]
        fartrials = events[~events['trial_no'].isin(nearindx)]
        
        onsets.append(list(neartrials[neartrials['event_no']==6].onset))
        onsets.append(list(fartrials[fartrials['event_no']==6].onset))
        
        durations.append(list(neartrials[neartrials['event_no']==8].onset.values - \
                              neartrials[neartrials['event_no']==6].onset.values))
        durations.append(list(fartrials[fartrials['event_no']==8].onset.values - \
                              fartrials[fartrials['event_no']==6].onset.values))
        
    elif model==3:
        # initial wide/narrow (just A/B)
        
        conditions = ['wide', 'narrow']
        wideindx = behav.index[behav['initpos']==1]
        widetrials = events[events['trial_no'].isin(wideindx)]
        narrtrials = events[~events['trial_no'].isin(wideindx)]
        
        onsets.append(list(widetrials[widetrials['event_no']==2].onset))
        onsets.append(list(narrtrials[narrtrials['event_no']==2].onset))
        
        durations.append(list(widetrials[widetrials['event_no']==3].onset.values - \
                              widetrials[widetrials['event_no']==2].onset.values))
        durations.append(list(narrtrials[narrtrials['event_no']==3].onset.values - \
                              narrtrials[narrtrials['event_no']==2].onset.values))
    elif model==4:
        
        conditions = ['wide', 'narrow']
        imwideindx = ((behav['initpos']==1) & (behav['finalview']==30)) | ((behav['initpos']==2) & (behav['finalview']==90))
        imwideindx = behav.index[(imwideindx) & (behav['target']==0)] # exclude actual stimulus reappearances
        imnarrindx = ((behav['initpos']==2) & (behav['finalview']==30)) | ((behav['initpos']==1) & (behav['finalview']==90))
        imnarrindx = behav.index[(imnarrindx) & (behav['target']==0)]
        
        imwidetrials = events[events['trial_no'].isin(imwideindx)]
        imnarrtrials = events[events['trial_no'].isin(imnarrindx)]
        
        onsets.append(list(imwidetrials[imwidetrials['event_no']==6].onset))
        onsets.append(list(imnarrtrials[imnarrtrials['event_no']==6].onset))
        
        durations.append(list(imwidetrials[imwidetrials['event_no']==8].onset.values - \
                              imwidetrials[imwidetrials['event_no']==6].onset.values))
        durations.append(list(imnarrtrials[imnarrtrials['event_no']==8].onset.values - \
                              imnarrtrials[imnarrtrials['event_no']==6].onset.values))
        
    elif model==5:
        # **reappeared** wide/narrow (not view-specific)
        
        conditions = ['wide', 'narrow']
        imwideindx = ((behav['initpos']==1) & (behav['finalview']==30)) | ((behav['initpos']==2) & (behav['finalview']==90))
        imwideindx = behav.index[(imwideindx) & (behav['target']==1)] # ONLY INCLUDE actual stimulus reappearances
        imnarrindx = ((behav['initpos']==2) & (behav['finalview']==30)) | ((behav['initpos']==1) & (behav['finalview']==90))
        imnarrindx = behav.index[(imnarrindx) & (behav['target']==1)]
        
        imwidetrials = events[events['trial_no'].isin(imwideindx)]
        imnarrtrials = events[events['trial_no'].isin(imnarrindx)]
        
        onsets.append(list(imwidetrials[imwidetrials['event_no']==6].onset))
        onsets.append(list(imnarrtrials[imnarrtrials['event_no']==6].onset))
        
        durations.append(list(imwidetrials[imwidetrials['event_no']==8].onset.values - \
                              imwidetrials[imwidetrials['event_no']==6].onset.values))
        durations.append(list(imnarrtrials[imnarrtrials['event_no']==8].onset.values - \
                              imnarrtrials[imnarrtrials['event_no']==6].onset.values))
        
    elif model==6:
        
        conditions = ['A30', 'A90', 'B30', 'B90']
        
        A30mask = ((behav['initpos']==1) & (behav['finalview']==30))
        A90mask = ((behav['initpos']==1) & (behav['finalview']==90))
        B30mask = ((behav['initpos']==2) & (behav['finalview']==30))
        B90mask = ((behav['initpos']==2) & (behav['finalview']==90))
        
        A30indx = behav.index[(A30mask) & (behav['target']==0)]
        A90indx = behav.index[(A90mask) & (behav['target']==0)]
        B30indx = behav.index[(B30mask) & (behav['target']==0)]
        B90indx = behav.index[(B90mask) & (behav['target']==0)]
        
        A30trials = events[events['trial_no'].isin(A30indx)]
        A90trials = events[events['trial_no'].isin(A90indx)]
        B30trials = events[events['trial_no'].isin(B30indx)]
        B90trials = events[events['trial_no'].isin(B90indx)]
        
        onsets.append(list(A30trials[A30trials['event_no']==6].onset))
        onsets.append(list(A90trials[A90trials['event_no']==6].onset))
        onsets.append(list(B30trials[B30trials['event_no']==6].onset))
        onsets.append(list(B90trials[B90trials['event_no']==6].onset))
        
        durations.append(list(A30trials[A30trials['event_no']==8].onset.values - \
                              A30trials[A30trials['event_no']==6].onset.values))
        durations.append(list(A90trials[A90trials['event_no']==8].onset.values - \
                              A90trials[A90trials['event_no']==6].onset.values))
        durations.append(list(B30trials[B30trials['event_no']==8].onset.values - \
                              B30trials[B30trials['event_no']==6].onset.values))
        durations.append(list(B90trials[B90trials['event_no']==8].onset.values - \
                              B90trials[B90trials['event_no']==6].onset.values))
        
    elif model==8:
        
        conditions = ['A30', 'A90', 'B30', 'B90']
        
        A30mask = ((behav['initpos']==1) & (behav['finalview']==30))
        A90mask = ((behav['initpos']==1) & (behav['finalview']==90))
        B30mask = ((behav['initpos']==2) & (behav['finalview']==30))
        B90mask = ((behav['initpos']==2) & (behav['finalview']==90))
        
        A30indx = behav.index[(A30mask) & (behav['target']==1)]
        A90indx = behav.index[(A90mask) & (behav['target']==1)]
        B30indx = behav.index[(B30mask) & (behav['target']==1)]
        B90indx = behav.index[(B90mask) & (behav['target']==1)]
        
        A30trials = events[events['trial_no'].isin(A30indx)]
        A90trials = events[events['trial_no'].isin(A90indx)]
        B30trials = events[events['trial_no'].isin(B30indx)]
        B90trials = events[events['trial_no'].isin(B90indx)]
        
        onsets.append(list(A30trials[A30trials['event_no']==6].onset))
        onsets.append(list(A90trials[A90trials['event_no']==6].onset))
        onsets.append(list(B30trials[B30trials['event_no']==6].onset))
        onsets.append(list(B90trials[B90trials['event_no']==6].onset))
        
        durations.append(list(A30trials[A30trials['event_no']==8].onset.values - \
                              A30trials[A30trials['event_no']==6].onset.values))
        durations.append(list(A90trials[A90trials['event_no']==8].onset.values - \
                              A90trials[A90trials['event_no']==6].onset.values))
        durations.append(list(B30trials[B30trials['event_no']==8].onset.values - \
                              B30trials[B30trials['event_no']==6].onset.values))
        durations.append(list(B90trials[B90trials['event_no']==8].onset.values - \
                              B90trials[B90trials['event_no']==6].onset.values))
    
    elif model==9:
        
        conditions = ['target', 'omission']
        
        tgtindx = behav.index[behav['target']==1]
        omsindx = behav.index[behav['target']==0]
        
        tgtevents = events[(events['trial_no'].isin(tgtindx))&(events['event_no']==7)]
        omsevents = events[(events['trial_no'].isin(omsindx))&(events['event_no']==8)]
        
        onsets.append(list(tgtevents.onset))
        onsets.append(list(omsevents.onset))
        
        durations.append([0]*len(tgtevents))
        durations.append([0]*len(omsevents))
    
    else:
        raise Exception('"{:d}" is not a known model.'.format(model))
        
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)
        
    return evs

################################## Single-trial betas #######################################

def Specify_TrialBetas(trialinfo):
    '''
    Input: trialinfo - Bunch containing:
                        - nifti
                        - evfile
                        - trial
                        - event
                        - motion
    -------------------------------------------
    Events:
    1 - Fixation (0.5 - 1.5 sec depending on previous trial's response time)
    2 - Scene view 1 (2 sec)
    3 - Scene view 2 (0.5 sec)
    4 - Scene view 3 (0.5 sec) **
    5 - Scene view 4 (0.5 sec) **
    6 - (Final) view 5 (1-1.5 sec) **
    7 - Probe 1 (0.05 sec)
    8 - ISI (0.1 sec)
    9 - Probe 2 (0.05 sec)
    10 - Probe offset/pre-response delay (0.05 sec)
    11 - Response window starts (until response, max 1.5 sec)
    12 - Response

    ** = occluder present

    '''
    
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch
    
    events = pd.read_csv(trialinfo.evfile, sep='\t')
    
    tr = trialinfo.trial
    ev = trialinfo.event
    
    conditions = ['thistrial', 'othertrials']
    #exclude_events = [8, 9, 10, 11, 12] # to be excluded from 'other' list because they shouldn't
                                # be decorrelated - they're basically continuations of the event of interest!
                                # maybe add 11 & 12 too?
    
    include_events = [2, 7]
    
    thistrial = events[events['trial_no']==tr]

    # two relevant events: initial view (2) and final view (7)

    theseonsets = [o for o in thistrial[thistrial['event_no']==ev].onset.values]
    thesedurations = []

    if ev==2: # initial view (event 2, lasts until onset of event 3)
        thesedurations.append(thistrial[thistrial['event_no']==3].onset.values[0] - thistrial[thistrial['event_no']==2].onset.values[0])

    elif ev==7: # final view (event 7, duration 0 - very brief)
        thesedurations.append(0)

    # loop through all other trials
    otheronsets = []
    otherdurations = []
    for othertr in events.trial_no.unique():
        for otherev in include_events: #events.event_no.unique():
            if not ((tr==othertr) & (ev==otherev)):
                # everything except this particular trial and event
                othertrial = events[events['trial_no']==othertr]
                otherons = othertrial[othertrial['event_no']==otherev].onset.values[0]

                if not np.isnan(otherons):
                    otheronsets.append(otherons)
                    if otherev != np.max(events.event_no.unique()):
                        otherdur = othertrial[othertrial['event_no']==otherev+1].onset.values[0] - othertrial[othertrial['event_no']==otherev].onset.values[0]
                    elif othertr != np.max(events.trial_no.unique()):
                        otherdur = events[(events['trial_no']==othertr+1)&(events['event_no']==1)].onset.values - othertrial[othertrial['event_no']==otherev].onset.values
                    else: # Last event of last trial
                        otherdur = 0

                    if otherdur < 1.0 or np.isnan(otherdur): 
                        otherdur = 0
                    otherdurations.append(otherdur)
        # for debugging:
        #otheronsets.append("--------------------")
        #otherdurations.append("--------------------")

    onsets = [theseonsets, otheronsets]
    durations = [thesedurations, otherdurations]
                            
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)

    return [evs], [trialinfo.nifti], trialinfo.motion, tr, ev


def Get_First_Beta(beta_images):
    '''
    Stupid utility to get the first of a list of beta images
    '''
    
    return beta_images[0]

def Rename_TrialBetas(in_file, trialno, eventno):
    
    import os
    from glob import glob
    
    newfile = 'trial-{:03d}_ev-{:g}.nii'.format(trialno, eventno)
    newfile = os.path.join(os.path.split(in_file)[0], newfile)
    os.rename(in_file, newfile)
    
    # remove useless betas:
    allbetas = glob(os.path.join(os.path.split(in_file)[0], 'beta_*.nii'))
    if len(allbetas) != 0:
        for b in allbetas:
            os.remove(b)
    
    return newfile
    
