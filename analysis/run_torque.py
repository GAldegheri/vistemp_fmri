from torque.torque_funcs import submit_job

if __name__=="__main__":
    subjlist = [f'sub-{i:03d}' for i in range(1, 35)]
    for s in subjlist:
        submit_job("/project/3018040.07/Scripts/vistemp_fmri/analysis/infocoupling_FIR_main_nonipype.py", 
                   {"sub": s, "roi": "ba-19-37_L_contr-objscrvsbas_top-3000_nothresh"}, "run_infoFIR")