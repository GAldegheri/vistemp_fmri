import os

def serialize_args(kwargs):
    """
    Serialize the kwargs into a command line argument string.
    """
    args_str = " ".join(f"--{key} {repr(value)}" for key, value in kwargs.items())
    return args_str

def create_job_script(script_path, args_str, script_name):
    """
    Create a job script file to run a Python script with arguments.
    """
    with open(script_name, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write("#PBS -N job_{}\n".format(os.path.basename(script_name)))
        file.write("#PBS -l walltime=01:00:00,mem=16g\n")  # Adjust walltime as needed
        file.write("#PBS -l nodes=1:ppn=1\n")      # Adjust resources as needed
        file.write("#PBS -q batch\n")              # Adjust queue as needed
        file.write("#PBS -o torque/jobs/output/{}_output.txt\n".format(os.path.basename(script_name)))
        file.write("#PBS -e torque/jobs/output/{}_error.txt\n".format(os.path.basename(script_name)))
        file.write(f"cd /project/3018040.7/Scripts/vistemp_fmri/analysis/\n")
        #file.write("module load anaconda3\n")
        file.write("source activate giacomo37\n")
        #file.write("python -c 'import sys; print(sys.version_info[:])'\n")
        file.write(f"python {script_path} {args_str}\n")

def submit_job(script_path, kwargs, job_name):
    """
    Submit a single job to run a Python script with given kwargs.
    """
    args_str = serialize_args(kwargs)
    script_name = "torque/jobs/{}".format(job_name)
    for k, v in kwargs.items():
        script_name += f"_{k}-{v}"
    script_name += ".sh"
    create_job_script(script_path, args_str, script_name)
    os.system(f"qsub {script_name}")