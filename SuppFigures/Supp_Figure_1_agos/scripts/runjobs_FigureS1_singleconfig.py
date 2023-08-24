
import os
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random


def runjobs():
    """
        Function to be run in a Sun Grid Engine queuing system. For testing the output, run it like
        python runjobs.py --test 1
        
    """
    
    #--------------------------------------------------------------------------
    # Test commands option
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, type=int, default=0)
    parser.add_argument("--cluster_", help=" String", default='burg')
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])

    
    if (args2.test):
        print ("testing commands")
    
    #--------------------------------------------------------------------------
    # Which cluster to use

    
    if platform=='darwin':
        cluster='local'
    
    currwd = os.getcwd()
    print(currwd)
    #--------------------------------------------------------------------------
    # Ofiles folder
        
    if cluster=='haba':
    
        path_2_package="./../"
        ofilesdir = path_2_package + "/Ofiles/"


    if cluster=='burg':
        path_2_package="./../"
        ofilesdir = path_2_package + "/Ofiles/"
        
        
    elif cluster=='local':
        path_2_package="./../"
        ofilesdir = path_2_package+"/Ofiles/"


    if not os.path.exists(ofilesdir):
        os.makedirs(ofilesdir)
    

    
    #--------------------------------------------------------------------------
    # The array of hashes
    nrates=10
    nJs=5
    r_X_Vec=10**(np.linspace(0.1,2,nrates))
    J_Vec=1e-3*np.array([1.000000,1.97932382]) #np.exp(np.linspace(-9,-5.3,nJs))
    nrep_Vec=np.arange(0,50,1)
    tensor=1

    results_files = os.listdir('./../Paper_Results')
    
    for nrep in nrep_Vec:
        for rX in r_X_Vec:
            for J in J_Vec:

                args = {
                    'rX': round(rX,6),
                    'J': round(J,6),
                    'nrep': nrep,
                    'tensor': True
                }
                file = 'FigureS1_nrep='+str(nrep)+'-'.join([kk+'_'+str(ll) for kk,ll in args.items()])+'.pkl'
                if file in results_files: continue

                time.sleep(0.2)
                
                #--------------------------------------------------------------------------
                # Make SBTACH
                inpath = currwd + "/FigureS1_singleconfig.py"
                c1 = "{:s} -nrep {:d} -rX {:f} -J {:f} -tensor {:d}".format(inpath, nrep, rX, J, tensor)
                
                jobname="FigureS1-nrep-{:d}-rX_{:.6f}-J_{:.6f}".format(nrep, rX, J)
                
                if not args2.test:
                    jobnameDir=os.path.join(ofilesdir, jobname)
                    text_file=open(jobnameDir, "w");
                    os. system("chmod u+x "+ jobnameDir)
                    text_file.write("#!/bin/sh \n")
                    if cluster=='burg':
                        text_file.write("#SBATCH --account=theory \n")
                    if cluster=='axon':
                        text_file.write("#SBATCH --partition=ctn \n")
                    text_file.write("#SBATCH --job-name="+jobname+ "\n")
                    text_file.write("#SBATCH -t 0-11:59  \n")
                    text_file.write("#SBATCH --gres=gpu\n")
                    text_file.write("#SBATCH --mem-per-cpu=12gb \n")
                    text_file.write("#SBATCH -c 1 \n")
                    text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
                    text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
                    if cluster=='burg':
                        text_file.write("module load anaconda \n")
                    elif cluster=='axon':
                        text_file.write("ml anaconda3-2019.03 \n")
                    text_file.write("python  -W ignore " + c1+" \n")
                    text_file.write("echo $PATH  \n")
                    text_file.write("exit 0  \n")
                    text_file.close()

                    os.system("sbatch " +jobnameDir);
                else:
                    print (c1)



if __name__ == "__main__":
    runjobs()


