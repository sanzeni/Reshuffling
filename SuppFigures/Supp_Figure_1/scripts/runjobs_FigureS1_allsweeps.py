
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
    nrep_Vec=range(50)
    nrates=10
    nJs=5
    
    for nrep in nrep_Vec:

        time.sleep(0.2)
        
        #--------------------------------------------------------------------------
        # Make SBTACH
        inpath = currwd + "/send_cluster_FigureS1.py"
        c1 = "{:s} -nrep {:d} -nrates {:d} -nJs {:d} ".format(inpath, nrep,nrates,nJs)
        
        jobname="send_cluster_FigureS1"+"-nrep-{:d}".format(nrep)
        
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


