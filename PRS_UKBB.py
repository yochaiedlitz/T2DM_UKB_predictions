from .PRS import PRS_sumstats as PRS
import os
import pandas as pd
import numpy as np
from addloglevels import sethandlers

Job_Name = 'No_UKBB_PRS'
dbg = True

if dbg:
    USE_FAKE_QUE = True
    Job_Name = Job_Name+"dbg"
    Num_of_Chrs = 2
else:
    USE_FAKE_QUE = False
    Num_of_Chrs = 22

Basic_Path='/net/mraid08/export/jafar/Microbiome/Analyses/Biobank/Bio_Gen/' #Location of UKBB Chromosomes files
Results_Folder="/net/mraid08/export/genie/Data/Yochai/PRS/PRS_Results/"+Job_Name+"/"
Final_PRS=pd.DataFrame()
NJOBS=1
N_THREADS=1

if USE_FAKE_QUE:
    from queue_tal.qp import fakeqp as qp
else:
    from queue_tal.qp import qp

if not os.path.exists(Results_Folder):
    os.mkdir(Results_Folder)

def upload_jobs(q):

    # bed_file_path_dict = {}
    # bim_file_path_dict = {}
    # fam_file_path_dict = {}
    #
    # df_bim_dict={}
    waiton = []

    for CHR_Num in np.arange(1,Num_of_Chrs+1,1): #Run over all chromosomes
        bfile_path = Basic_Path+"ukb_snp_chr"+str(CHR_Num)+"_v2"
        print("Chromosome number:", str(CHR_Num)," sent to Que")
        waiton.append(q.method(PRS.get_UKBB_predictions,
                               tuple([bfile_path]+[Results_Folder]+[Job_Name]+[str(CHR_Num)])))#Predict score per for chromosome
        if CHR_Num == Num_of_Chrs:
            q.waitforresults(waiton)

    for CHR_Num in np.arange(1, Num_of_Chrs+1, 1):
        if CHR_Num == 1:
            Final_PRS=pd.read_csv(Results_Folder+Job_Name+"_CHR_"+str(CHR_Num)+".csv", index_col="eid")
        else:
            Final_PRS = Final_PRS.add(pd.read_csv(Results_Folder+Job_Name+"_CHR_"+str(CHR_Num)+".csv",index_col="eid"), fill_value=0)

    Final_PRS.to_csv(Results_Folder+Job_Name+"_Final_Result.csv")

    print("Finito la Comedia, files saved at: ",Results_Folder)

def main():

    with qp(jobname=Job_Name, max_u=600/(NJOBS*N_THREADS), mem_def='150G', trds_def=NJOBS*N_THREADS, q=['himem7.q']) as q:
        os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
        q.startpermanentrun()
        upload_jobs(q)

sethandlers()
main()