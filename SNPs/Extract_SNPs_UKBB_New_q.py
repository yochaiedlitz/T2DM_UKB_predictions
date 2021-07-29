import PRS_sumstats as PRS
import os
import pandas as pd
import pickle
import numpy as np
from LabData import config_global as config
from LabUtils.addloglevels import sethandlers
from LabQueue.qp import fakeqp

import os
Job_Name = 'demo_amit'
dbg = True

if dbg:
    USE_FAKE_QUE = True
    Job_Name = Job_Name+"dbg"
    Num_of_Chrs = 2
else:
    USE_FAKE_QUE = False
    Num_of_Chrs = 22

Basic_Path='/net/mraid08/export/jafar/Microbiome/Analyses/Biobank/Bio_Gen/' #Location of UKBB Chromosomes files
Dict_Folder='/net/mraid08/export/jafar/Yochai/PRS/PRS_Results/'

Results_Folder=Dict_Folder+Job_Name+"/"
Final_PRS=pd.DataFrame()
NJOBS=1
N_THREADS=1
CSV_Dict={}
uniques = {}

with open(Dict_Folder + "Orig_trait_dict", 'rb') as fp:
    Top_P_Dict=pickle.load(fp)

if USE_FAKE_QUE:
    from queue_tal.qp import fakeqp as qp
else:
    from queue_tal.qp import qp

if not os.path.exists(Results_Folder):
    os.mkdir(Results_Folder)
if not os.path.exists(Results_Folder+"Final_Results/"):
    os.mkdir(Results_Folder+"Final_Results/")


    with qp(jobname='prs_slp', q=['himem7.q'], _trds_def=2, _mem_def='1G', _tryrerun=True, max_r=300) as q:
        q.startpermanentrun()
        waiton = []

def upload_jobs(q):
    waiton = []
    for CHR_Num in np.arange(1,Num_of_Chrs+1,1):
        bfile_path = Basic_Path+"ukb_snp_chr"+str(CHR_Num)+"_v2"
        print("Chromosome number:", str(CHR_Num)," sent to Que")
        waiton.append(q.method(PRS.extract_relevant_SNPS,
                               tuple([Top_P_Dict]+[bfile_path]+[Results_Folder]+[Job_Name]+[str(CHR_Num)])))
        if CHR_Num == Num_of_Chrs:
            q.waitforresults(waiton)

    for CHR_Num in np.arange(1,Num_of_Chrs+1,1):
        bfile_path = Basic_Path+"ukb_snp_chr"+str(CHR_Num)+"_v2"
        for trait in Top_P_Dict.iterkeys():
            print(trait,"in:",str(CHR_Num))
            if CHR_Num==1:
                CSV_Dict[trait] =pd.read_csv(Results_Folder + trait + "_" + str(CHR_Num) + "_.csv")
                CSV_Dict[trait].set_index("eid",inplace=True,drop=True)
            else:
                temp_csv=pd.read_csv(Results_Folder + trait + "_" + str(CHR_Num) + "_.csv")
                temp_csv.set_index("eid", inplace=True, drop=True)
                CSV_Dict[trait] = pd.merge(CSV_Dict[trait], temp_csv, how='outer',left_index=True,right_index=True)

    for trait in Top_P_Dict.iterkeys():
        CSV_Dict[trait].to_csv(path_or_buf=Results_Folder + "Final_Raw_SNPs" + trait + ".csv",index=True)

    ind = 0
    for trait in Top_P_Dict.iterkeys():
        waiton.append(q.method(PRS.Convert_to_Class,tuple([trait]+[Results_Folder])))
        if ind == (len(Top_P_Dict)-1):
            q.waitforresults(waiton)
        ind += 1
    print ("Finito la Comedia, files saved at: ", Results_Folder)

def main():
    sethandlers(file_dir=config.log_dir)
    os.chdir('/net/mraid08/export/genie/LabData/Analyses/Yochai/Jobs')
    with qp(jobname=Job_Name, max_u=600/(NJOBS*N_THREADS), mem_def='1G', trds_def=NJOBS*N_THREADS, q=['himem7.q']) as q:
        q.startpermanentrun()
        upload_jobs(q)

main()