from PRS_sumstats import *
from addloglevels import sethandlers
import os

USE_FAKE_QUE=True
full_bfile_path = "/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/cleanData/PNP_autosomal_clean2_nodfukim"
JOB_NAME = 'dbg_prs'
NJOBS=1
N_THREADS=2

if USE_FAKE_QUE:
    from queue_tal.qp import fakeqp as qp
else:
    from queue_tal.qp import qp
# extract phenotypes IID Vs. Measured Phenotypes
# df_pheno = PRS_extract_phenotypes.extract('s_stats_pheno') #Used for training set

# extract the predicted trait


def upload_jobs(q):
    waiton = []
    TempPDFsQ = []
    traits_dict = get_traits_dict()  # Dictionary with all Traits
    # pheno_col = traits_dict[trait]
    df_prs = compute_prs(bfile_path=full_bfile_path)

    df_prs.hist(bins=100)

    bed = read_bfile_forsumstats(bfile_path)
    df_bim = pd.read_csv(bfile_path + '.bim', delim_whitespace=True, header=None,
                         names=['chr', 'rs', 'cm', 'bp', 'a1', 'a2'])  # List of al SNPS
    df_bed = pd.DataFrame(bed.sid, columns=['rs'])  # SNP names
    df_bed = df_bed.merge(df_bim, how='left', on='rs')

    df_predictions = pd.DataFrame(index=bed.iid[:, 1].astype(np.int))

    print ("Finished")
    for SN in range(HYP_PAR_ITER):
        df_predictions = get_predictions(bfile_path)

        Params_list = Choose_params(Hyp_Param_Dict)
        print(" SN: ", SN)
        waiton.append(q.method(Predict, tuple([SN] + Params_list)))

    q.waitforresults(waiton)

    if not os.path.exists(FINAL_RESULTS_FOLDER):
        os.mkdir(FINAL_RESULTS_FOLDER)

    Hyp_Param_Dict['num_threads'] = [20]

    waiton.append(q.method(Sort_AUC_APS))
    q.waitforresults(waiton)
    # print_to_file(result)

    try:
        shutil.rmtree(SAVE_TO_FOLDER)
    except OSError as e:
        print(("Error: %s - %s." % (e.filename, e.strerror)))


def main():

    with qp(jobname=JOB_NAME, max_u=299/(NJOBS*N_THREADS), mem_def='3G', trds_def=NJOBS*N_THREADS, q=['himem7.q']) as q:
        os.chdir('/net/mraid08/export/jafar/Microbiome/Analyses/Edlitzy/tempq_files/')
        q.startpermanentrun()
        upload_jobs(q)

sethandlers()
main()