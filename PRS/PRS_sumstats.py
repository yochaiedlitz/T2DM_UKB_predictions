import numpy as np
import os.path
import pandas as pd
import sys
import time
import os
from bisect import bisect
import pickle
pd.set_option('display.width', 1000)
np.set_printoptions(precision=4, linewidth=200)
from pysnptools.snpreader.bed import Bed

from sklearn.model_selection import KFold
import scipy.stats as stats

CLEAN_DATA='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/cleanData'
TEMP_DATA='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/rawData/tmp'
PCA_DIR='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/PCA'
RAWDATA_DIR='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/rawData'
GCTA_PATH='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/Analysis/gcta'
GCTA_SUMSTATS_PATH='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/Analysis/gcta/sumstats'
# SUMSTATS_DIR1 = '/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/sumstats'
SUMSTATS_DIR_New= '/net/mraid08/export/jafar/Yochai/sumstats'
SUMSTATS_DIR = '/net/mraid08/export/jafar/Yochai/Orig_sumstats/'
PRS_P_Sort_Dict='/net/mraid08/export/jafar/Yochai/PRS/PRS_Results/Orig_trait_dict"'
Gen_DIR = "/net/mraid08/export/jafar/Yochai/PRS/PRS_Results/Extract_1K_SNPs_UKBB/Final_Results/"
PKL_PATH = os.path.join(GCTA_PATH, 'df_PRS_NETO_predictions.pkl')
Quant_PATH=os.path.join(GCTA_PATH, 'df_PRS_NETO_quantile.pkl')
if not os.path.exists(GCTA_SUMSTATS_PATH): os.makedirs(GCTA_SUMSTATS_PATH)

PVAL_CUTOFFS = [1.1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
#PVAL_CUTOFFS = [1.1, 1e-1, 1e-2, 1e-3, 1e-4]


def read_bfile_forsumstats(bfile_path):
    """read plink file and allele frequencies from a summary statistics file
    merginh SNPs from bed file with the ones fom summary statistics
    performing Binomical distibution average, consider using external imputations. There is an imputation file
    standardize SNPs using external MAfs
    """
    bed = Bed(bfile_path+".bed", count_A1=True)    #read plink file and allele frequencies from a summary statistics file
    bed_snps = pd.DataFrame(bed.sid, columns=['MarkerName'])
    files_dict = get_files_dict()
    df_mafs = pd.read_csv(files_dict['height'], delim_whitespace=True, usecols=['MarkerName', 'Freq.Allele1.HapMapCEU'])#Minor allile frequencies
    df_mafs = bed_snps.merge(df_mafs, on='MarkerName', how='left')#merginh SNPs from bed file with the ones fom summary statistics
    assert (df_mafs['MarkerName'] == bed_snps['MarkerName']).all()
    snps_to_keep = df_mafs['Freq.Allele1.HapMapCEU'].notnull()
    bed = bed[:, snps_to_keep].read() #Reads the SNP values and returns a .SnpData (with .SnpData.val property containing a new ndarray of the SNP values).
    df_mafs = df_mafs.ix[snps_to_keep, :]    
    allele_freqs = df_mafs['Freq.Allele1.HapMapCEU'].values

    #impute SNPs according to external MAFs    
    print ('imputing SNPs using external MAFs...')
    isNan = np.isnan(bed.val)
    for i in xrange(bed.sid.shape[0]):
        bed.val[isNan[:,i], i] = 2*allele_freqs[i] #Binomical distibution average, consider using external imputations. There is an imputation file
        
    #standardize SNPs using external MAfs
    print ('standardizing SNPs using external MAFs...')
    snpsMean = 2*allele_freqs
    snpsStd = np.sqrt(2*allele_freqs*(1-allele_freqs))
    snpsStd[snpsStd==0] = np.inf #Probably not an SNP
    bed.val -= snpsMean
    ###bed.val /= snps Std #not clear what did the people who calculated the summary statistics did
    return bed

def get_files_dict():
    """Dictionary with paths to different PRS summary statistics"""
    files_dict = dict([])
    files_dict['height'] = os.path.join(SUMSTATS_DIR, 'height',
                                        'GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt')
    #For metabolon
    files_dict["CARDIoGRAM_GWAS"] = os.path.join(SUMSTATS_DIR, 'CARDIO_Yeela', 'CARDIoGRAM_GWAS_RESULTS.txt')#For Metabolon

    files_dict['alzheimer'] = os.path.join(SUMSTATS_DIR, 'Alzheimer',
                                           'IGAP_stage_1_2_combined.txt')  # Jean-Charles Lambert et al.
    files_dict['bmi'] = os.path.join(SUMSTATS_DIR, 'bmi',
                                     'SNP_gwas_mc_merge_nogc.tbl.uniq')  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4382211/
    files_dict['anorexia'] = os.path.join(SUMSTATS_DIR, 'Anorexia',
                                          'gcan_meta.out')  # A genome-wide association study of anorexia nervosa,https://www.nature.com/articles/mp2013187
    # TODO: check for Asthma pvalue
    # files_dict['ashtma'] = os.path.join(SUMSTATS_DIR, 'Ashtma','gabriel_asthma_meta-analysis_36studies_format_repository_NEJM.txt')  # https://www.cnrgh.fr/gabriel/study_description.html
    files_dict['t2d_mega_meta'] = os.path.join(SUMSTATS_DIR, 't2d',
                                              'diagram.mega-meta.txt')  # FKA iris Trans-ethnic T2D GWAS meta-analysis, http://diagram-consortium.org/downloads.html
    files_dict['cardio'] = os.path.join(SUMSTATS_DIR, 'Cardio',
                                        'cardiogramplusc4d_data.txt')  # CARDIoGRAMplusC4D Metabochip is a two stage meta-analysis of Metabochip and GWAS studies of European and South Asian descent involving 63,746 cases and 130,681 controls. The CARDIoGRAM GWAS data was used as Stage 1 - data as published in: CARDIoGRAMplusC4D Consortium, Deloukas P, Kanoni S, Willenborg C, Farrall M, Assimes TL, Thompson JR, et al. Large-scale association analysis identifies new risk loci for coronary artery disease. Nat Genet 2013 45:25-33
    files_dict['hips'] = os.path.join(SUMSTATS_DIR, 'hips',
                                      'GIANT_2015_HIP_COMBINED_EUR.txt')  # https://www.nature.com/articles/nature14132,https://portals.broadinstitute.org/collaboration/giant/index.php/GIANT_consortium_data_files
    files_dict['waist'] = os.path.join(SUMSTATS_DIR, 'waist',
                                       'GIANT_2015_WC_COMBINED_EUR2.txt')  # https://www.nature.com/articles/nature14132,https://portals.broadinstitute.org/collaboration/giant/index.php/GIANT_consortium_data_files
    #TODO:Clean the data below
    # files_dict["whr_WHR_COMBINED_EUR2"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_2015_WHR_COMBINED_EUR2.txt')
    # files_dict["whr_WHRadjBMI_COMB_All"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_2015_WHRadjBMI_COMBINED_AllAncestries.txt')
    # files_dict["whr_WHRadjBMI_COMB_EUR"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_2015_WHRadjBMI_COMBINED_EUR.txt')
    # files_dict["whr_WHR_COMBINED_All"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_2015_WHR_COMBINED_AllAncestries.txt')
    # files_dict["whr_WHR_COMBINED_EUR"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_2015_WHR_COMBINED_EUR.txt')
    # files_dict["whr_WHR_FEMALES_EUR"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_2015_WHR_FEMALES_EUR.txt')
    # files_dict["whr_WHR_MALES_EUR"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_2015_WHR_MALES_EUR.txt')
    # files_dict["whr_WHR_MEN_N"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_WHR_MEN_N.txt')
    # files_dict["whr_WHR_WOMEN_N"] = os.path.join(SUMSTATS_DIR_New, 'whr', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_WHR_WOMEN_N.txt')


    files_dict['overweight'] = os.path.join(SUMSTATS_DIR, 'overweight',
                                            'GIANT_OVERWEIGHT_Stage1_Berndt2013_publicrelease_HapMapCeuFreq.txt')  # https://portals.broadinstitute.org/collaboration/giant/index.php/Main_Page
    files_dict['obesity_class1'] = os.path.join(SUMSTATS_DIR, 'obesity_class1',
                                                'GIANT_OBESITY_CLASS1_Stage1_Berndt2013_publicrelease_HapMapCeuFreq.txt')  # https://portals.broadinstitute.org/collaboration/giant/index.php/Main_Page
    files_dict['obesity_class2'] = os.path.join(SUMSTATS_DIR, 'obesity_class2',
                                                'GIANT_OBESITY_CLASS2_Stage1_Berndt2013_publicrelease_HapMapCeuFreq.txt')  # https://portals.broadinstitute.org/collaboration/giant/index.php/Main_Page
    #TODO: Check for hba1c P value
    # files_dict['hba1c'] = os.path.join(SUMSTATS_DIR, 'HbA1C','MAGIC_HbA1C.txt')  # ftp://ftp.sanger.ac.uk/pub/magic/MAGIC_HbA1C.txt.gz
    # files_dict['Non_Diabetic_glucose2'] = os.path.join(SUMSTATS_DIR, 'glucose',
    #                                                    'MAGIC_Manning_et_al_FastingGlucose_MainEffect.txt.gz')  # ftp://ftp.sanger.ac.uk/pub/magic/MAGIC_HbA1C.txt.gz
    # files_dict['Magnetic_glucose'] = os.path.join(SUMSTATS_DIR, 'glucose', 'Summary_statistics_MAGNETIC_Glc.txt.gz') #ftp://ftp.sanger.ac.uk/pub/magic/MAGIC_HbA1C.txt.gz
    files_dict['cigs_per_day'] = os.path.join(SUMSTATS_DIR, 'smoke',
                                              'tag.cpd.tbl')  # Nature Genetics volume 42, pages 441 447 (2010),http://www.med.unc.edu/pgc/files/resultfiles/readme.tag.txt/view
    files_dict['ever_smoked'] = os.path.join(SUMSTATS_DIR, 'smoke',
                                             'tag.evrsmk.tbl')  # Nature Genetics volume 42, pages 441 447 (2010),http://www.med.unc.edu/pgc/files/resultfiles/readme.tag.txt/view
    files_dict['age_smoke'] = os.path.join(SUMSTATS_DIR, 'smoke',
                                           'tag.logonset.tbl')  # Nature Genetics volume 42, pages 441 447 (2010),http://www.med.unc.edu/pgc/files/resultfiles/readme.tag.txt/view
    files_dict['hdl'] = os.path.join(SUMSTATS_DIR, 'HDL',
                                     'jointGwasMc_HDL.txt')  # https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx
    files_dict['ldl'] = os.path.join(SUMSTATS_DIR, 'LDL',
                                     'jointGwasMc_LDL.txt')  ##https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx
    files_dict['triglycerides'] = os.path.join(SUMSTATS_DIR, 'triglycerides',
                                               'jointGwasMc_TG.txt')  ##https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx
    files_dict['cholesterol'] = os.path.join(SUMSTATS_DIR, 'cholesterol',
                                             'jointGwasMc_TC.txt')  ##https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx

    files_dict['diabetes_BMI_Unadjusted'] = os.path.join(SUMSTATS_DIR, 'diabetes',
                                                         'T2D_TranEthnic.BMIunadjusted.txt')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['diabetes_BMI_Adjusted'] = os.path.join(SUMSTATS_DIR, 'diabetes',
                                                       'T2D_TranEthnic.BMIadjusted.txt')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    # files_dict['Coronary_Artery_Disease'] = os.path.join(SUMSTATS_DIR, 'CAD', 'MICAD.EUR.ExA.Consortium.PublicRelease.310517.txt')#This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).

    # files_dict["diabetes_Saxena"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'Saxena-17463246.txt')
    # files_dict["diabetes_Fuchsberger2016"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'DIAGRAMmeta_Fuchsberger2016.txt')
    # files_dict["diabetes_Morris2012.females"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'DIAGRAM.Morris2012.females.txt')
    # files_dict["diabetes_Morris2012.males"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'DIAGRAM.Morris2012.males.txt')
    # files_dict["diabetes_metabochip.only"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'DIAGRAM.website.metabochip.only.txt')
    # files_dict["diabetes_GWAS.metabochip"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'DIAGRAM.website.GWAS.metabochip.txt')
    # files_dict["diabetes_Gaulton_2015"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'DIAGRAM_Gaulton_2015.txt')
    # files_dict["diabetes_v3.2012DEC17"] = os.path.join(SUMSTATS_DIR_New, 'diabetes', 'DIAGRAMv3.2012DEC17.txt')

    files_dict['FastingGlucose'] = os.path.join(SUMSTATS_DIR, 'Fasting',
                                                'MAGIC_FastingGlucose.txt')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['ln_HOMA-B'] = os.path.join(SUMSTATS_DIR, 'Fasting',
                                           'MAGIC_ln_HOMA-B.txt')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['ln_FastingInsulin'] = os.path.join(SUMSTATS_DIR, 'Fasting',
                                                   'MAGIC_ln_FastingInsulin.txt')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['ln_HOMA-IR'] = os.path.join(SUMSTATS_DIR, 'Fasting',
                                            'MAGIC_ln_HOMA-IR.txt')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).

    files_dict['Leptin_BMI'] = os.path.join(SUMSTATS_DIR, 'Leptin', 'Leptin_Adjusted_for_BMI.txt')
    files_dict['Leptin_Unadjusted_BMI'] = os.path.join(SUMSTATS_DIR, 'Leptin', 'Leptin_Not_Adjusted_for_BMI.txt')
    files_dict['Body_fat'] = os.path.join(SUMSTATS_DIR, 'Body_fat',
                                          'body_fat_percentage_GWAS_PLUS_MC_ALL_ancestry_se_Sex_combined_for_locus_zoom_plot.TBL.txt')
    files_dict['Heart_Rate'] = os.path.join(SUMSTATS_DIR, 'Heart_rate', 'META_STAGE1_GWASHR_SUMSTATS.txt')#PMID 23583979
    files_dict['Magic_2hrGlucose'] = os.path.join(SUMSTATS_DIR, '2hr_Glucose', 'MAGIC_2hrGlucose_AdjustedForBMI.txt')
    files_dict['MAGIC_fastingProinsulin'] = os.path.join(SUMSTATS_DIR, 'Pro_Insulin', 'MAGIC_ln_fastingProinsulin.txt')
    files_dict['MAGIC_Scott_2hGlu'] = os.path.join(SUMSTATS_DIR, 'Insulin/Magic_Metabochip',
                                                   'MAGIC_Scott_et_al_2hGlu_Jan2013.txt')
    files_dict['MAGIC_Scott_FG'] = os.path.join(SUMSTATS_DIR, 'Insulin/Magic_Metabochip',
                                                'MAGIC_Scott_et_al_FG_Jan2013.txt')
    files_dict['MAGIC_Scott_FI_adjBMI'] = os.path.join(SUMSTATS_DIR, 'Insulin/Magic_Metabochip',
                                                       'MAGIC_Scott_et_al_FI_adjBMI_Jan2013.txt')
    files_dict['MAGIC_Scott_FI'] = os.path.join(SUMSTATS_DIR, 'Insulin/Magic_Metabochip',
                                                'MAGIC_Scott_et_al_FI_Jan2013.txt')
    files_dict['MAGIC_HbA1C'] = os.path.join(SUMSTATS_DIR, 'HbA1C', 'MAGIC_HbA1C.txt')  # Fasting Insulin

    files_dict['Manning_FG'] = os.path.join(SUMSTATS_DIR, 'Insulin/Manning',
                                            'MAGIC_Manning_et_al_FastingGlucose_MainEffect.txt')  # Fasting Glucose
    files_dict['Manning_BMI_ADJ_FG'] = os.path.join(SUMSTATS_DIR, 'Insulin/Manning',
                                                    'BMI_ADJ_FG_Manning.txt')  # Fasting Glucose
    files_dict['Manning_Fasting_Insulin'] = os.path.join(SUMSTATS_DIR, 'Insulin/Manning',
                                                    'MAGIC_Manning_et_al_lnFastingInsulin_MainEffect.txt')  # Fasting Insulin
    files_dict['Manning_BMI_ADJ_FI'] = os.path.join(SUMSTATS_DIR, 'Insulin/Manning',
                                                    'BMI_ADJ__Manning_Fasting_Insulin.txt')  # Fasting Insulin
    files_dict['HBA1C_ISI'] = os.path.join(SUMSTATS_DIR, 'HBA1C_ISI',
                                           'MAGIC_ISI_Model_1_AgeSexOnly.txt')  # Fasting Insulin
    files_dict['HBA1C_ISI'] = os.path.join(SUMSTATS_DIR, 'HBA1C_ISI',
                                           'MAGIC_ISI_Model_2_AgeSexBMI.txt')  # Fasting Insulin
    files_dict['HBA1C_ISI'] = os.path.join(SUMSTATS_DIR, 'HBA1C_ISI', 'MAGIC_ISI_Model_3_JMA.txt')  # Fasting Insulin
    files_dict['HbA1c_MANTRA'] = os.path.join(SUMSTATS_DIR, 'HbA1C', 'HbA1c_MANTRA.txt')  # Fasting Insulin


    # TODO delete
    #files_dict['A1C_Mantra'] = os.path.join(SUMSTATS_DIR, 'a1c', 'HbA1c_MANTRA.txt')
    #files_dict['Alzheimer_1_2'] = os.path.join(SUMSTATS_DIR, 'Alzheimer', 'IGAP_stage_1_2_combined.txt')
    #files_dict['Asthma '] = os.path.join(SUMSTATS_DIR, 'Asthma', 'gabriel_asthma_meta-analysis_36studies_format_repository_NEJM.txt')
    #files_dict['bmi'] = os.path.join(SUMSTATS_DIR, 'bmi', 'SNP_gwas_mc_merge_nogc.tbl.uniq')
    #files_dict["Body_Fat"] = os.path.join(SUMSTATS_DIR, 'Body_Fat', 'body_fat_percentage_GWAS_PLUS_MC_ALL_ancestry_se_Sex_combined_for_locus_zoom_plot.TBL.txt')
    #files_dict["cardiogramplusc4d"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'cardiogramplusc4d_data.txt')
    #files_dict["MICAD.EUR.ExA.310517"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'MICAD.EUR.ExA.Consortium.PublicRelease.310517.txt')
    #files_dict["Cholesterol"] = os.path.join(SUMSTATS_DIR, 'cholesterol ', 'jointGwasMc_TC.txt')
    # files_dict["diabetes_TranEthnic"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'T2D_TranEthnic.BMIunadjusted.txt')
    # files_dict["diabetes_mega-meta"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'diagram.mega-meta.txt')
    # files_dict["FastingGlucose"] = os.path.join(SUMSTATS_DIR, 'Glucose', 'MAGIC_FastingGlucose.txt')
    # files_dict["2hrGlucose_AdjustedForBMI"] = os.path.join(SUMSTATS_DIR, 'Glucose', 'MAGIC_2hrGlucose_AdjustedForBMI.txt')
    # files_dict["LDL_Joint"] = os.path.join(SUMSTATS_DIR, 'LDL ', 'jointGwasMc_LDL.txt')
    # files_dict["Heart_rate"] = os.path.join(SUMSTATS_DIR, 'Heart_rate', 'META_STAGE1_GWASHR_SUMSTATS.txt')
    # files_dict["HIP_COMBINED_EUR"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_2015_HIP_COMBINED_EUR.txt')
    # files_dict["INSULIN_FastingInsulin"] = os.path.join(SUMSTATS_DIR, 'Insulin', 'MAGIC_ln_FastingInsulin.txt')
    # files_dict["INSULIN_fastingProinsulin"] = os.path.join(SUMSTATS_DIR, 'Insulin', 'MAGIC_ln_fastingProinsulin.txt')
    # files_dict["INSULIN_HOMA-B"] = os.path.join(SUMSTATS_DIR, 'Insulin', 'MAGIC_ln_HOMA-B.txt')
    # files_dict["INSULIN_HOMA-IR"] = os.path.join(SUMSTATS_DIR, 'Insulin', 'MAGIC_ln_HOMA-IR.txt')
    # files_dict["Leptin_adj_BMI"] = os.path.join(SUMSTATS_DIR, 'Leptin', 'Leptin_Adjusted_for_BMI.txt')
    # files_dict["Leptin_not_adj_bmi"] = os.path.join(SUMSTATS_DIR, 'Leptin', 'Leptin_Not_Adjusted_for_BMI.txt')
    # files_dict["Obesity"] = os.path.join(SUMSTATS_DIR, 'Obesity', 'GIANT_OBESITY_CLASS1_Stage1_Berndt2013_publicrelease_HapMapCeuFreq.txt')
    # files_dict["smoke_cpd"] = os.path.join(SUMSTATS_DIR, 'smoke', 'tag.cpd.tbl')
    # files_dict["smoke_evrsmk"] = os.path.join(SUMSTATS_DIR, 'smoke', 'tag.evrsmk.tbl')
    # files_dict["smoke_logonset"] = os.path.join(SUMSTATS_DIR, 'smoke', 'tag.logonset.tbl')
    # files_dict["triglycerides_Joint"] = os.path.join(SUMSTATS_DIR, 'triglycerides', 'jointGwasMc_TG.txt')
    # files_dict["Waist_EUR2"] = os.path.join(SUMSTATS_DIR, 'waist', 'GIANT_2015_WC_COMBINED_EUR2.txt')
    # files_dict["Waist__EUR"] = os.path.join(SUMSTATS_DIR, 'waist', 'GIANT_2015_WC_COMBINED_EUR.txt')
    # files_dict["Waist_Fem_Euro"] = os.path.join(SUMSTATS_DIR, 'waist', 'GIANT_2015_WC_FEMALES_EUR.txt')
    # files_dict["Waist_Males_Euro"] = os.path.join(SUMSTATS_DIR, 'waist', 'GIANT_2015_WC_MALES_EUR.txt')
    # files_dict["Waist_WC_MEN_N"] = os.path.join(SUMSTATS_DIR, 'waist', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_WC_MEN_N.txt')
    #

    # TODO Add to list
    #files_dict['A1C_Metal'] = os.path.join(SUMSTATS_DIR, 'a1c', 'HbA1c_METAL_European.txt')
    #files_dict['ADHD'] = os.path.join(SUMSTATS_DIR, 'ADHD', 'adhd_jul2017')
    #files_dict['Alzheimer_1'] = os.path.join(SUMSTATS_DIR, 'Alzheimer', 'IGAP_stage_1.txt')
    #files_dict["Breast_Cancer"] = os.path.join(SUMSTATS_DIR, 'Breast_Cancer', 'icogs_bcac_public_results_euro (1).txt')
    #files_dict["cad.add.160614"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'cad.add.160614.website.txt')
    #files_dict["cad.rec.090715"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'cad.rec.090715.web.txt')

    #files_dict["CAD_mi.add.030315"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'mi.add.030315.website.txt')
    #files_dict["CARDIoGRAM_Ia_All"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'DataForCARDIoGRAMwebpage_Ia_All_20160105.csv')
    #files_dict["CARDIoGRAMIb_All"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'DataForCARDIoGRAMwebpage_Ib_All_20160105.csv')
    #files_dict["CARDIoGRAMIIa_All"] = os.path.join(SUMSTATS_DIR, 'Cardiogram','DataForCARDIoGRAMwebpage_IIa_All_20160105.csv')
    #files_dict["CARDIoGRAM_IIb_All"] = os.path.join(SUMSTATS_DIR, 'Cardiogram', 'DataForCARDIoGRAMwebpage_IIb_All_20160105.csv')
    #files_dict["Cognitive"] = os.path.join(SUMSTATS_DIR, 'Cognitive', 'GWAS_CP_10k.txt')
    # files_dict["diabetes_Saxena"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'Saxena-17463246.txt')
    # files_dict["diabetes_Fuchsberger2016"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'DIAGRAMmeta_Fuchsberger2016.txt')
    # files_dict["diabetes_Morris2012.females"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'DIAGRAM.Morris2012.females.txt')
    # files_dict["diabetes_Morris2012.males"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'DIAGRAM.Morris2012.males.txt')
    # files_dict["diabetes_metabochip.only"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'DIAGRAM.website.metabochip.only.txt')
    # files_dict["diabetes_GWAS.metabochip"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'DIAGRAM.website.GWAS.metabochip.txt')
    # files_dict["diabetes_Gaulton_2015"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'DIAGRAM_Gaulton_2015.txt')
    # files_dict["diabetes_v3.2012DEC17"] = os.path.join(SUMSTATS_DIR, 'diabetes', 'DIAGRAMv3.2012DEC17.txt')
    # files_dict["HDL"] = os.path.join(SUMSTATS_DIR, 'HDL', 'AGEN_lipids_hapmap_hdl_m2.txt')
    # files_dict["LDL_AGEN"] = os.path.join(SUMSTATS_DIR, 'LDL ', 'AGEN_lipids_hapmap_ldl_m2.txt')
    # files_dict["HIPadjBMI_AllAncestries"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_2015_HIPadjBMI_COMBINED_AllAncestries.txt')
    # files_dict["HIPadjBMI_COMBINED_EUR"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_2015_HIPadjBMI_COMBINED_EUR.txt')
    # files_dict["HIP_COMBINED_AllAncestries"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_2015_HIP_COMBINED_AllAncestries.txt')
    # files_dict["HIP_FEMALES_EUR"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_2015_HIP_FEMALES_EUR.txt')
    # files_dict["HIP_MALES_EUR"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_2015_HIP_MALES_EUR.txt')
    # files_dict["HIP_HapMapCeuFreq_MEN"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_HIP_MEN_N.txt')
    # files_dict["HIP_HapMapCeuFreq_WOMEN"] = os.path.join(SUMSTATS_DIR, 'HIP', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_HIP_WOMEN_N.txt')
    # files_dict["INSULIN_SECRETION_AUCins"] = os.path.join(SUMSTATS_DIR, 'Insulin', 'MAGIC_INSULIN_SECRETION_AUCins_AUCgluc_for_release_HMrel27.txt')
    # files_dict["INSULIN_SECRETION_for_release"] = os.path.join(SUMSTATS_DIR, 'Insulin', 'MAGIC_INSULIN_SECRETION_AUCins_for_release_HMrel27.txt')
    # files_dict["OCD"] = os.path.join(SUMSTATS_DIR, 'OCD', 'ocd_aug2017')
    # files_dict["PTSD"] = os.path.join(SUMSTATS_DIR, 'PTSD', 'SORTED_PTSD_EA9_AA7_LA1_SA2_ALL_study_specific_PCs1.txt')
    # files_dict["Psoriasis"] = os.path.join(SUMSTATS_DIR, 'OCD', 'tsoi_2012_23143594_pso_efo0000676_1_ichip.sumstats.tsv')
    # files_dict["T1D"] = os.path.join(SUMSTATS_DIR, 'T1D', 'bradfield_2011_21980299_t1d_efo0001359_1_gwas.sumstats.tsv')
    # files_dict["Total_Cholesterol_AGEN"] = os.path.join(SUMSTATS_DIR, 'Total_Cholesterol', 'AGEN_lipids_hapmap_tc_m2.txt')
    # files_dict["triglycerides_AGEN"] = os.path.join(SUMSTATS_DIR, 'triglycerides', 'AGEN_lipids_hapmap_tg_m2.txt')

    # files_dict["Waist_WCadjBMI_ALL"] = os.path.join(SUMSTATS_DIR, 'waist', 'GIANT_2015_WCadjBMI_COMBINED_AllAncestries.txt')
    # files_dict["Waist_ALL"] = os.path.join(SUMSTATS_DIR, 'waist', 'GIANT_2015_WC_COMBINED_AllAncestries.txt')
    # files_dict["whr_WHRadjBMI_COMB_All"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_2015_WHRadjBMI_COMBINED_AllAncestries.txt')
    # files_dict["whr_WHRadjBMI_COMB_EUR"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_2015_WHRadjBMI_COMBINED_EUR.txt')
    # files_dict["whr_WHR_COMBINED_All"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_2015_WHR_COMBINED_AllAncestries.txt')
    # files_dict["whr_WHR_COMBINED_EUR"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_2015_WHR_COMBINED_EUR.txt')
    # files_dict["whr_WHR_FEMALES_EUR"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_2015_WHR_FEMALES_EUR.txt')
    # files_dict["whr_WHR_MALES_EUR"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_2015_WHR_MALES_EUR.txt')
    # files_dict["whr_WHR_MEN_N"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_WHR_MEN_N.txt')
    # files_dict["whr_WHR_WOMEN_N"] = os.path.join(SUMSTATS_DIR, 'whr', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_WHR_WOMEN_N.txt')

    return files_dict

def get_traits_dict():
    """Building dictionary with Traits names, paths to traits are being built at get_files_dict()"""
    traits_dict = dict([])
    traits_dict['height'] = 'Height'
    traits_dict['diabetes_BMI_Adjusted']='Diabetes'
    traits_dict['diabetes_BMI_Unadjusted']='Diabetes'
    traits_dict['ADHD'] = 'ADHD'
    traits_dict['alzheimer'] = 'Alzheimer'
    traits_dict['cognitive'] ='Cognitive'
    traits_dict['anorexia'] = 'Anorexia'
    traits_dict['ashtma'] = 'Ashtma'
    traits_dict['baldness'] = 'Baldness'
    traits_dict['depression'] = 'Depression'
    traits_dict['cognitive'] ='Cognitive'
    # traits_dict['crohns'] = 'Crohns'
    # Dont Erase Used for calibration
    traits_dict['cardio'] = 'Cardio'
    traits_dict['bmi'] = 'BMI'
    traits_dict['waist'] = 'Waist'
    traits_dict['hips'] = 'Hips'
    traits_dict['glucose2'] = 'WakeupGlucose'
    traits_dict['glucose_iris'] = 'median_Without_BMI_ALT_Overall'
    traits_dict['whr'] = 'WHR'
    traits_dict['median_glucose'] = 'Median_Glucose'
    traits_dict['hba1c'] = 'HbA1C%'
    traits_dict['hdl'] = 'HDLCholesterol'
    traits_dict['ldl'] = 'LDLCholesterol'
    traits_dict['triglycerides'] = 'Triglycerides'
    traits_dict['creatinine'] = 'Creatinine'
    traits_dict['albumin'] = 'Albumin'
    traits_dict['overweight'] = 'Overweight'
    traits_dict['obesity_class1'] = 'Obesity_class1'
    traits_dict['obesity_class2'] = 'Obesity_class2'
    traits_dict['cholesterol'] = 'Cholesterol,total'
    traits_dict['ever_smoked'] = 'Ever_smoked'
    traits_dict['age_smoke'] = 'Start_smoking_age'
    traits_dict['cigs_per_day'] = 'Cigarretes_per_day'
    traits_dict['lactose'] = 'lactose'
    #
    return traits_dict

def Get_Top_Gen_Dict():
    files_dict = dict([])
    files_dict['height'] = os.path.join(Gen_DIR, 'Final_SNPs_height.csv')
    files_dict['alzheimer'] = os.path.join(Gen_DIR, 'Final_SNPs_alzheimer.csv')  # Jean-Charles Lambert et al.
    files_dict['bmi'] = os.path.join(Gen_DIR, 'Final_SNPs_bmi.csv')  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4382211/
    files_dict['anorexia'] = os.path.join(Gen_DIR, 'Final_SNPs_anorexia.csv')  # A genome-wide association study of anorexia nervosa,https://www.nature.com/articles/mp2013187
    # TODO: check for Asthma pvalue
    # files_dict['ashtma'] = os.path.join(Gen_DIR, 'Ashtma','gabriel_asthma_meta-analysis_36studies_format_repository_NEJM.txt')  # https://www.cnrgh.fr/gabriel/study_description.html
    files_dict['t2d_mega_meta'] = os.path.join(Gen_DIR, 'Final_SNPs_t2d_mega_meta.csv')  # FKA iris Trans-ethnic T2D GWAS meta-analysis, http://diagram-consortium.org/downloads.html
    files_dict['cardio'] = os.path.join(Gen_DIR, 'Final_SNPs_cardio.csv')  # CARDIoGRAMplusC4D Metabochip is a two stage meta-analysis of Metabochip and GWAS studies of European and South Asian descent involving 63,746 cases and 130,681 controls. The CARDIoGRAM GWAS data was used as Stage 1 - data as published in: CARDIoGRAMplusC4D Consortium, Deloukas P, Kanoni S, Willenborg C, Farrall M, Assimes TL, Thompson JR, et al. Large-scale association analysis identifies new risk loci for coronary artery disease. Nat Genet 2013 45:25-33
    files_dict['hips'] = os.path.join(Gen_DIR, 'Final_SNPs_hips.csv')  # https://www.nature.com/articles/nature14132,https://portals.broadinstitute.org/collaboration/giant/index.php/GIANT_consortium_data_files
    files_dict['waist'] = os.path.join(Gen_DIR, 'Final_SNPs_waist.csv')  # https://www.nature.com/articles/nature14132,https://portals.broadinstitute.org/collaboration/giant/index.php/GIANT_consortium_data_files
    #TODO:Clean the data below
    # files_dict["whr_WHR_COMBINED_EUR2"] = os.path.join(Gen_DIR, 'whr', 'GIANT_2015_WHR_COMBINED_EUR2.txt')
    # files_dict["whr_WHRadjBMI_COMB_All"] = os.path.join(Gen_DIR, 'whr', 'GIANT_2015_WHRadjBMI_COMBINED_AllAncestries.txt')
    # files_dict["whr_WHRadjBMI_COMB_EUR"] = os.path.join(Gen_DIR, 'whr', 'GIANT_2015_WHRadjBMI_COMBINED_EUR.txt')
    # files_dict["whr_WHR_COMBINED_All"] = os.path.join(Gen_DIR, 'whr', 'GIANT_2015_WHR_COMBINED_AllAncestries.txt')
    # files_dict["whr_WHR_COMBINED_EUR"] = os.path.join(Gen_DIR, 'whr', 'GIANT_2015_WHR_COMBINED_EUR.txt')
    # files_dict["whr_WHR_FEMALES_EUR"] = os.path.join(Gen_DIR, 'whr', 'GIANT_2015_WHR_FEMALES_EUR.txt')
    # files_dict["whr_WHR_MALES_EUR"] = os.path.join(Gen_DIR, 'whr', 'GIANT_2015_WHR_MALES_EUR.txt')
    # files_dict["whr_WHR_MEN_N"] = os.path.join(Gen_DIR, 'whr', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_WHR_MEN_N.txt')
    # files_dict["whr_WHR_WOMEN_N"] = os.path.join(Gen_DIR, 'whr', 'GIANT_Randall2013PlosGenet_stage1_publicrelease_HapMapCeuFreq_WHR_WOMEN_N.txt')
    files_dict['overweight'] = os.path.join(Gen_DIR, 'Final_SNPs_overweight.csv')  # https://portals.broadinstitute.org/collaboration/giant/index.php/Main_Page
    files_dict['obesity_class1'] = os.path.join(Gen_DIR, 'Final_SNPs_obesity_class1.csv')  # https://portals.broadinstitute.org/collaboration/giant/index.php/Main_Page
    files_dict['obesity_class2'] = os.path.join(Gen_DIR, 'Final_SNPs_obesity_class2.csv')  # https://portals.broadinstitute.org/collaboration/giant/index.php/Main_Page
    #TODO: Check for hba1c P value
    # files_dict['hba1c'] = os.path.join(SUMSTATS_DIR, 'HbA1C','MAGIC_HbA1C.txt')  # ftp://ftp.sanger.ac.uk/pub/magic/MAGIC_HbA1C.txt.gz
    # files_dict['Non_Diabetic_glucose2'] = os.path.join(SUMSTATS_DIR, 'glucose','MAGIC_Manning_et_al_FastingGlucose_MainEffect.txt.gz')  # ftp://ftp.sanger.ac.uk/pub/magic/MAGIC_HbA1C.txt.gz
    # files_dict['Magnetic_glucose'] = os.path.join(SUMSTATS_DIR, 'glucose', 'Summary_statistics_MAGNETIC_Glc.txt.gz') #ftp://ftp.sanger.ac.uk/pub/magic/MAGIC_HbA1C.txt.gz
    files_dict['cigs_per_day'] = os.path.join(Gen_DIR, 'Final_SNPs_cigs_per_day.csv')  # Nature Genetics volume 42, pages 441 447 (2010),http://www.med.unc.edu/pgc/files/resultfiles/readme.tag.txt/view
    files_dict['ever_smoked'] = os.path.join(Gen_DIR, 'Final_SNPs_ever_smoked.csv')  # Nature Genetics volume 42, pages 441 447 (2010),http://www.med.unc.edu/pgc/files/resultfiles/readme.tag.txt/view
    files_dict['age_smoke'] = os.path.join(Gen_DIR, 'Final_SNPs_age_smoke.csv')  # Nature Genetics volume 42, pages 441 447 (2010),http://www.med.unc.edu/pgc/files/resultfiles/readme.tag.txt/view
    files_dict['hdl'] = os.path.join(Gen_DIR, 'Final_SNPs_hdl.csv')  # https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx
    files_dict['ldl'] = os.path.join(Gen_DIR, 'Final_SNPs_ldl.csv')  ##https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx
    files_dict['triglycerides'] = os.path.join(Gen_DIR, 'Final_SNPs_triglycerides.csv')  ##https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx
    files_dict['cholesterol'] = os.path.join(Gen_DIR, 'Final_SNPs_cholesterol.csv')  ##https://www.nature.com/articles/ng.2797,https://grasp.nhlbi.nih.gov/FullResults.aspx

    files_dict['diabetes_BMI_Unadjusted'] = os.path.join(Gen_DIR, 'Final_SNPs_diabetes_BMI_Unadjusted.csv')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['diabetes_BMI_Adjusted'] = os.path.join(Gen_DIR, 'Final_SNPs_diabetes_BMI_Adjusted.csv')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['FastingGlucose'] = os.path.join(Gen_DIR, 'Final_SNPs_FastingGlucose.csv')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['ln_HOMA-B'] = os.path.join(Gen_DIR, 'Final_SNPs_ln_HOMA-B.csv')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['ln_FastingInsulin'] = os.path.join(Gen_DIR, 'Final_SNPs_ln_FastingInsulin.csv')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).
    files_dict['ln_HOMA-IR'] = os.path.join(Gen_DIR, 'Final_SNPs_ln_HOMA-IR.csv')  # This file contains association summary statistics for the DIAGRAMv3 GWAS meta-analysis, as published in Morris et al. (2012).

    files_dict['Leptin_BMI'] = os.path.join(Gen_DIR, 'Final_SNPs_Leptin_BMI.csv')
    files_dict['Leptin_Unadjusted_BMI'] = os.path.join(Gen_DIR, 'Final_SNPs_Leptin_Unadjusted_BMI.csv')
    # files_dict['Body_fat'] = os.path.join(Gen_DIR, 'Final_SNPs_Body_fat.csv')
    files_dict['Heart_Rate'] = os.path.join(Gen_DIR, 'Final_SNPs_Heart_Rate.csv')
    files_dict['Magic_2hrGlucose'] = os.path.join(Gen_DIR, 'Final_SNPs_Magic_2hrGlucose.csv')
    files_dict['MAGIC_fastingProinsulin'] = os.path.join(Gen_DIR, 'Final_SNPs_MAGIC_fastingProinsulin.csv')
    files_dict['MAGIC_Scott_2hGlu'] = os.path.join(Gen_DIR, 'Final_SNPs_MAGIC_Scott_2hGlu.csv')
    files_dict['MAGIC_Scott_FG'] = os.path.join(Gen_DIR, 'Final_SNPs_MAGIC_Scott_FG.csv')
    files_dict['MAGIC_Scott_FI_adjBMI'] = os.path.join(Gen_DIR, 'Final_SNPs_MAGIC_Scott_FI_adjBMI.csv')
    files_dict['MAGIC_Scott_FI'] = os.path.join(Gen_DIR, 'Final_SNPs_MAGIC_Scott_FI.csv')
    files_dict['MAGIC_HbA1C'] = os.path.join(Gen_DIR, 'Final_SNPs_MAGIC_HbA1C.csv')  # Fasting Insulin

    files_dict['Manning_FG'] = os.path.join(Gen_DIR, 'Final_SNPs_Manning_FG.csv')  # Fasting Glucose
    files_dict['Manning_BMI_ADJ_FG'] = os.path.join(Gen_DIR, 'Final_SNPs_Manning_BMI_ADJ_FG.csv')  # Fasting Glucose
    files_dict['Manning_Fasting_Insulin'] = os.path.join(Gen_DIR, 'Final_SNPs_Manning_Fasting_Insulin.csv')  # Fasting Insulin
    files_dict['Manning_BMI_ADJ_FI'] = os.path.join(Gen_DIR, 'Final_SNPs_Manning_BMI_ADJ_FI.csv')  # Fasting Insulin
    # files_dict['HBA1C_ISI'] = os.path.join(Gen_DIR, 'Final_SNPs_HBA1C_ISI',
    #                                        'MAGIC_ISI_Model_1_AgeSexOnly.txt')  # Fasting Insulin
    files_dict['HBA1C_ISI'] = os.path.join(Gen_DIR, 'Final_SNPs_HBA1C_ISI.csv')  # Fasting Insulin
    # files_dict['HBA1C_ISI'] = os.path.join(SUMSTATS_DIR, 'HBA1C_ISI', 'MAGIC_ISI_Model_3_JMA.txt')  # Fasting Insulin
    files_dict['HbA1c_MANTRA'] = os.path.join(Gen_DIR, 'Final_SNPs_HbA1c_MANTRA.csv')  # Fasting Insulin
    return files_dict

def get_predictions(bfile_path):
    """Function that gets bfile of persons and computes their PRS"""
    bed = read_bfile_forsumstats(bfile_path) #bfile_path for the bed file
    df_bim = pd.read_csv(bfile_path+'.bim', delim_whitespace=True, header=None, names=['chr', 'rs', 'cm', 'bp', 'a1', 'a2']) #List of al SNPS
    df_bed = pd.DataFrame(bed.sid, columns=['rs']) #SNP names
    df_bed = df_bed.merge(df_bim, how='left', on='rs')
    df_bed.rename(index=str, columns={"a1": "a1_bim", "a2": "a2_bim"})
    files_dict = get_files_dict()    
    df_predictions = pd.DataFrame(index=bed.iid[:,1].astype(np.int))

    for f_i,(trait, sumstats_file) in enumerate(files_dict.iteritems()):
        
        ###if (trait not in ['bmi', 'height', 'hdl', 'creatinine', 'glucose2']): continue
        ###if (trait not in ['glucose_iris']): continue
        
        #read summary statistics file
        print ('reading summary statistics and performing prediction for %s...'%(trait))
        if (trait == 'creatinine'): df_sumstats = pd.read_csv(sumstats_file, sep=',')
        else: df_sumstats = pd.read_csv(sumstats_file, delim_whitespace=True)        
        found_snp_col = False
        #Checking for all posible SNP name versions
        for snp_name_col in ['SNP_ID','MarkerName', 'SNP', 'rsID', 'snp', 'rsid', 'sid', 'Snp','rs','Markername',"ID"]:
            if (snp_name_col not in df_sumstats.columns): continue
            found_snp_col = True
            break
        assert found_snp_col, 'No SNP column found'
        df_sumstats.drop_duplicates(subset=snp_name_col, inplace=True)
        df_merge = df_bed.merge(df_sumstats, left_on='rs', right_on=snp_name_col)
        df_merge_snps_set = set(df_merge['rs'])
        is_snp_found = [(s in df_merge_snps_set) for s in bed.sid]
        
        #find allele columns
        try:
            df_merge['A1'] = df_merge['Allele1'].str.upper()
            df_merge['A2'] = df_merge['Allele2'].str.upper()
        except: pass

        try:
            df_merge['A1'] = df_merge['Allele_1'].str.upper()
            df_merge['A2'] = df_merge['Allele_2'].str.upper()
        except: pass

        try:  # ~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['allele1'].str.upper()
            df_merge['A2'] = df_merge['allele2'].str.upper()
        except: pass

        try:
            df_merge['A1'] = df_merge['A1'].str.upper()
            df_merge['A2'] = df_merge['A2'].str.upper()
        except: pass
        try:
            df_merge['A1'] = df_merge['NEA'].str.upper() #Switched EA and NEA
            df_merge['A2'] = df_merge['EA'].str.upper()
        except: pass        
        try:
            df_merge['A1'] = df_merge['other_allele'].str.upper()
            df_merge['A2'] = df_merge['effect_allele'].str.upper()
        except: pass

        try:
            df_merge['A1'] = df_merge['Other_allele'].str.upper()
            df_merge['A2'] = df_merge['Effect_allele'].str.upper()
        except: pass

        try:
            df_merge['A1'] = df_merge['OTHER_ALLELE'].str.upper()
            df_merge['A2'] = df_merge['RISK_ALLELE'].str.upper()
        except: pass            

        try: #~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['other_allele'].str.upper()
            df_merge['A2'] = df_merge['reference_allele'].str.upper()
        except: pass

        try:  # ~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['Non_Effect_allele'].str.upper()
            df_merge['A2'] = df_merge['Effect_allele'].str.upper()
        except: pass

        #flip alleles quickly
        a1 = df_merge['a1_bim'].values.copy()
        is_A = (a1=='A')
        is_T = (a1=='T')
        is_C = (a1=='C')
        is_G = (a1=='G')
        a1[is_A] = 'T'
        a1[is_T] = 'A'
        a1[is_C] = 'G'
        a1[is_G] = 'C'
        df_merge['flip_a1'] = a1
        
        a2 = df_merge['a2_bim'].values.copy()
        is_A = (a2=='A')
        is_T = (a2=='T')
        is_C = (a2=='C')
        is_G = (a2=='G')
        a2[is_A] = 'T'
        a2[is_T] = 'A'
        a2[is_C] = 'G'
        a2[is_G] = 'C'
        df_merge['flip_a2'] = a2
        
        #do some standardization        
        # try:
        #     is_same    =    ((df_merge['a1'] == df_merge['Allele1']) & (df_merge['a2'] == df_merge['Allele2'])).values
        #     is_reverse =    ((df_merge['a2'] == df_merge['Allele1']) & (df_merge['a1'] == df_merge['Allele2'])).values
        #     is_flipped = ((df_merge['flip_a1'] == df_merge['Allele1']) & (df_merge['flip_a2'] == df_merge['Allele2'])).values
        #     is_reverse_flipped = ((df_merge['flip_a2'] == df_merge['Allele1']) & (df_merge['flip_a1'] == df_merge['Allele2'])).values
        # except:
        is_same    =    ((df_merge['a1_bim'] == df_merge['A1']) & (df_merge['a2_bim'] == df_merge['A2'])).values
        is_reverse =    ((df_merge['a2_bim'] == df_merge['A1']) & (df_merge['a1_bim'] == df_merge['A2'])).values
        is_flipped = ((df_merge['flip_a1'] == df_merge['A1']) & (df_merge['flip_a2'] == df_merge['A2'])).values
        is_reverse_flipped = ((df_merge['flip_a2'] == df_merge['A1']) & (df_merge['flip_a1'] == df_merge['A2'])).values

  
        #decide which SNPs to keep
        keep_snps = ((is_same) | (is_reverse))

        #find the column of the effect sizes
        found_effects_col = False        
        for effects_col in ['b', 'Beta', 'beta', 'effect', 'OR', 'MainEffects',"log_odds","OR_fix","log_odds_(stage2)"
            ,"Effect","log10bf"]: #"log_odds" was added by Yochai for the Cardio Estimation
            if (effects_col not in df_merge.columns): continue
            found_effects_col = True
            if ((effects_col == 'OR') or (effects_col == 'OR_fix')):
                df_merge['Beta'] = np.log10(df_merge[effects_col].values)
                effects_col = 'Beta'
            effects = df_merge[effects_col].values
        assert found_effects_col, 'couldn\'t find a column of effects'
        
        #flip effects if needed
        effects[is_reverse] *= (-1)

        #compute prediction for each p-values cutoff
        best_corr = -np.inf

        df_predictions.loc[ID,'predict_' + trait] = (bed.val[df_predictions.index, is_snp_found]).dot(effects)  # Performing the dot product


    return df_predictions

def Personal_PRS(bfile_path,ID,full_predictions=None,res=0.025): #Calculate a single person from PNP statistics (Quantile)
    """
    full_predictions is a dataframe with the whole PNP cohort score for chosen phenotype
    bfile_path is the path to the PNP SNPs data
    ID is the ID of a person that we would like to get his statistics
    """
    df_predictions = pd.read_pickle(PKL_PATH)
    df_quantiles = df_predictions.quantile(np.arange(res, 1, res))
    df_quantiles.to_pickle(Quant_PATH)

    bed = read_bfile_forsumstats(bfile_path)

    df_bim = pd.read_csv(bfile_path + '.bim', delim_whitespace=True, header=None,
                         names=['chr', 'rs', 'cm', 'bp', 'a1', 'a2'])  # List of al SNPS

    df_bed = pd.DataFrame(bed.sid, columns=['rs'])  # SNP names
    df_bed = df_bed.merge(df_bim, how='left', on='rs')
    files_dict = get_files_dict()
    df_predictions = pd.DataFrame(index=bed.iid[:, 1].astype(np.int))
    personal_predictions = pd.DataFrame(index=[ID])
    personal_quantiles = pd.DataFrame(index=[ID])
    for f_i, (trait, sumstats_file) in enumerate(files_dict.iteritems()):

        # read summary statistics file
        print 'reading summary statistics and performing prediction for %s...' % (trait)
        if (trait == 'creatinine'):
            df_sumstats = pd.read_csv(sumstats_file, sep=',')
        else:
            df_sumstats = pd.read_csv(sumstats_file, delim_whitespace=True)
        found_snp_col = False
        # Checking for all posible SNP name versions
        for snp_name_col in ['SNP_ID','MarkerName', 'SNP', 'rsID', 'snp', 'rsid', 'sid', 'Snp','rs','Markername',"ID"]:
            if (snp_name_col not in df_sumstats.columns): continue
            found_snp_col = True
            break
        assert found_snp_col, 'No SNP column found'
        df_sumstats.drop_duplicates(subset=snp_name_col, inplace=True)
        df_merge = df_bed.merge(df_sumstats, left_on='rs', right_on=snp_name_col)
        df_merge_snps_set = set(df_merge['rs'])
        is_snp_found = [(s in df_merge_snps_set) for s in bed.sid]

        # find allele columns
        try:
            df_merge['Allele1'] = df_merge['Allele1'].str.upper()
            df_merge['Allele2'] = df_merge['Allele2'].str.upper()
        except:
            pass
        try:
            df_merge['Allele1'] = df_merge['Allele_1'].str.upper()
            df_merge['Allele2'] = df_merge['Allele_2'].str.upper()
        except:
            pass

        try:
            df_merge['A1'] = df_merge['A1'].str.upper()
            df_merge['A2'] = df_merge['A2'].str.upper()
        except:
            pass

        try:
            df_merge['A1'] = df_merge['NEA'].str.upper()  # Switched EA and NEA
            df_merge['A2'] = df_merge['EA'].str.upper()
        except:
            pass
        try:
            df_merge['A1'] = df_merge['other_allele'].str.upper()
            df_merge['A2'] = df_merge['effect_allele'].str.upper()
        except:
            pass

        try:
            df_merge['A1'] = df_merge['Other_allele'].str.upper()
            df_merge['A2'] = df_merge['Effect_allele'].str.upper()
        except:
            pass

        try:
            df_merge['A1'] = df_merge['OTHER_ALLELE'].str.upper()
            df_merge['A2'] = df_merge['RISK_ALLELE'].str.upper()
        except:
            pass

        try:  # ~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['other_allele'].str.upper()
            df_merge['A2'] = df_merge['reference_allele'].str.upper()
        except:
            pass

        try:  # ~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['Non_Effect_allele'].str.upper()
            df_merge['A2'] = df_merge['Effect_allele'].str.upper()
        except:
            pass

        # flip alleles quickly
        a1 = df_merge['a1'].values.copy() #consider converting a1, which is from the bim file, to a1_bim in order not
                                                # to be confused witrh a1 from PRS file
        is_A = (a1 == 'A')
        is_T = (a1 == 'T')
        is_C = (a1 == 'C')
        is_G = (a1 == 'G')
        a1[is_A] = 'T'
        a1[is_T] = 'A'
        a1[is_C] = 'G'
        a1[is_G] = 'C'
        df_merge['flip_a1'] = a1

        a2 = df_merge['a2'].values.copy()
        a2 = df_merge['A2'].values.copy()
        is_A = (a2 == 'A')
        is_T = (a2 == 'T')
        is_C = (a2 == 'C')
        is_G = (a2 == 'G')
        a2[is_A] = 'T'
        a2[is_T] = 'A'
        a2[is_C] = 'G'
        a2[is_G] = 'C'
        df_merge['flip_a2'] = a2

        # do some standardization
        try:
            is_same = ((df_merge['A1'] == df_merge['Allele1']) & (df_merge['A2'] == df_merge['Allele2'])).values
            is_reverse = ((df_merge['A2'] == df_merge['Allele1']) & (df_merge['A1'] == df_merge['Allele2'])).values
            is_flipped = (
            (df_merge['flip_a1'] == df_merge['Allele1']) & (df_merge['flip_a2'] == df_merge['Allele2'])).values
            is_reverse_flipped = (
            (df_merge['flip_a2'] == df_merge['Allele1']) & (df_merge['flip_a1'] == df_merge['Allele2'])).values
        except:
            is_same = ((df_merge['a1'] == df_merge['A1']) & (df_merge['a2'] == df_merge['A2'])).values
            is_reverse = ((df_merge['a2'] == df_merge['A1']) & (df_merge['a1'] == df_merge['A2'])).values
            is_flipped = ((df_merge['flip_a1'] == df_merge['A1']) & (df_merge['flip_a2'] == df_merge['A2'])).values
            is_reverse_flipped = (
            (df_merge['flip_a2'] == df_merge['A1']) & (df_merge['flip_a1'] == df_merge['A2'])).values

        # decide which SNPs to keep
        keep_snps = ((is_same) | (is_reverse))

        # find the column of the effect sizes
        found_effects_col = False
        for effects_col in ['b', 'Beta', 'beta', 'effect', 'OR', 'MainEffects', "log_odds", "OR_fix",
                            "log_odds_(stage2)", "BETA", "Effect", "BMIadjMainEffects", "log10bf"]:  # "log_odds" was added by Yochai for the Cardio Estimation
            if (effects_col not in df_merge.columns): continue
            found_effects_col = True
            effects = df_merge[effects_col].values
        assert found_effects_col, 'couldn\'t find a column of effects'

        # flip effects if needed
        effects[is_reverse] *= (-1)

        # compute prediction for each p-values cutoff
        best_corr = -np.inf
        personal_predictions.loc[ID,'predict_' + trait] = (bed.val[df_predictions.index == ID, is_snp_found]).dot(effects)  # Performing the dot product
        personal_quantiles.loc[ID, 'predict_' + trait] = bisect(df_quantiles.loc[:,'predict_' + trait].values,
                                                            personal_predictions.loc[ID,'predict_' + trait])
    return personal_quantiles

def compute_prs(bfile_path=None, verbose=False,res=0.025):
    
    if (bfile_path is None): df_predictions = pd.read_pickle(PKL_PATH)
    else: 
        #compute predictions for a grid of p-values 
        verbose = True       
        df_predictions = get_predictions(bfile_path)
        df_quantiles = df_predictions.quantile([np.arange(res, 1, res)])
        df_predictions.to_pickle(PKL_PATH)
        df_quantiles.to_pickle(Quant_PATH)
    return df_predictions

def Trait_top_SNPs(PRS_file,trait):
    """Adding top 1000 P values of PRS_file of trait to existing dictionary"""
    found_P_col=False
    snp_name_col=False

    sumstats_file=PRS_file
    # read summary statistics file
    # print 'reading summary statistics and performing prediction for',trait,' at CHR#', str(CHR_Num)
    if (trait == 'creatinine'):
        df_sumstats = pd.read_csv(sumstats_file, sep=',')
    else:
        df_sumstats = pd.read_csv(sumstats_file, delim_whitespace=True)
    found_snp_col = False
    # Checking for all posible SNP name versions

    for P_Name in ['P', 'p', 'P_value', 'Pvalue', 'P_VALUE','P-value',"MainP",'pvalue',
                   "Pvalue_Stage2","P-value","p_sanger","P.value"]:
        if (P_Name not in df_sumstats.columns): continue
        found_P_col = True
        break
    assert found_P_col, 'No P column found'

    for snp_name_col in ['rsID', 'rsid', 'rs', 'sid', 'Markername', 'MarkerName', 'SNP', 'Snp', 'snp',
                         'SNP_ID','SNPID']:
        if (snp_name_col not in df_sumstats.columns): continue
        found_snp_col = True
        break

    df_sumstats=df_sumstats.loc[:,[snp_name_col,P_Name]]
    df_sumstats.set_index(snp_name_col,inplace=True,drop=True)
    df_sumstats.sort_values(by=P_Name,axis=0,inplace=True)
    df1000=df_sumstats.iloc[0:1000]
    df1000.columns=['P']
    return df1000

def All_Traits_Top_SNPs(final_folder,dict_name,n_snps=1000):
    found_P_col = False
    snp_name_col = False
    trait_dict = {}
    files_dict = get_files_dict()
    for f_i, (trait, sumstats_file) in enumerate(files_dict.iteritems()):
        # read summary statistics file
        #     print 'reading summary statistics and performing prediction for',trait,' at CHR#', str(CHR_Num)
        if (trait == 'creatinine'):
            df_sumstats = pd.read_csv(sumstats_file, sep=',')
        else:
            df_sumstats = pd.read_csv(sumstats_file, delim_whitespace=True)
        found_snp_col = False
        # Checking for all posible SNP name versions

        for P_Name in ['P', 'p', 'P_value', 'Pvalue', 'P_VALUE', 'P-value', "MainP", 'pvalue',
                       "Pvalue_Stage2", "P-value", "p_sanger", "P.value"]:
            if (P_Name not in df_sumstats.columns): continue
            found_P_col = True
            break
        assert found_P_col, 'No P column found'

        for snp_name_col in ['rsID', 'rsid', 'rs', 'sid', 'Markername', 'MarkerName', 'SNP', 'Snp', 'snp',
                             'SNP_ID', 'SNPID']:
            if (snp_name_col not in df_sumstats.columns): continue
            found_snp_col = True
            break
        assert found_snp_col, 'No SNP column found'

        print "SNP COL NAME for trait:", trait, ' is:', snp_name_col

        df_sumstats = df_sumstats.loc[:, [snp_name_col, P_Name]]
        df_sumstats.set_index(snp_name_col, inplace=True, drop=True)
        df_sumstats.sort_values(by=P_Name, axis=0, inplace=True)
        trait_dict[trait] = df_sumstats.iloc[0:n_snps]
        trait_dict[trait].columns = ["P"]
        trait_dict[trait].index.name = ["SNP"]

    with open(final_folder + dict_name, 'wb') as fp:
        pickle.dump(trait_dict, fp)

def extract_relevant_SNPS(top_P_dict,bfile_path, Results_Folder, Job_Name, CHR_Num):
    bed = read_bfile_forsumstats(bfile_path)  # bfile_path for the bed file
    df_bim = pd.read_csv(bfile_path + '.bim', delim_whitespace=True, header=None,
                         names=['chr', 'rs', 'cm', 'bp', 'a1', 'a2'])  # List of al SNPS
    df_fam = pd.read_csv(bfile_path + '.fam', delim_whitespace=True, header=None)
    df_bed = pd.DataFrame(bed.sid, columns=['rs'])  # SNP names
    df_bed = df_bed.merge(df_bim, how='left', on='rs')
    df_bed = df_bed.rename(index=str, columns={"a1": "a1_bim", "a2": "a2_bim"})
    df_merge = {}
    is_snp_found = {}
    df_ID_SNPs_for_trait = {}
    for trait in top_P_dict.iterkeys():
        df_merge[trait] = df_bed.merge(top_P_dict[trait].reset_index(), left_on='rs', right_on='SNP')
        df_merge[trait] = df_merge[trait].drop_duplicates(subset="rs")
        df_merge[trait] = df_merge[trait].set_index('rs', drop=True)
        print df_merge[trait].head()
        df_merge_snps_set = set(df_merge[trait].index.values)
        is_snp_found[trait] = [(s in df_merge_snps_set) for s in bed.sid]
        df_ID_SNPs_for_trait[trait] = pd.DataFrame(data=bed.val[:, is_snp_found[trait]],
                                                   index=df_fam.iloc[:, 0].values,
                                                   columns=df_merge[trait].index.values)
        df_ID_SNPs_for_trait[trait].index.name = "eid"
        df_ID_SNPs_for_trait[trait]=df_ID_SNPs_for_trait[trait].reset_index()
        df_ID_SNPs_for_trait[trait].to_csv(path_or_buf=Results_Folder + trait +"_"+CHR_Num+"_.csv", index=False)

def get_UKBB_predictions(bfile_path, Results_Folder, Job_Name, CHR_Num):
    """Function that gets bfile of persons and computes their PRS"""
    print "Started CHR#", CHR_Num
    bed = read_bfile_forsumstats(bfile_path)  # bfile_path for the bed file
    df_bim = pd.read_csv(bfile_path + '.bim', delim_whitespace=True, header=None,
                         names=['chr', 'rs', 'cm', 'bp', 'a1', 'a2'])  # List of al SNPS
    df_bed = pd.DataFrame(bed.sid, columns=['rs'])  # SNP names
    df_bed = df_bed.merge(df_bim, how='left', on='rs')
    df_bed=df_bed.rename(index=str, columns={"a1": "a1_bim", "a2": "a2_bim"})
    files_dict = get_files_dict()
    df_predictions = pd.DataFrame(index=bed.iid[:, 1].astype(np.int))
    df_predictions.index.name = "eid"
    for f_i, (trait, sumstats_file) in enumerate(files_dict.iteritems()):

        ###if (trait not in ['bmi', 'height', 'hdl', 'creatinine', 'glucose2']): continue
        ###if (trait not in ['glucose_iris']): continue

        # read summary statistics file
        print 'reading summary statistics and performing prediction for',trait,' at CHR#', str(CHR_Num)
        if (trait == 'creatinine'):
            df_sumstats = pd.read_csv(sumstats_file, sep=',')
        else:
            df_sumstats = pd.read_csv(sumstats_file, delim_whitespace=True)

        found_snp_col = False
        # Checking for all posible SNP name versions
        for snp_name_col in ['rsID', 'rsid', 'rs', 'sid', 'Markername', 'MarkerName', 'SNP', 'Snp', 'snp',
                             'SNP_ID','SNPID']:
            if (snp_name_col not in df_sumstats.columns): continue
            found_snp_col = True
            break
        assert found_snp_col, 'No SNP column found'
        print "SNP COL NAME for trait:", trait,' is:',snp_name_col

        df_sumstats.drop_duplicates(subset=snp_name_col, inplace=True)
        df_merge = df_bed.merge(df_sumstats, left_on='rs', right_on=snp_name_col)
        print "df_merge.shape[0] according to RSID is: ", df_merge.shape[0],"(i.e. number of recognised SNPS of trarit", \
            trait, " of CHR: ", str(CHR_Num), "of Jobname: ", Job_Name, " )"

        if df_merge.shape[0] == 0:
            print "No RS numbers, merging according to CHR:BP using HG37"
            try:
                df_merge = df_bed.merge(df_sumstats, left_on=['chr', "bp"], right_on=["CHR", "BP"])
            except:
                pass

            try:
                df_merge = df_bed.merge(df_sumstats, left_on=['CHR', "BP"], right_on=["CHR", "BP"])
            except:
                pass

            try:
                df_merge = df_bed.merge(df_sumstats, left_on=['CHR', "POS"], right_on=["CHR", "BP"])
            except:
                pass


        if df_merge.shape[0]==0:
            print "No matching SNPS Found for: ",bfile_path, "for trait:", trait

        df_merge_snps_set = set(df_merge['rs'])
        is_snp_found = [(s in df_merge_snps_set) for s in bed.sid]

        # find allele columns
        try:
            df_merge['A1'] = df_merge['Allele1'].str.upper()
            df_merge['A2'] = df_merge['Allele2'].str.upper()
        except:
            pass

        try:
            df_merge['A1'] = df_merge['Allele_1'].str.upper()
            df_merge['A2'] = df_merge['Allele_2'].str.upper()
        except:
            pass

        try:  # ~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['allele1'].str.upper()
            df_merge['A2'] = df_merge['allele2'].str.upper()
        except: pass

        try:
            df_merge['A1'] = df_merge['A1'].str.upper()
            df_merge['A2'] = df_merge['A2'].str.upper()
        except:
            pass
        try:
            df_merge['A1'] = df_merge['NEA'].str.upper()  # Switched EA and NEA
            df_merge['A2'] = df_merge['EA'].str.upper()
        except:
            pass
        try:
            df_merge['A1'] = df_merge['other_allele'].str.upper()
            df_merge['A2'] = df_merge['effect_allele'].str.upper()
        except:
            pass

        try:
            df_merge['A1'] = df_merge['Other_allele'].str.upper()
            df_merge['A2'] = df_merge['Effect_allele'].str.upper()
        except:
            pass

        try:
            df_merge['A1'] = df_merge['OTHER_ALLELE'].str.upper()
            df_merge['A2'] = df_merge['RISK_ALLELE'].str.upper()
        except:
            pass

        try:  # ~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['other_allele'].str.upper()
            df_merge['A2'] = df_merge['reference_allele'].str.upper()
        except:
            pass

        try:  # ~~~Yochai~~~ Addition for the Cardio file ()
            df_merge['A1'] = df_merge['Non_Effect_allele'].str.upper()
            df_merge['A2'] = df_merge['Effect_allele'].str.upper()
        except:
            pass

        try:  # ~~~Yochai~~~ Addition for the Diabetes file ()
            df_merge['A1'] = df_merge['OTHER_ALLELE'].str.upper()
            df_merge['A2'] = df_merge['EFFECT_ALLELE'].str.upper()
        except:
            pass

        try:  # ~~~Yochai~~~ Addition for the Diabetes file ()
            df_merge['A1'] = df_merge['Other_all'].str.upper()
            df_merge['A2'] = df_merge['Effect_all'].str.upper()
        except:
            pass


        # flip alleles quickly
        a1 = df_merge['a1_bim'].values.copy()
        is_A = (a1 == 'A')
        is_T = (a1 == 'T')
        is_C = (a1 == 'C')
        is_G = (a1 == 'G')
        a1[is_A] = 'T'
        a1[is_T] = 'A'
        a1[is_C] = 'G'
        a1[is_G] = 'C'
        df_merge['flip_a1'] = a1

        a2 = df_merge['a2_bim'].values.copy()
        is_A = (a2 == 'A')
        is_T = (a2 == 'T')
        is_C = (a2 == 'C')
        is_G = (a2 == 'G')
        a2[is_A] = 'T'
        a2[is_T] = 'A'
        a2[is_C] = 'G'
        a2[is_G] = 'C'
        df_merge['flip_a2'] = a2

        # do some standardization
        # try:
        #     is_same = ((df_merge['a1'] == df_merge['Allele1']) & (df_merge['a2'] == df_merge['Allele2'])).values
        #     is_reverse = ((df_merge['a2'] == df_merge['Allele1']) & (df_merge['a1'] == df_merge['Allele2'])).values
        #     is_flipped = (
        #     (df_merge['flip_a1'] == df_merge['Allele1']) & (df_merge['flip_a2'] == df_merge['Allele2'])).values
        #     is_reverse_flipped = (
        #     (df_merge['flip_a2'] == df_merge['Allele1']) & (df_merge['flip_a1'] == df_merge['Allele2'])).values
        # except:
        is_same = ((df_merge['a1_bim'] == df_merge['A1']) & (df_merge['a2_bim'] == df_merge['A2'])).values
        is_reverse = ((df_merge['a2_bim'] == df_merge['A1']) & (df_merge['a1_bim'] == df_merge['A2'])).values
        is_flipped = ((df_merge['flip_a1'] == df_merge['A1']) & (df_merge['flip_a2'] == df_merge['A2'])).values
        is_reverse_flipped = ((df_merge['flip_a2'] == df_merge['A1']) & (df_merge['flip_a1'] == df_merge['A2'])).values

        # decide which SNPs to keep
        keep_snps = ((is_same) | (is_reverse))

        # find the column of the effect sizes

        found_effects_col = False
        for effects_col in ['b', 'Beta', 'beta', 'effect', 'OR', 'MainEffects', "log_odds", "OR_fix",
                            "log_odds_(stage2)", "BETA", "Effect", "BMIadjMainEffects", "log10bf"]:  # "log_odds" was added by Yochai for the Cardio Estimation
            if (effects_col not in df_merge.columns): continue
            found_effects_col = True
            effects = df_merge[effects_col].values
        assert found_effects_col, 'couldn\'t find a column of effects:' + df_merge.columns.values

        if (((effects_col == 'OR') or (effects_col == 'OR_fix')) and (np.min(df_merge[effects_col].values) > 0)):
            df_merge['Beta'] = np.log10(df_merge[effects_col].values)
            effects_col='Beta'
        # flip effects if needed
        effects[is_reverse] *= (-1)

        # compute prediction for each p-values cutoff
        best_corr = -np.inf

        df_predictions.loc[df_predictions.index, 'predict_' + trait] = (bed.val[:, is_snp_found]).dot(
            effects)  # Performing the dot product
        print "Finished trait#",trait," in chromosom number", CHR_Num,"Which is:",str(f_i),"out of", len(files_dict)

    df_predictions.to_csv(Results_Folder+Job_Name+"_CHR_"+CHR_Num+".csv")
    print "Finished CHR#", CHR_Num

def Convert_to_Class(trait, Results_Folder):
    print "Start reading csv:", trait
    CSV_file = pd.read_csv(Results_Folder + "Final_Raw_SNPs" + trait + ".csv")
    print "Finished reading csv:", trait
    uniques={}
    print trait
    print CSV_file
    # print CSV_Dict[trait].isna().sum()
    CSV_file.set_index("eid", inplace=True, drop=True)
    print "Started filna:", trait
    CSV_file = CSV_file.fillna("-1")
    print CSV_file.isnull().sum()
    for col in CSV_file.columns.values:
        uniques[col] = CSV_file.loc[:, col].unique()
        for ind, val in enumerate(uniques[col]):
            if np.issubdtype(type(val), np.number):
                CSV_file.loc[CSV_file.loc[:, col] == val, col] = str(int(ind + 1))
        print CSV_file.loc[:, col].head()
    print "Started saving:", trait
    CSV_file.to_csv(path_or_buf=Results_Folder + "Final_Results/Final_SNPs_" + trait + ".csv", index=True)
    print "finished trait :",trait

    
    
    
    
    
    

