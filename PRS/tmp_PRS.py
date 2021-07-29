from PRS import PRS_extract_phenotypes
import PRS_sumstats
full_bfile_path="/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/cleanData/PNP_autosomal_clean2_nodfukim"
#extract phenotypes IID Vs. Measured Phenotypes
df_pheno = PRS_extract_phenotypes.extract('s_stats_pheno') #Used for training set

#extract the predicted trait
trait = 'bmi' #
traits_dict = PRS.PRS_sumstats.get_traits_dict() #Dictionary with all Traits
assert (trait in traits_dict)
pheno_col = traits_dict[trait]
df_pheno_train = df_pheno[[pheno_col]] #Building Training set
train_index = df_pheno_train.index[:100] #use the first 100 individuals for training
df_prs = PRS.PRS_sumstats.compute_prs(df_pheno, bfile_path=full_bfile_path, train_indices=train_index, trait_name=trait)
df_prs.hist(bins=100)
print ("Finished")