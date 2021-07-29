import PRS_sumstats
from matplotlib.backends.backend_pdf import PdfPages
import os
# from pandas.tools.plotting import table
import pandas as pd
RES=0.025
id=3
full_bfile_path="/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/cleanData/PNP_autosomal_clean2_nodfukim" #bfile that we want to calculate PRS for
PDF_FILE_PATH=os.path.join("/home/edlitzy/10K_PRS/PDF_Files/","Whole_PRS_2405.pdf")
import matplotlib.pyplot as plt
DPI=100
traits_dict = PRS_sumstats.get_traits_dict() #Dictionary with all Traits
pdf = PdfPages(PDF_FILE_PATH)
# df_prs = PRS_sumstats.compute_prs(bfile_path=full_bfile_path)

df_prs = PRS_sumstats.compute_prs(res=RES)

personal_scores= PRS_sumstats.Personal_PRS(bfile_path=full_bfile_path, ID=id, full_predictions=None, res=RES)
personal_scores_Explicit="Top "+ (personal_scores*RES*100).astype(int).astype(str)+"%"

fig = plt.figure()
ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
pd.plotting.table(ax, personal_scores_Explicit)  # where df is your data frame
pdf.savefig()
plt.close()

for col in personal_scores.columns.values:
    fig=plt.figure(figsize=(20, 20))
    ax = plt.subplot(111, frame_on=False)
    df_prs.loc[:,col].hist(bins=100,ax=ax)
    plt.suptitle(col,fontsize=24)
    ax.axvline(df_prs.loc[:, col].quantile(personal_scores.loc[id, col] * RES), color='r',linewidth=2)
    pdf.savefig()
    plt.close()

pdf.close()

print ("Finished")

