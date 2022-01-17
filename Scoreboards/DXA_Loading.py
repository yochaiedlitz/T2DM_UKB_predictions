from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
subjects = SubjectLoader()
rld = DEXALoader(gen_cache=True)
data = rld.get_data(study_ids=['10K'])
print(data.df.columns)
print("Wait")

