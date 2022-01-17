import pandas
import os
import numpy as np
import sys
from pysnptools.snpreader.bed import Bed
import subprocess
cleanDataPath='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/cleanData/'
rawDataPath='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/rawData'
pheno_fn_bac =os.path.join(cleanDataPath,'noMissingKakiPhenotypesWithCovariates_nodfukim.phenotypes')
#pheno_fn_bac =os.path.join(cleanDataPath,'allChipPhenotypes_nodfukimWith5PCair.phenotypes')
pheno_fn_bacDic=os.path.join(cleanDataPath,'dicNoMissingKakiPhenotypesWithCovariates_nodfukim.phenotypes')
pheno_fn_bacAllPNP=os.path.join(rawDataPath,'allPNPPhenotypes.phenotypes')
iidsNoSharedEnv='/net/mraid08/export/genie/Microbiome/Analyses/PNPChip/cleanData/PNP_autosomal_clean2_nodfukim_NoCouples.txt'
PNP_16S_DIR = '/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/16S'
glycemicStatusPath='/net/mraid08/export/jafar/Microbiome/Analyses/PNPChip/glycemic_status.csv'

def extract(*args,**kwargs):
    
    known_args = ['dic', 'all_bac', 's', 'g','f','o','c','p','otu', 'all_non_bac', 'covars', 'blood', 'glucose', 'ffq', 'antropo', 
                   's_stats_pheno', 'fid', 'keep_household', 'no_log', 'keep_related', 'keep_sterile', '16s', '-9', 
                   'covars_noPCs', 'PCs', 'lactose','include_allPNP','IsGenotek','permute','meals','other','drugs',
                   'calories','bloodType','questionnaires','keep_missingCovars','activity','activityTypesFreq','cereals', 'delivery','dressSweetners','drinks','fruits','hunger', 
                  'legumes','meatProducts','pastry','qualityOfLiving', 'smoking','sweets','vegetables','womenOnlyQuestions',
                  'genotek_only', 'swab_only']
    ffq_args = ['activity','activityTypesFreq','cereals', 'delivery','dressSweetners','drinks','fruits','hunger', 
                  'legumes','meatProducts','pastry','qualityOfLiving', 'smoking','sweets','vegetables','womenOnlyQuestions']
    drug_args=['D.lipid', 'D.All', 'D.Psychiatric', 'D.pain', 'D.CVD', 'D.GI','D.Thyroid', 'D.NSAID','D.Contraception']
    meals=['Vodka or Arak', 'Avocado', 'Parsley', 'Coated peanuts',
       'Sugar', 'Smoked Salmon', 'Melon', 'Roll', 'Whipped cream',
       'Coconut milk', 'Pretzels', 'Kohlrabi', 'Eggplant Salad',
       'Cooked green beans', 'Cooked mushrooms', 'Watermelon',
       'Grilled cheese', 'Bissli', 'Pullet', 'Hummus',
       'Chinese Chicken Noodles', 'Shakshouka', 'Tahini',
       'Chicken breast', 'Steak', 'Light Bread',
       'Wholemeal Crackers', 'Sugar Free Gum', 'Hamburger',
       'Dark Beer', 'Cooked beets', 'Almonds', 'Falafel', 'Noodles',
       'Jachnun', 'Turkey', 'Sushi', 'Brazil nuts', 'Orange', 'Rice',
       'Diet Fruit Drink', 'Corn schnitzel', 'Cappuccino',
       'Low fat Milk', 'Pickled cucumber', 'Soymilk',
       'Dates', 'Croissant', 'Biscuit', 'Potato chips',
       'White Cheese', 'French fries', 'Wholemeal Bread', 'Tuna Salad',
       'Chocolate spread', 'Kebab', 'Rice crackers', 'Wafers',
       'Lettuce', 'Rice Noodles', 'Lentils', 'Mutton',
       'Wholemeal Noodles', 'Green Tea', 'Schnitzel', 'Brown Sugar',
       'Peanuts', 'Mayonnaise', 'Persimmon', 'Apple juice',
       'Stuffed Peppers', 'Egg', 'Pear', 'Peas', 'Pecan',
       'Cooked cauliflower', 'Cooked Sweet potato', 'Butter',
       'Omelette', 'Coated Wafers', 'Boiled corn', 'Chicken drumstick',
       'Pita', 'Pasta Bolognese', 'Chicken Meatballs', 'Burekas',
       'Carrots', 'Tofu', 'Wholemeal Pita', 'Sunflower seeds',
       'Coriander', 'Ciabatta', 'Tomato sauce', 'Heavy cream',
       'Banana', 'Kif Kef', 'Mustard', 'Coke', 'Vegetable Soup',
       'Sausages', 'Pancake', 'Pasta', 'Sauteed vegetables', 'Plum',
       'Goat Milk Yogurt', 'Orange juice', 'Potatoes', 'Halva',
       'Yellow pepper', 'Mango', 'Lasagna', 'Popcorn', 'Hummus Salad',
       'Tilapia', 'Pizza', 'Fried cauliflower', 'Roasted eggplant',
       'Baguette', 'Lentil Soup', 'Tzfatit Cheese', 'Nectarine',
       'Chicken legs', 'Nuts', 'Goat Cheese', 'Jam', 'Feta Cheese',
       'Mandarin', 'Pesto', 'Sugar substitute', 'Cheesecake',
       'Raisins', 'Chocolate', 'Quinoa', 'Cooked broccoli',
       'Beef Cholent', 'Cracker', 'Chocolate Cookies', 'White beans',
       'Cooked zucchini', 'Sweet potato', 'Wine', 'Cookies',
       'Challah', 'Spelled', 'Honey', 'Green beans', 'Milk',
       'Peanut Butter', 'Cooked carrots', 'Lemon', 'Salty Cookies',
       'Beef', 'Meatballs', 'Hamburger sandwich', 'Chicken thighs',
       'Granola', 'Beet', 'Couscous', 'Beet Salad',
       'Chocolate Mousse Cake', 'Sweet Roll', 'Danish', 'Coffee',
       'Pasta Salad', 'Cuba', 'Chicken Liver', 'Sweet Challah',
       'Minced meat', 'Chocolate cake', 'Diet Coke', 'Dried dates',
       'Carrot Cake', 'Doritos', 'Israeli couscous', 'Pistachio',
       'Date honey', 'Vinaigrette', 'Bamba', 'Dark Chocolate',
       'Turkey Shawarma', 'Olive oil', #u'Parmesan\xc2\xa0cheese',
       'Guacamole', 'Coleslaw', 'Americano', 'Pesek Zman snack',
       'Green onions', 'Mushrooms', 'Lemon juice', 'Canned Tuna Fish',
       'Vegetable Salad', 'Fried eggplant', 'Salmon', 'Cashew',
       'Jewish donut', 'Rugelach', 'Cake', 'Ravioli', 'Tomatoes',
       'Wholemeal Light Bread', 'Marble Cake', 'Brown Rice',
       'Cold cut', 'Gilthead Bream', 'Garlic', 'Grapes',
       'Chocolate Chip Cookies', 'Cucumber', 'Mung Bean', 'Ketchup',
       'Sweet Yogurt', 'Bread', 'Onion', 'Cream Cheese',
       'Chicken soup', 'Wholemeal Roll', 'Canned corn', 'Salty Cheese',
       'Melawach', 'White cake', 'Apple', 'Lettuce Salad', 'Cereals',
       'Yellow Cheese', 'Tea', 'Beer', 'Mozzarella Cheese',
       'Fried onions', 'Ice cream', 'Cream Cake', 'Green cabbage',
       'Olives', 'Balsamic vinegar', 'Peach', 'Light Yellow Cheese',
       'Red pepper', 'Bagel', 'Entrecote', 'Cottage cheese', 'Oil',
       'Natural Yogurt', 'Walnuts', 'Edamame', 'Majadra', 'Oatmeal',
       'Soy sauce', 'Strawberry', 'Pastrami', 'Lemonade',
        'Pasta with tomato sauce', 'Chicken']#removed: u'Soda water',u'Water', u'Salt',
    known_args+= ffq_args              
    known_args+= drug_args         
    known_kwargs = ['ratio', 'threshold','taxa']
    
    for arg in args: assert arg in known_args, 'unkown arg: %s'%(arg)
    for kwarg in list(kwargs.keys()): assert kwarg in known_kwargs, 'unkown kwarg: %s'%(kwarg)
    if ('16s' in args): assert 'dic' not in args, '16s and dic are mutually exclusive'
    if ('taxa' in list(kwargs.keys())): assert len(set(['all_bac','s', 'g','f','o','c','p','otu']).intersection(set(args)))==0, \
    'taxa is mutual exclusive with all_bac,s,g,f,o,c,p,otu'
    if 'include_allPNP' in args: assert 'dic' not in args, 'include_allPNP does not support dicotomize bacteria'
    if 'IsGenotek' in args: assert 'covars' not in args, 'IsGenotek and covars are mutually exclusive'
    if 'otu' in args: assert '16s' in args 
        
    if 'dic' in args:
        pheno =pandas.read_csv(pheno_fn_bacDic,sep='\t')
        pheno.set_index('IID', inplace=True, drop=True)
        pheno_nodic =pandas.read_csv(pheno_fn_bac,sep='\t')
        pheno_nodic.set_index('IID', inplace=True, drop=True)
        pheno_s = pheno_nodic[[c for c in pheno_nodic.columns if c[:2]=='s_']]        
        pheno_g = pheno_nodic[[c for c in pheno_nodic.columns if c[:2]=='g_']]
    else:
        if 'include_allPNP' in args:
           pheno =pandas.read_csv(pheno_fn_bacAllPNP,sep='\t')
        else:
            pheno =pandas.read_csv(pheno_fn_bac,sep='\t')
        pheno.set_index('IID', inplace=True, drop=True)
        if 'include_allPNP'in args:
            status, output = subprocess.getstatusoutput("cut -f 1 %s -d ' ' | cut -f 1 -d '_'"%os.path.join(rawDataPath,'tmp','dfukim.txt'))
            pheno =pheno[~pheno.index.isin([int(dafook) for dafook in output.split('\n')])]
        if ('16s' in args):
            pheno = pheno[[c for c in pheno if c[:2] not in ('s_', 'g_', 'f_', 'o_', 'c_', 'p_')]]
            for taxa_level in ['otu', 'species', 'genus', 'family', 'order', 'class', 'phylum']:
                df_taxa = pandas.read_csv(os.path.join(PNP_16S_DIR, taxa_level+'.txt'), sep='\t', index_col=0)
                df_taxa[df_taxa<1e-3] = 1e-4
                df_taxa = np.log10(df_taxa)
                pheno = pheno.merge(df_taxa, left_index=True, right_index=True)
        
        pheno_s = pheno[[c for c in pheno.columns if c[:2]=='s_']]
        pheno_g = pheno[[c for c in pheno.columns if c[:2]=='g_']]
        
###     for c in pheno:
###         if (c[:2] not in ['c_', 'g_', 'o_', 's_', 'k_', 'p_', 'f_']): print c
        
    alpha_diversity_s = (pheno_s>pheno_s.min().min()).sum(axis=1)
    alpha_diversity_g = (pheno_g>pheno_g.min().min()).sum(axis=1)
    pheno.loc[pheno.Hips==-9, 'WHR'] = np.nan
    pheno.loc[pheno.Waist==-9, 'WHR'] = np.nan
    pheno['LDLCholesterol'] = pheno['Cholesterol,total'] - pheno['HDLCholesterol'] - 2*pheno['Triglycerides']
    
    if 'genotek_only' in args:
        pheno = pheno.loc[pheno['IsGenotek']==1]  
    if 'swab_only' in args:
        pheno = pheno.loc[pheno['IsGenotek']==0]  
    
    mb_columns = []
    if 'taxa' in kwargs:
        if kwargs['taxa'][0]=='*':
            kwargs['taxa']=[initial+kwargs['taxa'][1:] for initial in ('s_', 'g_', 'f_', 'o_', 'c_', 'p_')]
        elif kwargs['taxa'][1]=='_':
            kwargs['taxa']=[kwargs['taxa']]
        for taxa in kwargs['taxa']:
            taxadf=pheno.filter(regex=(taxa))
            mb_columns += taxadf.columns.values.tolist()
    if 'all_bac' in args:
        args=list(args)+['s','g','f','o','c','p']
    if 's' in args:
        mb_columns += [c for c in pheno.columns if c[:2]=='s_' ]
    if 'g' in args:
        mb_columns += [c for c in pheno.columns if c[:2]=='g_' ]
    if 'f' in args:
        mb_columns += [c for c in pheno.columns if c[:2]=='f_' ]
    if 'o' in args:
        mb_columns += [c for c in pheno.columns if c[:2]=='o_' ]
    if 'c' in args:
        mb_columns += [c for c in pheno.columns if c[:2]=='c_' ]
    if 'p' in args:
        mb_columns += [c for c in pheno.columns if c[:2]=='p_' ]
    if 'otu' in args:
        mb_columns += [c for c in pheno.columns if c[:4]=='OTU_' ]
       
    if 'no_log' in args:
        assert 'dic' not in args, 'dic and no_log are mutually exclusive'
        pheno[mb_columns] = 10**pheno[mb_columns]
        
    if 'all_non_bac' in args:
        args=list(args)+['covars','blood','glucose','ffq','antropo']
    mb_columns += ['Age','Gender','Calories_kcal','Carbs_g','Fat_g','Protain_g','IsGenotek']
    if 'include_allPNP' not in args or ('PCs') in args:
        mb_columns += [c for c in pheno.columns if c[:2]=='PC']  
    if 'lactose' in args:
        mb_columns += ['lactose']    

    if 'blood' in args:
#         mb_columns += ['ALT','Albumin','AST','Basophils%','Calcium','Chloride','Cholesterol,total','Creatinine',
#                        'CRP(WIDERANGE)','CRPhs','Eosinophils%','HCT','HDLCholesterol','Hemoglobin','HbA1C%','Lymphocytes%',
#                        'MCH','MCHC','MCV','MPV','Monocytes%','Neutrophils%','Phosphorus','Platelets','Potassium','RBC',
#                        'RDW','Sodium','TSH','WBC','AlkalinePhosphatase','GGT','LDH','Iron','LDLCholesterol','Magnesium',
#                        'Triglycerides','TotalProtein','TotalBilirubin','Urea']
        mb_columns += ['ALT','Albumin','AST','Basophils%','Calcium','Chloride','Cholesterol,total','Creatinine',
                       'CRP(WIDERANGE)','CRPhs','Eosinophils%','HCT','HDLCholesterol','Hemoglobin','HbA1C%','Lymphocytes%',
                       'MCH','MCHC','MCV','MPV','Monocytes%','Neutrophils%','Phosphorus','Platelets','Potassium','RBC',
                       'RDW','Sodium','TSH','WBC','LDLCholesterol']
                       
        
    if 'glucose' in args:
        mb_columns += ['95P_Glucose','Glucose_Noise','Max_Glucose','Median_Glucose','WakeupGlucose',
                       'MeanGlucoseResponse','MeanBreadResponse','MeanBreadButterResponse']
    if 'ffq' in args:
        mb_columns += ['Alcoholic_Drinks_Freq','Cigarretes_per_day','Coffee_Freq','Start_smoking_age']        
    if 'antropo' in args:
        mb_columns += ['BMI','BPdia','BPsys','HeartRate','Height','Hips','WHR','Waist']
    if 's_stats_pheno' in args:
        s_stats=['BMI','Cholesterol,total','WakeupGlucose','Albumin','Creatinine','HbA1C%','Height','Hips','Waist','WHR','HDLCholesterol'] #, 'Triglycerides', 'LDLCholesterol']         
        mb_columns+=s_stats
        mb_columns=list(set(mb_columns))
    if 'fid' in args:
        mb_columns = ['FID']+mb_columns
    ########################FFQ START#####################
    if 'questionnaires' in args:
        args=list(args)+ffq_args
    mb_columns_extra=[]    
    if 'activity' in args:
        mb_columns_extra += ['Work activity','Physical activity - mins','Physical activity - freq']
    if 'activityTypesFreq' in args:
        mb_columns_extra += ['T1Activity kind','Type 1 activity - freq','T2Activity kind',
                       'Type 2 activity - freq','T3Activity kind','Type 3 activity - freq']
    if 'bloodType' in args:
        mb_columns_extra += ['Blood A','Blood B','Blood RH-']
    if 'cereals' in args:
        mb_columns_extra += ['Cornflakes Freq','Granola or Bernflaks Freq','Cooked Cereal such as Oatmeal Porridge Freq',
                       'Rice Freq','Couscous, Burgul, Mamaliga, Groats Freq', 'Potatoes Boiled, Baked, Mashed, Potatoes Salad Freq',
                       'Fries  Freq', 'Pasta or Flakes Freq']
    if 'delivery' in args:
        mb_columns_extra += ['C-Section','Home delivery','Was breastfed']
    if 'dressSweetners' in args:
        mb_columns_extra += ['Oil as an addition for Salads or Stews Freq','Mayonnaise Including Light Freq', 
                       'Thousand Island Dressing, Garlic Dressing Freq', 'Honey, Jam, fruit syrup, Maple syrup Freq',
                       'White or Brown Sugar Freq', 'Artificial Sweeteners Freq',]
    if 'drinks' in args:
        mb_columns_extra += ['Nectar, Cider Freq', 'Diet Juice Freq', 'Juice Freq', 'Diet Soda Freq', 
                       'Regular Sodas with Sugar Freq', 'Decaffeinated Coffee Freq', 'Coffee Freq', 'Herbal Tea Freq',
                       'Green Tea Freq', 'Regular Tea Freq', 'Beer Freq', 'Sweet Dry Wine, Cocktails Freq', 'Alcoholic Drinks Freq']
    if 'fruits' in args:
        mb_columns_extra += ['Mandarin or Clementine Freq', 'Orange or Grapefruit Freq', 'Orange or Grapefruit Juice Freq', 
                       'Apple Freq', 'Apricot Fresh or Dry, or Loquat Freq', 'Grapes or Raisins Freq', 'Banana Freq',
                       'Melon Freq', 'Kiwi or Strawberries Freq', 'Mango Freq', 'Peach, Nectarine, Plum Freq', 
                       'Pear Fresh, Cooked or Canned Freq','Persimmon Freq', 'Watermelon Freq', 'Dried Fruits Freq', 'Fruit Salad Freq']
    if 'hunger' in args:
        mb_columns_extra += ['General Hunger','Morning Hunger', 'Midday Hunger', 'Evening Hunger']
    if 'legumes' in args:
        mb_columns_extra += ['Falafel in Pita Bread Freq', 'Cooked Legumes Freq', 'Processed Meat Free Products Freq']
    if 'meatProducts' in args:
        mb_columns_extra += ['Egg Recipes Freq', 'Egg, Hard Boiled or Soft Freq', 'Schnitzel Turkey or Chicken Freq', 
                       'Chicken or Turkey With Skin Freq', 'Chicken or Turkey Without Skin Freq', 'Sausages Freq', 
                       'Sausages such as Salami Freq', 'Pastrami or Smoked Turkey Breast Freq', 
                       'Turkey Meatballs, Beef, Chicken Freq', 'Shish Kebab in Pita Bread Freq', 
                       'Falafel in Pita version 2 Freq','Processed Meat Products Freq','Beef, Veal, Lamb, Pork, Steak, Golash Freq',
                       'Mixed Meat Dishes as Moussaka, Hamin, Cuba Freq', 'Mixed Chicken or Turkey Dishes Freq', 
                       'Beef or Chicken Soup Freq', 'Internal Organs Freq', 'Fish Cooked, Baked or Grilled Freq', 'Fried Fish Freq',
                       'Canned Tuna or Tuna Salad Freq', 'Fish (not Tuna) Pickled, Dried, Smoked, Canned Freq']
    if 'pastry' in args:
        mb_columns_extra += ['Ordinary Bread or Challah Freq', 'Light Bread Freq', 'Wholemeal or Rye Bread Freq', 'Baguette Freq', 
                       'Roll or Bageles Freq', 'Pita Freq', 'Saltine Crackers or Matzah Freq', 'Wholemeal Crackers Freq', 
                       'Small Burekas Freq', 'Jachnun, Mlawah, Kubana, Cigars Freq', 'Pizza Freq']
    if 'qualityOfLiving' in args:
        mb_columns_extra += ['Stress','Sleep quality']
    if 'smoking' in args:
        mb_columns_extra += ['Currently smokes','Ever smoked']
    if 'sweets' in args:
        mb_columns_extra +=  ['Milk or Dark Chocolate Freq', 'Salty Snacks Freq', 'Cheese Cakes or Cream Cakes Freq', 
                        'Yeast Cakes and Cookies as Rogallach, Croissant or Donut Freq', 'Cake, Torte Cakes, Chocolate Cake Freq',
                        'Fruit Pie or Cake Freq', 'Coated or Stuffed Cookies, Waffles or Biscuits Freq', 
                        'Simple Cookies or Biscuits Freq', 'Ice Cream or Popsicle which contains Dairy Freq', 
                        'Popsicle Without Dairy Freq', 'Black or White Grains, Watermelon Seeds Freq', 
                        'Nuts, almonds, pistachios Freq','Peanuts Freq']
    if 'vegetables' in args:
        mb_columns_extra += ['Tomato Freq','Cooked Tomatoes, Tomato Sauce, Tomato Soup Freq', 'Red Pepper Freq', 'Green Pepper Freq', 
                       'Cucumber Freq', 'Zucchini or Eggplant Freq','Peas, Green Beans or Okra Cooked Freq', 
                       'Cauliflower or Broccoli Freq','Sweet Potato Freq', 'Brussels Sprouts, Green or Red Cabbage Freq', 
                       'Lettuce Freq','Carrots, Fresh or Cooked, Carrot Juice Freq', 'Corn Freq', 
                       'Parsley, Celery, Fennel, Dill, Cilantro, Green Onion Freq', 
                       'Fresh Vegetable Salad Without Dressing or Oil Freq', 'Fresh Vegetable Salad With Dressing or Oil Freq', 
                       'Avocado Freq','Lemon Freq', 'Onion Freq', 'Garlic Freq', 'Vegetable Soup Freq', 'Hummus Salad Freq', 
                       'Tahini Salad Freq', 'Cooked Vegetable Salads Freq', 'Pickled Vegetables  Freq', 'Olives Freq']
    if 'womenOnlyQuestions' in args:
        mb_columns_extra += ['Is pregnant','Is breastfeeding','Is after birth', 'Taking contraceptives', 'Regular period', 
                       'Irregular period', 'No period','Hormonal replacment', 'Past breastfeeding']
    if 'other' in args:
        #AddingIrisGlucose
        df_glucose = pandas.read_csv(glycemicStatusPath).set_index('RegNum')
        pheno = df_glucose.merge(pheno, left_index=True, right_index=True,how='right')
        mb_columns +=['median_Without_BMI_ALT_Overall','WakeupGlucose','BMI','VegeterianScale']
        pheno.loc[pheno['VegeterianScale']<0, 'VegeterianScale']=np.nan
    if 'drugs' in args:
        mb_columns+=drug_args
    else:
        for arg in drug_args:
            if arg in args:
                mb_columns += [arg]

    mb_columns_extra=[val.replace(' ','_') for val in mb_columns_extra]
    mb_columns+=mb_columns_extra
    if 'meals' in args:
        mealsColumns=[val.replace(' ','_') for val in meals]
        #Correct by total calories
        pheno.loc[:,mealsColumns]=pheno[mealsColumns][pheno[mealsColumns]!=-9].div(pheno['Calories_kcal_Total'].values,axis=0)
        pheno.replace(np.nan, 0,inplace=True)
        mb_columns += mealsColumns
    ########################FFQ END#####################
    #for c in pheno: print c
    mb_columns=list(set(mb_columns))
    pheno= pheno[mb_columns]
    if 'threshold' not in kwargs:
        threshold = -4
    else:
        threshold=kwargs['threshold']
    if 'ratio' in kwargs:
        ratio=kwargs['ratio']
        mb_columns = [c for c in pheno.columns if c[:2] in ['s_','g_','f_','o_','c_','p_']]
        other_columns = [c for c in pheno.columns if c[:2] not in ['s_','g_','f_','o_','c_','p_']]
        if 'dic' in args:
            presence=((pheno[mb_columns]>threshold +1e-5)&(pheno[mb_columns]!=0)).astype(int).sum()
        else:
            presence=(pheno[mb_columns]>threshold +1e-5).astype(int).sum()
        presence=presence[presence > len(presence)*ratio].index.values.tolist()
        pheno=pheno[other_columns+presence]
        
    if ('keep_related' not in args):
        #bed = Bed(os.path.join(cleanDataPath, 'PNP_autosomal_clean2_nodfukim_norelated'), count_A1=True)#.read()
        df_fam_no_related = pandas.read_csv(os.path.join(cleanDataPath, 'PNP_autosomal_clean2_nodfukim_norelated.fam'), delim_whitespace=True, index_col=0, header=None)
        df_fam = pandas.read_csv(os.path.join(cleanDataPath, 'PNP_autosomal_clean2_nodfukim.fam'), delim_whitespace=True, index_col=0, header=None)
        df_related=df_fam[~df_fam.index.isin(df_fam_no_related.index)]
        pheno=pheno[(~pheno.index.isin(df_related.index))] 
        
    if ('keep_sterile') not in args:
        if '16s' in args: sterile_individuals = alpha_diversity_g[alpha_diversity_g < 4].index
        else: sterile_individuals = alpha_diversity_s[alpha_diversity_s < 15].index
        pheno=pheno[~pheno.index.isin(sterile_individuals)]
        
    if 'keep_household' not in args:
        #noSharedEnvIID=pandas.read_csv(iidsNoSharedEnv,usecols=[0],header=None,sep='\t')
        #pheno=pheno[pheno.index.isin(noSharedEnvIID[0].astype(int).values)]
        
        #new code that decides which individuals to remove on the fly
        import ForPaper.VertexCut as vc
        df_household = pandas.read_csv(os.path.join(cleanDataPath, 'EnvironmentBlock.txt'), delim_whitespace=True)
        df_household = df_household[[c for c in df_household.columns if int(c) in pheno.index]]
        df_household = df_household[df_household.index.isin(pheno.index)]
        remove_inds = df_household.index[vc.VertexCut().work(df_household.values, 0.5)]
        pheno=pheno[~pheno.index.isin(remove_inds)]
    
    if 'keep_missingCovars' not in args:
        #One participant 244624 has no 'Calories_kcal','Carbs_g','Fat_g','Protain_g'
        #3 participant 86356,762339,805175 have no 'Age','Gender'
#         if set(['Age','Gender','Calories_kcal','Carbs_g','Fat_g','Protain_g','IsGenotek'])<=set(pheno.columns.values): 
        keep_inds=pheno.loc[:,['Age','Gender','Calories_kcal','Carbs_g','Fat_g','Protain_g','IsGenotek']].replace(-9, np.nan).dropna().index.values
        beforeNumParticpants=pheno.shape[0]
        pheno=pheno.loc[keep_inds]
        afterNumParticpants=pheno.shape[0]
        if beforeNumParticpants-afterNumParticpants>0:
            pass
            #print "Removing participants with missing covars!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            #print beforeNumParticpants-afterNumParticpants
            #print "805175 has no FFQ!!!!! that is why we remove him"
        features_to_drop=[]
        if ('IsGenotek' not in args) and ('covars' not in args) and ('covars_noPCs' not in args):
            features_to_drop += ['IsGenotek']
        if ('covars' not in args) and ('covars_noPCs' not in args) and ('other' not in args):
            if 'calories' not in args:
                features_to_drop +=['Age','Gender','Calories_kcal','Carbs_g','Fat_g','Protain_g','IsGenotek']
            else:
                features_to_drop +=['Age','Gender']
            if ('include_allPNP' not in args) and ('PCs' not in args):
                features_to_drop+=['PC1','PC2','PC3','PC4','PC5']
        pheno=pheno.drop(features_to_drop,axis=1)          
    if ('-9' not in args):
        pheno.replace(-9, np.nan, inplace=True)
    
    if 'permute' in args:
        pheno=pandas.DataFrame(pheno.values[np.random.permutation(pheno.shape[0])],index=pheno.index,columns=pheno.columns)
    return pheno
if __name__=="__main__":
#     pheno=extract('dic','covars','keep_household',"pastry",ratio=0.2)#'all_bac'
    phenoAll = extract('s','include_allPNP','covars')#'include_allPNP','keep_household','ffq','keep_related')#'include_allPNP',
    print(phenoAll.shape)
    print(phenoAll.columns)
    phenoAll = extract('s','include_allPNP')
    print(phenoAll.shape)
    print(phenoAll.columns)
    phenoChip = extract('keep_household','s','keep_related')
    print(phenoChip.shape)
    print(phenoChip.columns)
#     print "Only in chip:"
#     print set(phenoChip.index.values)-set(phenoAll.index.values)
#     print len(set(phenoChip.index)-set(phenoAll.index))
    
    print(pheno.columns.values.tolist())
    print(pheno.shape) 
    sum=0
    for participant in  pheno[['Age','Gender']].index.values:
#         if np.isnan(pheno.loc[participant,'Calories_kcal']) or \
#          np.isnan(pheno.loc[participant,'Carbs_g']) or \
#          np.isnan(pheno.loc[participant,'Fat_g']) or \
#          np.isnan(pheno.loc[participant,'Protain_g']):
#             sum+=1
#             print participant
#             print pheno.loc[participant,['Calories_kcal','Carbs_g','Fat_g','Protain_g','Protain_g']]
#     print sum
        if np.isnan(pheno.loc[participant,'Age']) or np.isnan(pheno.loc[participant,'Gender']) :
            print("Participant %s, age %s, gender %s" %(participant,pheno.loc[participant,'Age'],pheno.loc[participant,'Gender'])) 
#     print pheno[['median_Without_BMI_ALT_Overall']]
    
     
        