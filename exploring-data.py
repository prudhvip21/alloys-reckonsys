

a = 3


import pandas as pd
import matplotlib.pyplot as plt
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
pd.set_option('display.max_columns', 500)

tap = pd.read_excel('Data_AI-ML-New.xlsx', sheet_name='TAP_Analysis')
raw_materials = pd.read_excel('Data_AI-ML-New.xlsx', sheet_name='Heat_Raw_Materials')
output = pd.read_excel('Data_AI-ML-New.xlsx', sheet_name='Product_Aim')
heat_Details = pd.read_excel('Data_AI-ML-New.xlsx', sheet_name='HeatDetails')
heat_raw_materials_ctd = pd.read_excel('Data_AI-ML-New.xlsx', sheet_name='Heat_Raw_Materials_Ctd')


tap.drop(columns='Unnamed: 32', inplace= True)

#fceno could be furnace num. Irrelevant for now.

data_optimisation = tap.copy(deep= True)

# filtering for 2022,2023 data in tap analysis data.

data_optimisation['year'] = data_optimisation['Heat '].str.split('-').str[-1]
data_optimisation['year'] = data_optimisation['year'].astype('int')
data_optimisation_latest = data_optimisation[data_optimisation['year']>2021].copy(deep = True)

data_optimisation_latest.columns = [i.strip() for i in data_optimisation_latest.columns]

data_optimisation_latest.rename(columns={'O': 'O2', 'Err': 'Emg'}, inplace = True)
# filtering 22,23 in heat raw materials ctd 


heat_raw_materials_ctd['year'] = heat_raw_materials_ctd['Heat '].str.split('-').str[-1]
heat_raw_materials_ctd['year'] = heat_raw_materials_ctd['year'].astype('int')
heat_raw_materials_ctd_latest = heat_raw_materials_ctd[heat_raw_materials_ctd['year']>2021].copy(deep = True)
heat_raw_materials_ctd_latest.columns = [i.strip() for i in heat_raw_materials_ctd.columns]



# data_optimisation_latest = df of all tap analysis in 22,23


# filtering for 2022,2023 data in raw material data

raw_materials['year'] = raw_materials['Heat '].str.split('-').str[-1].astype(int)
raw_materials_latest = raw_materials[raw_materials['year']>2021]
raw_materials_latest.columns = [i.strip() for i in raw_materials_latest.columns]

# raw materials latest = df of all raw materials file greater than 2021.

heatwise_materials = raw_materials_latest.groupby('Heat')['Material_No'].agg(list).reset_index()
# heatwise_materials  = for each heat, get a list of raw materials used.

df_for_algo = pd.merge(data_optimisation_latest, heatwise_materials, on= 'Heat', how = 'inner')

df_for_algo.columns = [i.strip() for i in df_for_algo.columns]

# df_for_algo = tap analysis details plus raw materails added details - heat wise.

#for each of this heat, get a product aim

product_output = pd.merge(output,heat_Details[['Heat','Product_No']], left_on= 'ProductNo', right_on = 'Product_No', how = 'inner')

product_output['year'] = product_output['Heat'].str.split('-').str[-1].astype(int)
product_output_latest = product_output[product_output['year']>2021]
product_output_latest.columns = [i.strip() for i in product_output_latest.columns]

# product_output_latest - for each heat, expected output with product No.


""" For a single heat, creating df with input, additives, output - all in weights """

sample_heat = '01-01-04-04-A-2022'
sample_heat_for_raw_material_sheet = sample_heat.replace('-A', '')

# input weight calculation

# get weight from  heat raw materials sheet and multiple with ratios to get each metal weight for each heat

sample_input = df_for_algo[df_for_algo['Heat']== sample_heat]
sample_raw_mat = raw_materials_latest[(raw_materials_latest['Heat']== sample_heat) & (raw_materials_latest['Weight_Comp']>0)]

sample_raw_mat.loc[:,'Tap_Or_Lot'] = sample_raw_mat['Tap_Or_Lot'].str.rstrip()

input_weight = sample_raw_mat[sample_raw_mat['Tap_Or_Lot'] == sample_heat_for_raw_material_sheet].loc[:,'Weight_Comp'].iloc[0]


input_metal_wise_weight = sample_input.iloc[0,0:30]*input_weight/100

df_heat = pd.DataFrame(input_metal_wise_weight) # all weights for each heat
df_heat_ratio = pd.DataFrame(sample_input.iloc[0,0:30]) # all ratios for each heat

# additives weight computation. For each additive, get their weight, multiple with ratios in heat raw material ctd

additives = sample_raw_mat[sample_raw_mat['Tap_Or_Lot'] != sample_heat_for_raw_material_sheet]


for i,row in additives.iterrows():
    material = row['Material_No']
    weight = row['Weight_Act']
    material_weight = heat_raw_materials_ctd_latest[heat_raw_materials_ctd_latest['Material_No']==material].iloc[0,2:32]*weight/100
    df_heat.loc[:,material] = material_weight
    df_heat_ratio.loc[:,material] = heat_raw_materials_ctd_latest[heat_raw_materials_ctd_latest['Material_No']==material].iloc[0,2:32]


df_heat['calc_output'] = df_heat.sum(axis = 1)

# out put calculated with product aim.

sample_heat_product_ratio = product_output_latest[product_output_latest['Heat']==sample_heat].iloc[:,0:30].T
total_weight = sample_raw_mat['Weight_Act'].sum()
sample_heat_product = sample_heat_product_ratio*total_weight/100

df_heat.loc[:,'output_given'] = sample_heat_product
df_heat_ratio.loc[:,'expected_output_ratio'] = sample_heat_product_ratio


df_heat.to_csv('sample_heat_calc_actuals.csv')



## Genetic algorithm

import pygad
import numpy as np



metal_names = df_heat.index.tolist()
metal_names = df_heat_ratio['expected_output_ratio'].sort_values(ascending= False)[:6].index.tolist()
input_total_weight = df_heat.iloc[:,0].sum()
actual_solution = additives['Weight_Comp'].tolist()



def fitness_func_multi(ga_instance, solution, solution_idx):
    fitness = []
    for metal in metal_names:
        additive_metal_weight = np.sum(df_heat_ratio.loc[metal,:].iloc[1:-1] * solution / 100)
        input_metal_weight = df_heat.loc[metal, :].iloc[0]
        output_metal_weight = input_metal_weight + additive_metal_weight
        output_total_weight = input_total_weight + np.sum(solution)
        metal_ratio = (output_metal_weight / output_total_weight) * 100
        metal_fitness = np.abs(metal_ratio - df_heat_ratio.loc[metal, 'expected_output_ratio'])
        if metal == 'Emg':
            fitness.append(0)
        else :
            fitness.append(metal_fitness)
    return fitness


def fitness_func_single(ga_instance, solution, solution_idx):
    fitness = []
    for metal in metal_names:
        additive_metal_weight = np.sum(df_heat_ratio.loc[metal,].iloc[1:-1] * solution / 100)
        input_metal_weight = df_heat.loc[metal, :].iloc[0]
        output_metal_weight = input_metal_weight + additive_metal_weight
        output_total_weight = input_total_weight + np.sum(solution)
        metal_ratio = (output_metal_weight / output_total_weight) * 100
        metal_fitness = np.abs(metal_ratio - df_heat_ratio.loc[metal, 'expected_output_ratio'])
        if metal == 'Emg':
            fitness.append(0)
        else :
            fitness.append(metal_fitness)
    return 1/sum(fitness)




num_generations = 50
num_parents_mating = 200
mutation_num_genes= 4
sol_per_pop = 500
num_genes = len(additives['Weight_Comp'].tolist())
gene_space = {'low': 100, 'high': 1200}

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       init_range_low = 0,
                       init_range_high = 1200,
                       keep_parents= 100,
                       crossover_type="uniform",
                       mutation_percent_genes= mutation_num_genes,
                        random_mutation_min_val= 10,
                        random_mutation_max_val=1200,
                       gene_space= gene_space,
                       num_genes=num_genes,
                       fitness_func= fitness_func_single,
                       parent_selection_type='sss')
# tournament_nsga2
ga_instance.run()
winsound.Beep(frequency, duration)

solution_best, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)


actual_solution
solution_best


actual_ratio = fitness_func_multi(ga_instance,actual_solution,solution_idx)
#actual_ratio = fitness_func_single(ga_instance,actual_solution,solution_idx)
#actual_ratio = [np.round(1/item,2) for item in actual_ratio if item!=0]
print(np.round(actual_ratio,2))

model_ratio = fitness_func_multi(ga_instance,solution_best,solution_idx)
#model_ratio = [np.round(1/item,2) for item in model_ratio if item!=0]
print(np.round(model_ratio,2))



actual_solution
np.round(solution,2)
np.round(solution_fitness,2)



ga_instance.plot_fitness()
plt.show()



print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {np.round(solution_fitness,2)}")

