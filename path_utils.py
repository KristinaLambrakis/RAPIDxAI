import os

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
config_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

master_cache_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
# cache the original data's file
cache_root_d0 = os.path.join(master_cache_root, 'cache_d0')
# cache the checked and modified original data's location
cache_root = os.path.join(master_cache_root, 'cache')
# cache for the rapidx-x second dataset
cache_root_d2 = os.path.join(master_cache_root, 'cache_d2')
# cache for the revascularization data
cache_root_dr = os.path.join(master_cache_root, 'cache_dr')
# cache for the ecg data
cache_root_de = os.path.join(master_cache_root, 'cache_de')
# cache for rapid-x third dataset (external validation), as per Joey's email "Adjudicated datasets" on 12/7/2022,
# filename: adjudicated_for_zhibin_June2022.csv - Joey indicated to use this file instead of the
# adjudicated_for_zhibin_July2022.csv. However, the as per communication with Joey (email: Question) on 28/07/2022,
# The hba1c definition (hba1c% vs hba1c mmolL) is different between data3 and data2, hence Joey sent another dataset
# adjudicated_for_zhibin_Aug2022.csv to change the hba1c mmolL to %. The adjudicated_for_zhibin_Aug2022.csv data is
# used. Joey also expressed that prioracs is not correct in data3 and possibly data_ecg, which then should be removed.
# Furthermore, the data3 is a superset of data_ecg.
cache_root_d3 = os.path.join(master_cache_root, 'cache_d3')

output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
pytorch_data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aiml/pytorch/data')
pytorch_data_server_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aiml/pytorch/data_server')

model_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

documentation_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'documentation')
