import jw_signals.data_pipeline as dp
from datetime import datetime
import os
import pandas as pd

os.chdir('sample_data')

def file_name_parser(string):
    elements = ['title', 'test_num', 'test_rep', 'location', 'datetime', 'distance_xyz']
    string = string.split('_')
    
    ret = [string[0], string[2][string[2].rfind('-')+1:], string[1], datetime.strptime(string[-1], '%Y%m%d-%H%M%S')]
    
    title = 'test%s-%s: %s' % (ret[0], ret[1], ret[2])
    
    distances = {'1' : '0.81m', '2' : '1.5m', '3' : '1.5m', '4' : '3.81m', '5' : '0.81m'}
    
    distance = distances[ret[0]]
    
    return pd.Series([title] + ret + [distance], index = elements)

data = dp.multipleDaqData(read_columns = ['time', 'A0', 'Ax', 'Ay', 'Az'], 
                          #analyse_columns = ['A0', 'Ax'],
                          units = {'time' : 's', 'A0' : 'g', 'Ax' : 'g', 'Ay' : 'g', 'Az' : 'g'},
                          ratios = ['Ax/A0', 'Az/Ax', 'Ay/Ax'], 
                          noise_col_regex = '.*test\d-0.*',
                          reg_str = '.*test(1|3|4)-(0|1|2).*.txt', 
                          file_name_parser = file_name_parser, 
                          plots = [],
                          window = {'type': 'hann', 'length_n': None, 'start_n' : None, 'params' : []},
                          quick_load = True)

data.find_variance_all_col(close_plot = True, quick_load = True)

data.find_variance_all_col(close_plot = False, quick_load = True, group_by = 'distance_xyz')

_ = data.plot_mean_fft(grouped_by = 'test_num', groups_to_plot = [], labels = ['0.81m', '1.5m', '3.81m'], 
                          smoothed = True, x_lim = (0, 1000), close_plot = False)

#f_coh, coh, figs = data.find_coherence([d for d in data.data_names if d.find('-1') != -1], close_plot = False)

#data.plot_mean_fft_ratio('Ax/Az', 'test_num', groups_to_plot = ['3'], smoothed = True, 
#                  labels = [], x_lim = (None, None), close_plot = False)
