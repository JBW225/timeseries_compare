
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from nptdms import TdmsFile
import os
import pandas as pd
import regex as re
import scipy.signal as sig
import shutil

from jw_signals.load_daq_data import plot_daq_data
from jw_signals.process_signals import get_fft
from jw_signals.plot_signals import plot_filt_from_sos
import jw_signals.inspect_daq_data_to_find_error_meanStd as inspect
from jw_utils.graph_utils import label_ax
    
#%% utility functions
def smooth_fft(fft_array, window_length = 51, polyorder = 3):
    return sig.savgol_filter(fft_array, window_length = window_length, polyorder = polyorder)

def parse_ratio(ratio):
    ratio_components = ratio.split('/')
    if len(ratio_components) != 2:
        raise ValueError("Invalid ratio (%s) passed to load_daq_data, should be x/y where x and y are columns in analyse_columns. Skipping ratio calculation" % ratio)
    return ratio_components
    
#%% declare class
class timeseries:
    data_time = {} # dict of dataframes
    data_fft = {} # dict of dataframes
    
    data_names = []
    
    read_columns = {}
    analyse_columns = {}
    noise_col_regex = None
    
    variance_fft = pd.DataFrame()
    variance_fft_smooth = pd.DataFrame()
    
    source_dir = '.'
    
    file_list = []
    labels = pd.DataFrame(columns = ['file_name_full', 'file_name', 'file_type']) # stores file names and information on them
    
    def __init__(self, read_columns, units, analyse_columns = [], ratios = [],
                 sr = None, f_filt = (None, None), 
                 window = {'type' : None, 'length_n' : None, 'start_n' : None, 'params' : [1]},
                 time_col = 'time', noise_col_regex = None, scale_cols = [], plot_col_ratios = [],
                 plots = [], quick_load = False,
                 source_dir = None, reg_str = None, file_name_parser = None):
        self.read_columns = read_columns
        if not set(read_columns).issubset(set(units.keys())):
            raise ValueError("Need units (after scaling) for every column read.")
        self.units = units
        self.analyse_columns = analyse_columns
        self.ratios = ratios
        self.sr = sr
        self.f_filt = f_filt
        self.window = window
        self.time_col = time_col
        self.noise_col_regex = noise_col_regex
        self.scale_cols = scale_cols
        self.plot_col_ratios = plot_col_ratios
        self.plots = plots
        if source_dir:
            self.source_dir = source_dir
            
        # make sure all in correct format
        if len(self.analyse_columns) == 0:
            self.analyse_columns = self.read_columns
        
        if len(self.scale_cols) == 0:
            self.scale_cols = [1] * len(self.analyse_columns)
        if len(self.scale_cols) != len(self.analyse_columns):
            raise ValueError("Invalid scale_cols passed to multipleDaqData, should be same length as analyse_columns, or empty list.")

        if self.time_col in self.analyse_columns:
            self.analyse_columns = [c for c in self.analyse_columns if c != self.time_col]
        
        # set up figure directory
        self.fig_dir = os.path.join(self.source_dir, 'figs')
        if not quick_load:
            if os.path.exists(self.fig_dir + '_old'):
                #check = input("figs_old already exists, delete? (Y/N)")
                #if check.lower() == 'y':
                print("Removing directory 'figs_old'...")
                shutil.rmtree(self.fig_dir + '_old')
                #else:
                #    raise RuntimeError("Delete/rename folder manually, then rerun.")
    
            if os.path.exists(self.fig_dir):
                print("Renaming directory 'figs' to 'figs_old'")
                os.rename(self.fig_dir, self.fig_dir + '_old')
            os.mkdir(self.fig_dir)
        elif not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
        
        # start loading
        self.get_file_list(self.source_dir, reg_str)
        self.get_file_labels(file_name_parser)
        print("loading data for:")
        print(self.labels)
        
        self.load_data(quick_load = quick_load)
        
        # processing
        
        if not quick_load:
            self.find_variance_all_col(group_by = 'test_num')
                
            _, _, _ = self.find_coherence(self.data_names)
                
    def get_file_list(self, source_dir, reg_str):
        file_list = []
        file_list = os.listdir(source_dir)
        file_list = [f for f in file_list if os.path.isfile(os.path.join(source_dir, f))]
        if reg_str:
            file_list = [f for f in file_list if re.search(reg_str, f)]
        self.file_list = file_list
    # ENDEF def get_file_list
        
    def get_file_labels(self, file_name_parser):
        if not file_name_parser:
            def file_name_parser(f):
                return f
        
        if len(self.file_list) == 0:
            raise ValueError("Cannot get file labels without file list.")

        self.labels['file_name_full'] = self.file_list
        self.labels['file_name'] = self.labels['file_name_full'].apply(lambda x: x[:x.rfind('.')])
        self.labels['file_type'] = self.labels['file_name_full'].apply(lambda x: x[x.rfind('.'):])
        
        self.labels['file_name'].apply(file_name_parser)
        
        self.labels = pd.concat([self.labels, self.labels['file_name'].apply(file_name_parser)], axis = 1)
        
        if self.noise_col_regex:
            self.labels['noise'] = self.labels['file_name_full'].apply(lambda x: True if re.search(self.noise_col_regex, x) else False)
    # ENDEF def get_file_labels
    
    #%%
    def load_data(self, plot_time = False, plot_fft = False, quick_load = False):
        # based on load_daq_data()
        if len(self.file_list) == 0:
            raise ValueError("Cannot load data without file list.")
        
        #decide which plots are being kept
        def check_plots(plots, name):
            if name in plots:
                return True
            return False
        plot_time_raw = check_plots(self.plots, 'time_raw')
        plot_time = check_plots(self.plots, 'time')
        plot_fft = check_plots(self.plots, 'fft')
        plot_psd = check_plots(self.plots, 'psd')
        plot_ratio = check_plots(self.plots, 'ratio')
        plot_filter = check_plots(self.plots, 'filter')
        
        # load each dataset
        for i, r in self.labels.iterrows():
            file = r['file_name_full']
            print(file)
            
            #%% load data
            if os.path.isdir(os.path.join(self.source_dir, file)):
                continue
            if 'title' in self.labels.columns:
                file_title = r['title']
            else:
                file_title = r['file_name']
            fig_save_title = file_title.replace(': ', '_')
            
            if r['file_type'] == '.txt':
                self.data_time[file_title] = pd.read_table(os.path.join(self.source_dir, file), header = None)
                if len(self.data_time[file_title].columns) != len(self.read_columns):
                    raise ValueError("Invalid read columns passed, need %d for file %s" % (len(self.data_time[file_title].columns), file))
                self.data_time[file_title].columns = self.read_columns
            elif r['file_type'] == '.tdms':
                tdms_file = TdmsFile(os.path.join(self.source_dir, file))

                self.data_time[file_title] = tdms_file['Untitled'].as_dataframe()
                
                self.data_time[file_title].columns = [c.lower().replace(' ', '_') for c in self.data_time[file_title].columns]
                
                if self.sr and not any([c.find(self.time_col) != -1 for c in self.data_time[file_title].columns]):
                    self.data_time[file_title][self.time_col] = np.arange(0, len(self.data_time[file_title]) * 1/self.sr, 1/self.sr)
                else:
                    raise ValueError("load_daq_data requires sr if file type = 'tdms' and no %s columns detected." % self.time_col)
                self.data_time[file_title] = self.data_time[file_title][[self.time_col] + [c for 
                                                        c in self.data_time[file_title].columns if c != self.time_col]]
            else:
                raise ValueError("File type %s not supported" % r['file_type'])
                
            if len(self.read_columns) > 0:
                self.data_time[file_title].columns = self.read_columns
            if len(self.analyse_columns) > 0:
                self.data_time[file_title] = self.data_time[file_title][[self.time_col] + self.analyse_columns]
            else:
                self.analyse_columns = self.data_time[file_title].columns
                    
            # apply scaling
            for i in range(len(self.analyse_columns)):
                self.data_time[file_title][self.analyse_columns[i]] *= self.scale_cols[i]
                
            self.data_names = list(self.data_time.keys())
        
            #%% plot against time
            if not quick_load:
                fig, _ = plot_daq_data(plot_type = 'time',
                              idx_col = self.time_col,
                              data = self.data_time[file_title],
                              plot_columns = self.analyse_columns,
                              plot_title = file_title)
                fig.tight_layout()
                fig.savefig(os.path.join(self.fig_dir, 'time_raw_' + fig_save_title + '.png'))
                if not plot_time_raw:
                    plt.close(fig)
                
            #%% filter, window
            # calculate sample rate
            dt = self.data_time[file_title].iloc[1,0]-self.data_time[file_title].iloc[0,0]
            sr = 1/dt
            if not self.sr:
                self.sr = sr
            elif self.sr != sr:
                raise Warning("Sample rate provided (%fHz) does not match sample rate from time column (%fHz)" % (self.sr, sr))
            
            # filter
            if self.f_filt[1]:    
                sos = sig.butter(4, self.f_filt, 'bandpass', fs = self.sr, output = 'sos')
            elif self.f_filt[0]:
                sos = sig.butter(3, self.f_filt[0], 'hp', fs = self.sr, output = 'sos')
            
            if self.f_filt[0] and not quick_load:
                fig, _ = plot_filt_from_sos(sos, sr = self.sr, 
                                            title = 'filter applied to raw self.data_time\nf = (%s, %s)' % self.f_filt)
                fig.savefig(os.path.join(self.fig_dir, 'filter_%s.jpg' % fig_save_title))
                if not plot_filter:
                    plt.close(fig)
                else:
                    plot_filter = False # only plot once
            
            for c in self.data_time[file_title].columns:
                if c != self.time_col:
                    if self.f_filt[0]:
                        self.data_time[file_title][c + '_filt'] = sig.sosfilt(sos, self.data_time[file_title][c])
                    else:
                        self.data_time[file_title][c + '_filt'] = self.data_time[file_title][c]
                    
            # window
            if not self.window['type']:
                w = []
            else:
                if not self.window['length_n']:
                    self.window['length_n'] = len(self.data_time[file_title])
                    
                if self.window['type'].lower() == 'hann':
                    w = sig.windows.get_window('hann', self.window['length_n'])
                elif self.window['type'].lower() == 'tukey':
                    w = sig.windows.tukey(self.window['length_n'], alpha = self.window['params'][0], sym = False)
                else:
                    raise ValueError("Invalid window, should be either hann or tukey.")
                    
                if self.window['start_n']:
                    w = np.append([0] * self.window['start_n'], w)
                    
                for c in self.data_time[file_title].columns:
                    if c != self.time_col:
                        self.data_time[file_title][c] = self.data_time[file_title][c] * w
                
            # plot_time_filtered:
            if not quick_load:
                fig, _ = plot_daq_data(plot_type = 'time',
                              idx_col = self.time_col,
                              data = self.data_time[file_title],
                              plot_columns = [c + '_filt' for c in self.analyse_columns],
                              plot_title = file_title)
                fig.tight_layout()
                fig.savefig(os.path.join(self.fig_dir, 'time_winfilt%s-%s_%s.jpg' % (str(self.f_filt[0]), 
                                                                                     str(self.f_filt[1]), fig_save_title)))
                if not plot_time:
                    plt.close(fig)
                
            #%% find fft
            self.data_fft[file_title] = pd.DataFrame()
            for c in self.analyse_columns:
                self.data_fft[file_title]['f'], self.data_fft[file_title][c] = get_fft(len(self.data_time[file_title]), 
                                                                         self.sr, 
                                                                         np.array(self.data_time[file_title][c]), 
                                                                         window = [], # already windowed
                                                                         shift = True,
                                                                         return_raw = True)
            
            # plot fft for both
            if not quick_load:
                fig, _ = plot_daq_data(plot_type = 'fft',
                              idx_col = 'f',
                              data = self.data_fft[file_title],
                              plot_columns = self.analyse_columns,
                              ignore_columns = ['f', self.time_col],
                              plot_title = file_title,
                              yscale = 'log')
                fig.tight_layout()
                fig.savefig(os.path.join(self.fig_dir, 'fft_%s.jpg' % fig_save_title))
                if not plot_fft:    
                    plt.close(fig)
                
            #%% plot psd
            if not quick_load:
                fig, _ = plot_daq_data(plot_type = 'psd',
                                          data = self.data_time[file_title],
                                          plot_columns = self.analyse_columns,
                                          ignore_columns = ['f', self.time_col],
                                          plot_title = file_title,
                                          yscale = 'log')
                fig.tight_layout()
                fig.savefig(os.path.join(self.fig_dir, 'psd_%s.jpg' % fig_save_title))
                if not plot_psd:    
                    plt.close(fig)
            
            #%% calculate and plot ratios
            for ratio in self.ratios:
                ratio_components = parse_ratio(ratio)
                if len(ratio_components) != 2:
                    continue
                elif (ratio_components[0] not in self.analyse_columns) or (ratio_components[1] not in self.analyse_columns):
                    raise Warning("Invalid ratio (%s) passed to load_daq_data, should be x/y where x and y are columns in analyse_columns. Skipping ratio calculation" % ratio)
                    continue
                else:
                    self.data_fft[file_title][ratio] = self.data_fft[file_title][ratio_components[0]]/self.data_fft[file_title][ratio_components[1]]
                
                if not quick_load:
                    fig, ax = plot_daq_data(plot_type = 'fft',
                              data = self.data_fft[file_title],
                              plot_columns = [ratio],
                              ignore_columns = ['f', self.time_col],
                              plot_title = file_title + '\nratio: %s' % ratio)
                    ax[0].set_ylabel('abs(fft(%s))/abs(fft(%s))' % (ratio_components[0], ratio_components[1]))
                    ax[1].set_ylabel('phase(fft(%s))/phase(fft(%s))' % (ratio_components[0], ratio_components[1]))
                    fig.tight_layout()
                    fig.savefig(os.path.join(self.fig_dir, 'ratio_fft_%s_%s.jpg' % (ratio.replace('/', '-'), fig_save_title)))
                    if not plot_ratio:
                        plt.close(fig)
                    
    # ENDEF def load_data
    
    def gen_variance_fft_colname(self, var_type, group_by, n, col):
        # generates column names in variance_fft
        return '%s_%s_%s_%s' % (group_by, str(n), col, var_type)
    
    def find_variance_all_col(self, group_by = 'test_num', 
                      savgol_win_len = 101, close_plot = True,
                      x_lim = (None, None), quick_load = False):
        for c in self.analyse_columns:
            _, _, _ = self.find_variance(group_by, col = c,
                              savgol_win_len = savgol_win_len, close_plot = close_plot,
                              x_lim = x_lim, quick_load = quick_load)

    
    def find_variance(self, group_by = 'test_num', col = 'A0', 
                      savgol_win_len = 101, close_plot = True,
                      x_lim = (None, None), quick_load = False):
        # find variance between repeats
        
        for n in np.unique(self.labels[group_by]):
            keys = list(self.labels.loc[self.labels[group_by] == n, 'title'])             
            
            data_cols = list(self.labels.loc[(self.labels[group_by] == n) & (self.labels['noise'] != True), 'title']) 
            
            noise_cols = list(self.labels.loc[(self.labels[group_by] == n) & (self.labels['noise'] == True), 'title']) 
            data_cols = [c + ': ' + col for c in data_cols]
            noise_cols = [c + ': ' + col for c in noise_cols]
            ref_col = 'mean' # data_cols[0]
            
            comb_df = inspect.combine_dict_to_df({k : v for k, v in self.data_fft.items() if k in keys}, cols = ['f', col])
            
            def process_variance(comb_df):
                comb_df['mean'] = comb_df[data_cols].mean(axis = 1)
                # comb_df['stdev'] = comb_df[data_cols].std(axis = 1)
                
                # find difference to a reference
                diff_df = inspect.find_diff_to_ref(comb_df, noise_cols, ref_col, data_cols)
    
                ## find stdev of difference between each column and reference
                diff_df_data_cols = [c + ': diff to ref' for c in data_cols]
                diff_df.columns = ['f'] + diff_df_data_cols
                diff_df['stdev'] = diff_df[diff_df_data_cols].std(axis = 1)
    
                comb_df = comb_df.merge(diff_df, on = 'f', validate = 'one_to_one')
                
                # use this to find min and max
                comb_df['min'] = comb_df['mean'] * (1 - comb_df['stdev'])
                comb_df['max'] = comb_df['mean'] * (1 + comb_df['stdev'])
                
                return comb_df
            
            def plot_variance(comb_df, close_plot, fig_label = ''):
                if quick_load:
                    return None
                else:
                    fig, axs = plt.subplots(2,1, sharex = True)
                    plot_cmap_colors = cm.get_cmap('viridis', len(data_cols)).colors
                    for i in range(len(data_cols)):
                        axs[0].plot(comb_df['f'], np.abs(comb_df[data_cols[i]]), 
                                    c = plot_cmap_colors[i], alpha = 0.5,
                                    label = data_cols[i])
                        axs[1].plot(comb_df['f'], np.real(comb_df[data_cols[i] + ': diff to ref']), 
                                    c = plot_cmap_colors[i], alpha = 0.8,
                                    label = data_cols[i])
                    axs[0].plot(comb_df['f'], np.abs(comb_df[ref_col]), c = 'b', label = ref_col)
                    
                    axs[0].set_xlim(x_lim)
                    
                    # context lines
                    xl=axs[1].get_xlim()
                    axs[1].plot(xl, [0.9,0.9], c = 'k', ls = '--', lw = 0.5)
                    axs[1].plot(xl, [1.1,1.1], c = 'k', ls = '--', lw = 0.5)
                    axs[1].plot(xl, [1, 1], c = 'k', ls = '-', lw = 0.5)
                    axs[1].set_xlim(xl)
                    
                    label_ax(axs[0], 'Comparing test %s\n%s %s' % (str(n), col, fig_label), '', 
                             'Fourier transform (%s)' % self.units[col], legend = True)
                    label_ax(axs[1], '', 'Frequency (Hz)', 'Ratio of test to\nreference fourier transform (%s)' % self.units[col],
                             legend = True)
                    
                    if len(fig_label) > 0:
                        fig_label = '_' + fig_label
                    fig.savefig(os.path.join(self.fig_dir, 'variance%s_test%s_%s' % (fig_label, str(n), col)))
                    if close_plot:
                        plt.close(fig)
                        return None
                    else:
                        return [fig, axs]
                
            
            comb_df = process_variance(comb_df)
            fig = plot_variance(comb_df, close_plot = close_plot)
            
            comb_df_smooth = comb_df[['f'] + data_cols].copy()
            for c in data_cols:
                comb_df_smooth[c] = smooth_fft(abs(comb_df_smooth[c]), window_length = savgol_win_len)
                
            comb_df_smooth = process_variance(comb_df_smooth)
            fig_smooth = plot_variance(comb_df_smooth, fig_label = 'smooth', close_plot = close_plot)
            
            if len(self.variance_fft) == 0:
                self.variance_fft = comb_df[['f', 'mean', 'stdev']]
                self.variance_fft.columns = ['f', '%s_%s_mean' % (group_by, str(n)), '%s_%s_stdev' % (group_by, str(n))]
            else:
                temp_df = comb_df[['f', 'mean', 'stdev']]
                temp_df.columns = ['f', self.gen_variance_fft_colname('mean', group_by, n, col), 
                                   self.gen_variance_fft_colname('stdev', group_by, n, col)]
                self.variance_fft = pd.merge(self.variance_fft, temp_df, how = 'outer', on = 'f', validate = 'one_to_one')
                
            if len(self.variance_fft_smooth) == 0:
                self.variance_fft_smooth = comb_df_smooth[['f', 'mean', 'stdev']]
                self.variance_fft_smooth.columns = ['f', self.gen_variance_fft_colname('mean', group_by, n, col), 
                                                    self.gen_variance_fft_colname('stdev', group_by, n, col)]
            else:
                temp_df = comb_df_smooth[['f', 'mean', 'stdev']]
                temp_df.columns = ['f', self.gen_variance_fft_colname('mean', group_by, n, col), 
                                   self.gen_variance_fft_colname('stdev', group_by, n, col)]
                self.variance_fft_smooth = pd.merge(self.variance_fft_smooth, temp_df, how = 'outer', on = 'f', validate = 'one_to_one')
        
        def plot_variance_comparison(variance_fft, close_plot, fig_label = ''):
            if quick_load:
                return
            else:
                fig, ax = plt.subplots(1,1)
                i = -1
                for n in np.unique(self.labels[group_by]):
                    i += 1
                    ax.plot(variance_fft['f'], np.abs(variance_fft[self.gen_variance_fft_colname('mean', group_by, n, col)]),
                            c = 'C%d' % i, label = '%s_%s' % (group_by, str(n)), alpha = 0.8)
                    ax.plot(variance_fft['f'], np.abs(variance_fft[self.gen_variance_fft_colname('mean', group_by, n, col)] * (1 - variance_fft[self.gen_variance_fft_colname('stdev', group_by, n, col)])),
                            c = 'C%d' % i, ls = '--', lw = 0.5, alpha = 0.8)
                    ax.plot(variance_fft['f'], np.abs(variance_fft[self.gen_variance_fft_colname('mean', group_by, n, col)] * (1 + variance_fft[self.gen_variance_fft_colname('stdev', group_by, n, col)])),
                            c = 'C%d' % i, ls = '--', lw = 0.5, alpha = 0.8)
                ax.set_xlim(x_lim)
                label_ax(ax, 'Comparison of mean and variance of each test\n%s' % fig_label, 'frequency (Hz)',
                         '%s fourier transform (%s)' % (col, self.units[col]))
                if len(fig_label) > 0:
                    fig_label = '_' + fig_label
                fig.savefig(os.path.join(self.fig_dir, 'variance_comparison%s_%s' % (fig_label, col)))
                if close_plot:
                    plt.close(fig)
                
        plot_variance_comparison(self.variance_fft, close_plot)
        plot_variance_comparison(self.variance_fft_smooth, close_plot, fig_label = 'smooth')
            
        return comb_df, fig, fig_smooth
    #ENDEF def find_variance
    
    def plot(self, dataset,
             plot_type = 'fft',
             idx_col = 'default',
             plot_columns = [],
             ignore_columns = [],
             plot_title = "",
             yscale = 'log',
             plot_max_f = None,
             ratio = None,
             y_label = None,
             x_label = None):
        if len(plot_columns) == 0:
            plot_columns = self.analyse_columns
            
        plot_daq_data(plot_type = plot_type,
                          idx_col = idx_col,
                          data = self.data_time[dataset] if plot_type != 'fft' else self.data_fft[dataset],
                          plot_columns = plot_columns,
                          ignore_columns = ignore_columns,
                          plot_title = plot_title,
                          yscale = yscale,
                          plot_max_f = plot_max_f,
                          ratio = ratio,
                          y_label = y_label,
                          x_label = x_label)
    
    def find_coherence(self, datasets, x_lim = (None, None), close_plot = True):
        coh = {}
        figs = {}
        for i in range(len(self.analyse_columns)):
            c1 = self.analyse_columns[i]
            for j in range(i + 1, len(self.analyse_columns)):
                c2 = self.analyse_columns[j]
                
                fig, ax = plt.subplots(1,1)
                
                for d in datasets:
                    clbl = '%s_%s_%s' % (d, c1, c2)
                    
                    f_coh, coh[clbl] = sig.coherence(self.data_time[d][c1], 
                                                                 self.data_time[d][c2],
                                                                 fs = self.sr)
                    ax.plot(f_coh, coh[clbl], label = clbl)
                
                ax.set_xlim(x_lim)
                
                #ax.plot(f_coh, coh['II'], label = 'II')
                label_ax(ax, '', 'Frequency [Hz]', 'Coherence', ylim = (0,1), legend_title = 'axes: ')
                
                fig.savefig(os.path.join(self.fig_dir, 'coherence_%s_%s' % (c1, c2)))
                if close_plot:
                    plt.close(fig)
                    figs[clbl] = None
                else:
                    figs[clbl] = fig
        return f_coh, coh, figs
    
    def plot_mean_fft(self, grouped_by, groups_to_plot = [], smoothed = False, 
                      labels = [], x_lim = (None, None), close_plot = True):
        if smoothed:
            variance_fft = self.variance_fft_smooth
        else:
            variance_fft = self.variance_fft
        
        # plots mean for a subset of datasets
        if len(variance_fft) == 0:
            raise ValueError("Cannot plot mean fft before running find_variance().")
            
        if len(groups_to_plot) == 0:
            groups_to_plot = np.unique(self.labels[grouped_by])
        if len(labels) == 0:
            labels = groups_to_plot
        if len(labels) != len(groups_to_plot):
            raise ValueError("labels needs to be same length as datasets")

        fig, axs = plt.subplots(len(self.analyse_columns),1,sharex = True)

        i = -1
        for c in self.analyse_columns:
             i += 1
             j = -1
             for n in groups_to_plot:
                 j += 1
                 
                 axs[i].plot(variance_fft['f'], 
                             np.abs(variance_fft[self.gen_variance_fft_colname('mean', grouped_by, n, c)]), 
                             label = labels[j])
        
        for i in range(1, len(self.analyse_columns)):
            c = self.analyse_columns[i]
            label_ax(axs[i], title = '', xlabel = '', ylabel = '%s (%s)' % (c, self.units[c]),
                      legend = False)
             
        c = self.analyse_columns[0]
        label_ax(axs[0], title = 'comparison of mean fft with %s' % grouped_by, 
                 xlabel = '', ylabel = '%s (%s)' % (c, self.units[c]), legend = True)
        
        axs[-1].set_xlabel('frequency (Hz)')

        axs[0].set_xlim(x_lim)
        
        fig.savefig(os.path.join(self.fig_dir, 'mean_fft_comparison_%s' % grouped_by))
        
        if close_plot:
            plt.close('all')
            return None
        else:
            return [fig, axs]
    
    def plot_mean_fft_ratio(self, ratio, grouped_by, groups_to_plot = [], smoothed = False, 
                      labels = [], x_lim = (None, None), close_plot = True):
        
        ratio_ans = {}
        
        if smoothed:
            variance_fft = self.variance_fft_smooth
        else:
            variance_fft = self.variance_fft
        
        # plots mean for a subset of datasets
        if len(variance_fft) == 0:
            raise ValueError("Cannot plot mean fft before running find_variance().")
            
        if len(groups_to_plot) == 0:
            groups_to_plot = np.unique(self.labels[grouped_by])
        if len(labels) == 0:
            labels = groups_to_plot
        if len(labels) != len(groups_to_plot):
            raise ValueError("labels needs to be same length as datasets")
            
        ratio_components = parse_ratio(ratio)

        fig, axs = plt.subplots(1,1,sharex = True)

        ratio_ans['f'] = variance_fft['f']
        j = -1
        for n in groups_to_plot:
            j += 1
            ratio_ans[n] = np.abs(variance_fft[self.gen_variance_fft_colname('mean', grouped_by, n, ratio_components[0])]) /  np.abs(variance_fft[self.gen_variance_fft_colname('mean', grouped_by, n, ratio_components[1])])
            axs.plot(variance_fft['f'], ratio_ans[n], 
                        label = labels[j])
            
             
        label_ax(axs, title = 'comparison of mean fft with %s' % grouped_by, 
                 xlabel = 'frequency (Hz)', ylabel = '%s (%s/%s)' % (ratio, self.units[ratio_components[0]], 
                                                                     self.units[ratio_components[1]]), 
                 legend = True, xlim = x_lim)
        
        fig.savefig(os.path.join(self.fig_dir, 'mean_fft_comparison_%s_%s' % (grouped_by, ratio.replace('/', '-'))))
        
        if close_plot:
            plt.close('all')
            return None
        else:
            return [pd.DataFrame(ratio_ans, columns = ['f'] + groups_to_plot), fig, axs]
        
    def find_impulse_response(self):
        ir = pd.DataFrame()
        
    def find_frf(self):
        frf = pd.DataFrame()
        
    def find_cross_correlation():
        xcorr = []
                
            
            
            


        
            

    
    
    
    
    
    
    
