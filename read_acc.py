import glob
from tqdm import tqdm
import numpy as np

'''
epoch
bs
lr
lrdr
lrde
wd
ki0_rc0
useNot
saveBest
estimatedGrad
structure
threshold
range
acc, f1
'''

from parse import compile
import pandas as pd

p = compile('e{e:d}_bs{bs:d}_lr{lr:f}_lrdr{lrdr:f}_lrde{lrde:d}|wd{wd:g}_ki0_rc0|useNOT{useNOT}|saveBest{saveBest}|estimatedGrad{estimatedGrad}|{structure}|threshold{threshold:f}|range{range:f}|acc={acc:f},f1={f1:f}')
print(p)

list_of_results = []

for acc_f1_file in tqdm(glob.glob('log_folder/baseball/baseball/e*/*/*/*/*/*/*/*/acc*')):
    file_string = '|'.join(acc_f1_file.split('/')[3:])
    parsed_result = p.parse(file_string).named
    # print(parsed_result)
    parsed_result['useNOT'] = True if parsed_result['useNOT'] == 'True' else False
    parsed_result['saveBest'] = True if parsed_result['saveBest'] == 'True' else False
    parsed_result['estimatedGrad'] = True if parsed_result['estimatedGrad'] == 'True' else False
    rrl_file = '/'.join(acc_f1_file.split('/')[:-1] + ['rrl.csv'])
    with open(rrl_file) as f:
        rrl_file = pd.read_csv(f, sep='\t')[:-1]
        if not rrl_file.empty:
            # breakpoint()
            rule_df = rrl_file['Rule'].str.count(r'>|<')
            parsed_result['num_rule'] = rule_df.size
            parsed_result['avg_rule'] = rule_df.mean().round(2)
            parsed_result['std_rule'] = rule_df.std().round(2)
            parsed_result['max_rule'] = rule_df.max()
        else:
            parsed_result['num_rule'] = 0
            parsed_result['avg_rule'] = np.nan
            parsed_result['std_rule'] = np.nan
            parsed_result['max_rule'] = 0
    list_of_results.append(parsed_result)

all_results = pd.DataFrame(list_of_results)
print(all_results)
print(all_results.dtypes)

if True:
    all_results.to_csv('log_folder/baseball/baseball/all_results_08101619.csv', index=False)