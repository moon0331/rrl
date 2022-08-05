import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-d', '--data_set', type=str, default='adult', # 'tic-tac-toe' 'baseball_binary'
parser.add_argument('-d', '--data_set', type=str, default='baseball', # 'tic-tac-toe' 'baseball_binary', 'baseball'
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-i', '--device_ids', type=str, default='0', help='Set the device (GPU ids). Split by @.'
                                                                       ' E.g., 0@2@3.') # None
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('-e', '--epoch', type=int, default=10, help='Set the total epoch.') # 41 400
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Set the batch size.') # 64
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Set the initial learning rate.')
parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='Set the learning rate decay rate.')
parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=100, help='Set the learning rate decay epoch.') # 10
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Set the weight decay (L2 penalty).') # 0
parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th 5-fold validation, 0 <= ki < 5.')
parser.add_argument('-rc', '--round_count', type=int, default=0, help='Count the round of experiments.')
parser.add_argument('-ma', '--master_address', type=str, default='127.0.0.1', help='Set the master address.')
parser.add_argument('-mp', '--master_port', type=str, default='12345', help='Set the master port.')
parser.add_argument('-li', '--log_iter', type=int, default=50, help='The number of iterations (batches) to log once.')
parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold value used in binarized_forward')
parser.add_argument('-r', '--range', type=float, default=3.0, help='range for binarization layer')

parser.add_argument('--use_not', action="store_true",
                    help='Use the NOT (~) operator in logical rules. '
                         'It will enhance model capability but make the RRL more complex.')
parser.add_argument('--save_best', action="store_true",
                    help='Save the model with best performance on the validation set.')
parser.add_argument('--estimated_grad', action="store_true",
                    help='Use estimated gradient.')
parser.add_argument('-s', '--structure', type=str, default='5@100', # '5@64@32'
                    help='Set the number of nodes in the binarization layer and logical layers. '
                         'E.g., 10@64, 10@64@32@16.')

rrl_args = parser.parse_args()
# rrl_args.folder_name = '{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_ki{}_rc{}_useNOT{}_saveBest{}_estimatedGrad{}'.format(
rrl_args.folder_name = '{}/e{}_bs{}_lr{}_lrdr{}_lrde{}/wd{}_ki{}_rc{}/useNOT{}/saveBest{}/estimatedGrad{}/L{}/threshold{}/range{}'.format(
# rrl_args.folder_name = '{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_ki{}_rc{}_useNOT{}_saveBest{}_estimatedGrad{}______'.format(
    rrl_args.data_set, rrl_args.epoch, rrl_args.batch_size, rrl_args.learning_rate, rrl_args.lr_decay_rate,
    rrl_args.lr_decay_epoch, rrl_args.weight_decay, rrl_args.ith_kfold, rrl_args.round_count, rrl_args.use_not,
    rrl_args.save_best, rrl_args.estimated_grad, 
    rrl_args.structure, rrl_args.threshold, rrl_args.range
)

if not os.path.exists('log_folder'):
    os.mkdir('log_folder')
# rrl_args.folder_name = rrl_args.folder_name + '_L' + rrl_args.structure
# rrl_args.folder_name += f'_threshold{rrl_args.threshold}_range{rrl_args.range}'
rrl_args.set_folder_path = os.path.join('log_folder', rrl_args.data_set)
if not os.path.exists(rrl_args.set_folder_path):
    os.mkdir(rrl_args.set_folder_path)
rrl_args.folder_path = os.path.join(rrl_args.set_folder_path, rrl_args.folder_name)
print(rrl_args.folder_path)
if not os.path.exists(rrl_args.folder_path):
    # os.mkdir(rrl_args.folder_path)
    os.makedirs(rrl_args.folder_path, exist_ok=True)
rrl_args.model = os.path.join(rrl_args.folder_path, 'model.pth')
rrl_args.rrl_file = os.path.join(rrl_args.folder_path, 'rrl.csv') #txt to csv
rrl_args.plot_file = os.path.join(rrl_args.folder_path, 'plot_file.pdf')
rrl_args.log = os.path.join(rrl_args.folder_path, 'log.txt')
rrl_args.test_res = os.path.join(rrl_args.folder_path, 'test_res.txt')
rrl_args.device_ids = list(map(int, rrl_args.device_ids.strip().split('@')))
rrl_args.gpus = len(rrl_args.device_ids)
rrl_args.nodes = 1
rrl_args.world_size = rrl_args.gpus * rrl_args.nodes
rrl_args.batch_size = int(rrl_args.batch_size / rrl_args.gpus)
