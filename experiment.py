import warnings

from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

import os
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'


def get_data_loader(dataset, batch_size, k=0, pin_memory=False, save_best=True, valid_ratio=0.95):
    # training scheme 비교 필요
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[k]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    train_len = int(len(train_set) * valid_ratio)
    train_sub, valid_sub = random_split(train_set, [train_len, len(train_set) - train_len])
    if not save_best:  # use all the training set for training, and no validation set used for model selections.
        train_sub = train_set

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_sub, num_replicas=world_size, rank=rank)

    # train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=train_sampler)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, test_loader


def get_baseball_data_loader(dataset, batch_size, pin_memory=False):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=False, is_baseball=True) # include GMKEY

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df.drop(columns='GMKEY'), y_df)

    X, y = db_enc.transform(X_df.drop(columns='GMKEY'), y_df, normalized=True, keep_stat=True) # not include GMKEY

    _train_year_range = tuple(map(str, range(2013, 2019+1))) # ('2013', '2014', '2015', '2016', '2017', '2018', '2019')
    X_train = X[X_df.GMKEY.str.startswith(_train_year_range)]
    y_train = y[X_df.GMKEY.str.startswith(_train_year_range)]

    X_valid = X[X_df.GMKEY.str.startswith('2020')]
    y_valid = y[X_df.GMKEY.str.startswith('2020')]

    X_test = X[X_df.GMKEY.str.startswith('2021')] # testset의 정의
    y_test = y[X_df.GMKEY.str.startswith('2021')]

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
    valid_set = TensorDataset(torch.tensor(X_valid.astype(np.float32)), torch.tensor(y_valid.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, test_loader

def get_test_loader(dataset, batch_size, pin_memory=False, save_best=True):
    data_path = os.path.join(DATA_DIR, dataset + '_test.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=False)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    X_test = X
    y_test = y

    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return test_loader


def train_model(gpu, args):
    rank = args.nr * args.gpus + gpu
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(42)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)

    # if gpu == 0:
    #     writer = SummaryWriter(args.folder_path)
    #     is_rank0 = True
    # else:
    #     writer = None
    #     is_rank0 = False

    writer = SummaryWriter(args.folder_path)
    # is_rank0 = True

    dataset = args.data_set
    if args.data_set == 'baseball':
        db_enc, train_loader, valid_loader, _ = get_baseball_data_loader(dataset, args.batch_size, pin_memory=True)
    else:
        db_enc, train_loader, valid_loader, _ = get_data_loader(dataset, args.batch_size, k=args.ith_kfold, 
                                                                pin_memory=True, save_best=args.save_best, valid_ratio=0.9)

    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen

    rrl = RRL(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
              device_id=device_id,
              use_not=args.use_not,
              is_rank0=True,
              log_file=args.log,
              writer=writer,
              save_best=args.save_best,
              estimated_grad=args.estimated_grad,
              save_path=args.model,
              left=-args.range,
              right=args.range
              )

    rrl.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter)


def load_model(path, device_id, log_file=None, left=None, right=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        dim_list=saved_args['dim_list'],
        device_id=device_id,
        is_rank0=True,
        use_not=saved_args['use_not'],
        log_file=log_file,
        left=left,
        right=right,
        distributed=distributed,
        estimated_grad=saved_args['estimated_grad']
    )
    stat_dict = checkpoint['model_state_dict']
    # for key in list(stat_dict.keys()):
    #     # remove 'module.' prefix
    #     stat_dict[key[7:]] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


def test_model(args, test_model=None):
    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, left=-args.range, right=args.range, distributed=False)
    dataset = args.data_set
    if args.data_set == 'baseball':
        db_enc, train_loader, valid_loader, test_loader = get_baseball_data_loader(dataset, args.batch_size, pin_memory=True)
    else:
        pass
        # db_enc, train_loader, valid_loader, _ = get_data_loader(dataset, args.batch_size, k=args.ith_kfold, 
        #                                                     pin_memory=True, save_best=args.save_best, valid_ratio=0.9)

        db_enc, train_loader, _, test_loader = get_data_loader(dataset, args.batch_size, args.ith_kfold, save_best=False, valid_ratio=0.9)

        # test_loader = get_test_loader(dataset, args.batch_size) # redefine
    
    rrl.test(test_loader=test_loader, set_name='Test', fname=args.rrl_file.replace('/rrl.csv', ''))
    with open(args.rrl_file, 'w', encoding='utf-8-sig') as rrl_file:
        rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std)


def train_main(args):
    # os.environ['MASTER_ADDR'] = args.master_address
    # os.environ['MASTER_PORT'] = args.master_port
    # mp.spawn(train_model, nprocs=args.gpus, args=(args,))
    train_model(0, args)


if __name__ == '__main__':
    from args import rrl_args
    # for arg in vars(rrl_args):
    #     print(arg, getattr(rrl_args, arg))
    print('train', rrl_args.data_set, rrl_args.epoch, rrl_args.structure)
    train_main(rrl_args)
    test_model(rrl_args)