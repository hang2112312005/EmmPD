import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
import argparse
import numpy as np
from modules.dataloaders import R2DataLoader
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from models.r2gen import R2GenModel
import warnings
warnings.filterwarnings("ignore")


import random

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--base_dir', type=str,
                        default='./Dataset/cameylon+/',
                        help='the path to the directory containing the encoded wsi patches.')
    parser.add_argument('--image_dir', type=str,
                        default='./camlyon+/h5_files',
                        help='the path to the directory containing the encoded wsi patches.')
    parser.add_argument('--ann_path', type=str,
                        default='./cameylon+/h5py-files',
                        help='the path to the directory containing the data.')
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='camelyon', choices=['TCGA',], help='the dataset to be used.')
    parser.add_argument('--n_classes', type=int, default=4, help='how many classes to predict')
    parser.add_argument('--k_sample', type=int, default=4000, help='the number of samples for a batch.')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples for a batch.')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=768, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=1024, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=3, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=1, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=str, default='2', help='the  gpus to be used.')
    parser.add_argument('--epochs', type=int, default=80, help='the number of training epochs.')
    parser.add_argument('--epochs_val', type=int, default=2, help='interval between eval epochs')
    parser.add_argument('--start_val', type=int, default=2, help='start eval epochs')
    parser.add_argument('--save_dir', type=str, default='results/camelyon_4000', help='the patch to save the models.')
    parser.add_argument('--save_period', type=int, default=2, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='ROCAUC', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=10, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')
    
    # debug
    parser.add_argument("--checkpoint_dir", type=str, default='results/')
    parser.add_argument("--mode", type=str, default='Train')
    parser.add_argument("--debug", type=str, default='False')
    parser.add_argument("--local_rank", type=int, default=-1)

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '30300'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
def main(local_rank, world_size):
    args = parse_agrs()

    # scaling learning rate
    args.lr_ed *= world_size
    
    setup(local_rank, world_size)
    if not args.debug:
        torch.cuda.set_device(local_rank)

    # fix random seeds
    init_seeds(args.seed+local_rank)

    # create data loader
    train_dataloader = R2DataLoader(args, split='train', shuffle=False)
    val_dataloader = R2DataLoader(args, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args).to(local_rank)
    #current_checkpoint.pth
    if args.mode == 'Test':
        resume_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)['state_dict']
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in checkpoint.items()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)


    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # build optimizer, learning rate scheduler. set after DDP.
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    
    checkpoint_dir = args.save_dir
    if not os.path.exists(checkpoint_dir):
        if local_rank == 0:
            os.makedirs(checkpoint_dir)
            
    if args.mode == 'Train':
        trainer.train(local_rank)
    else:
        trainer.test(local_rank)

    if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_agrs()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.n_gpu
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus


    if args.debug:
        assert n_gpus==1
        main(0, 1)
    else:
        mp.spawn(main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
