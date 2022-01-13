import os,shutil
import sys
import argparse
import random
import time
import torch

def backup(dirname):
    os.makedirs(dirname, exist_ok=True)
    import shutil
    dirname = os.path.join(dirname,'backup-{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    shutil.copytree('./modules', os.path.join(dirname, 'modules'))
    shutil.copy('./model.py', dirname)
    shutil.copy('./dataset.py', dirname)
    shutil.copy('./utils.py', dirname)
    shutil.copy('./test.py', dirname)
    shutil.copy('./train_ft.py', dirname)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("RunTrain")
    # parser.add_argument('--train_data', type=str, default='/mnt/local0/data_lmdb_release/training')
    # parser.add_argument('--valid_data', type=str, default='/mnt/local0/data_lmdb_release/validation')
    # parser.add_argument('--eval_data', type=str, default='/mnt/local0/data_lmdb_release/evaluation')
    parser.add_argument('--train_data', type=str, default='data_lmdb_release/training')
    parser.add_argument('--valid_data', type=str, default='data_lmdb_release/validation')
    parser.add_argument('--eval_data', type=str, default='data_lmdb_release/evaluation')
    parser.add_argument('--select_data', type=str, default='MJ-ST')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5')
    parser.add_argument('--Transformation', type=str, default='None')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='None')
    parser.add_argument('--Prediction', type=str, default='CTC')
    parser.add_argument('--seed', type=int, default='0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--pad', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')

    parser.add_argument('--hidden_size', type=int, default=512, help='the size of the LSTM hidden state')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--milestones', type=str, default='0.7', help='milestones of learning rate decay e.g. 0.7_0.9')
    parser.add_argument('--distributed', action='store_true', default=False, help='distribute')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--scheduler', type=str, default='Cosine', help='scheduler')
    parser.add_argument('--warmup_a', type=float, default=0.02, help='warmup_a')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='/data/chuxiaojie/projects/STR/deep-text-recognition-benchmark/result/TPSHD-ResNet-BiLSTM-AttnGRU-MJ-S-real_0.484375-0.484375-0.03125_512-0.001Cosine0.02-Seed0_best_accuracy.pth/best_accuracy.pth', help="path to model to continue training")
    args = parser.parse_args()

    experiment_name = 'FT-{}-{}-{}-{}-{}_{}_{}-{}{}{}-Seed{}'.format(\
        args.Transformation,args.FeatureExtraction,args.SequenceModeling,args.Prediction,\
            args.select_data, args.batch_ratio, args.hidden_size, \
                   args.lr, args.scheduler,args.warmup_a, args.seed)
    # args.saved_model='saved_models/{}/best_accuracy.pth'.format(experiment_name)
    

    if args.train:
        if not args.force:
            assert not os.path.exists('saved_models/{}'.format(experiment_name))
        instr = 'train_ft.py --train_data {} --valid_data {} --eval_data {} --select_data {} --batch_ratio {} --Transformation {} --FeatureExtraction {} --SequenceModeling {} --Prediction {} --manualSeed {} --lr {} --batch_size {} --imgH {} --imgW {} --num_iter {}  --milestones {} --hidden_size {} --experiment_name {} --scheduler {} --warmup_a {} --output_channel {} --valInterval {}'.format(
            args.train_data,args.valid_data,args.eval_data, args.select_data,args.batch_ratio,args.Transformation,args.FeatureExtraction,args.SequenceModeling,args.Prediction,int(args.seed%(1e9+7)),args.lr,args.batch_size,args.imgH,args.imgW,args.num_iter, args.milestones, args.hidden_size, experiment_name, args.scheduler, args.warmup_a, args.output_channel, args.valInterval)
        if args.pad:
            instr += ' --PAD'
        if args.sensitive:
            instr += ' --sensitive'
        if args.rgb:
            instr += ' --rgb'
        instr += ' --alphanumeric'
        instr += ' --adam'
        instr += ' --FT'
        assert os.path.isfile(args.saved_model)
        instr += ' --saved_model {}'.format(args.saved_model)    
        backup(dirname=f'./saved_models/{experiment_name}')

        num_gpu = torch.cuda.device_count()
        if args.distributed:
            assert num_gpu>1
            instr = 'python3 -m torch.distributed.launch --nproc_per_node={} {}'.format(num_gpu, instr)
        else:
            assert num_gpu == 1
            instr = 'python3 {}'.format(instr)
        print(instr)
        os.system(instr)
    elif args.test:
        args.saved_model='saved_models/{}/best_eval_accuracy.pth'.format(experiment_name)
        instr = 'python3 test.py --eval_data {} --benchmark_all_eval --Transformation {} --FeatureExtraction {} --SequenceModeling {} --Prediction {} --saved_model {} --imgH {} --imgW {} --hidden_size {} --output_channel {}'.format(
            args.eval_data,args.Transformation,args.FeatureExtraction,args.SequenceModeling,args.Prediction,args.saved_model, args.imgH,args.imgW, args.hidden_size,  args.output_channel)

        instr += ' --data_filtering_off'
        if args.sensitive:
            instr += ' --sensitive'
        if args.pad:
            instr += ' --PAD'
        if args.rgb:
            instr += ' --rgb'
        instr += ' --alphanumeric'
        print(instr)
        os.system(instr)

