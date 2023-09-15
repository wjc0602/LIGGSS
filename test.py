import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/data/comptition/jittor3/Jittor1_test_a')
parser.add_argument('--img_path', type=str, default='/data/comptition/jittor3/Jittor1_train_resized/imgs')
parser.add_argument('--output_path', type=str, default='./results/')
parser.add_argument('--which_epoch', type=str, default='latest')
parser.add_argument('--exp_name', type=str, default='baseline')
args = parser.parse_args()
cmd_str = f'python spade_test.py --name {args.exp_name} --dataset_mode custom --label_dir {args.input_path} --image_dir {args.img_path} --results_dir {args.output_path} --label_nc 29 --no_instance True --use_vae True --which_epoch {args.which_epoch} --no_pairing_check'
subprocess.call(cmd_str, shell=True)
