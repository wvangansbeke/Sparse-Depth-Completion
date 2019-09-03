#!/usr/bin/sh
# source Shell/train.sh $data_path
model='mod'
optimizer='adam'
data_path=${1}
batch_size=${2-7}
lr=${3-0.001}
lr_policy='plateau'
nepochs=60
patience=5
wrgb=${4-0.1}
nsamples=${5-0}
multi=${6-0}
out_dir='Saved'

export OMP_NUM_THREADS=1
python main.py --mod $model --data_path $data_path --optimizer $optimizer --learning_rate $lr --lr_policy $lr_policy --batch_size $batch_size --nepochs $nepochs --no_tb true --lr_decay_iters $patience --num_samples $nsamples --multi $multi --nworkers 4 --save_path $out_dir --wrgb $wrgb 

echo "python has finisched its "$nepochs" epochs!"
echo "Job finished"
