#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONIOENCODING=utf-8

dataset="MNIST"
output_dir="../../results_pdf/"$dataset"/"

time_stamp=`date '+%s'`

mkdir -p $output_dir"board/"
mkdir -p $output_dir"error/"
mkdir -p $output_dir"stdout/"
mkdir -p $output_dir"stat/"

config_file="./"$dataset".json"

_commit_id=`git rev-parse HEAD`
commit_id=${_commit_id:0:7}

out_file=${output_dir}"stdout/"${time_stamp}_${commit_id}".out"
err_file=${output_dir}"error/"${time_stamp}_${commit_id}".err"

nohup python -u ./main.py --config=$config_file --id=$commit_id --ts=$time_stamp --dir=$output_dir 1>$out_file 2>$err_file &

pid=$!

echo "Stdout dir:   $out_file"
echo "Start time:   `date -d @$time_stamp  '+%Y-%m-%d %H:%M:%S'`"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
echo "pid:          $pid"
cat $config_file

tail -f $out_file $err_file -q
