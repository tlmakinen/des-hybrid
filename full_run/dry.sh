#!/bin/bash


# dry run through 


python compress_cls2.py --config config_dry.yaml

python compress_patch2.py --config config_dry.yaml --patch A

python compress_patch2.py --config config_dry.yaml --patch B --summary_file summaries_test_A.npz

python compress_patch2.py --config config_dry.yaml --patch C --summary_file summaries_test_B.npz




## python compress_patch2.py --config config/config_dry.yaml --patch A --no_train 1 --load_dir /data103/makinen/des_results/dry_run/patch_A_net_test/best_params.pkl


python compress_patch2.py --config config_dry2.yaml --patch A


python compress_patch2.py --config config/config_dry2.yaml --patch B --summary_file summaries3_test3_A.npz




# FOR THE EXTREME COMPRESSION RUN
python compress_patch2.py --config config/extreme.yaml --patch A


python compress_patch2.py --config config/dry.yaml --patch A --no_noise_cls 1



python compress_patch2.py --config config/extreme2.yaml --patch A



# get summaries
python compress_patch2.py --config config/extreme2.yaml --patch A --no_train 1 --custom_name in_prog


python compress_patch2.py --config config/dry.yaml --patch A --no_train 1 --custom_name in_prog_2




# CURRENT RUN PROGRESS
# TO RUN:
python compress_patch2.py --config config/extreme.yaml --patch A --scale_kappa 10.0

python compress_patch2.py --config config/extreme.yaml --patch A --no_train 1 --custom_name in_prog_2


# rerun dry run with sacling 
python compress_patch2.py --config config/dry.yaml --patch A --scale_kappa 10.0 



# best run so far >>>>
# run patch B for scaled extreme summaries
python compress_patch2.py --config config/extreme.yaml --patch A --scale_kappa 10.0
python compress_patch2.py --config config/extreme.yaml --patch B --scale_kappa 10.0 --summary_file summaries_test_scaled_A.npz



# to obtain summaries:
python compress_patch2.py --config config/extreme.yaml --patch B --no_train 1 --custom_name in_prog --scale_kappa 10.0 --summary_file summaries_test_scaled_A.npz



# RUN THE DRY SCHEME FOR PATCH B

python compress_patch2.py --config config/dry.yaml --patch B --no_train 1 --custom_name in_prog --summary_file summaries_vanilla2_A.npz


# and for patch C
python compress_patch2.py --config config/dry.yaml --patch C --summary_file summaries_vanilla2_B.npz




# RUN the convonly scheme for patch A etc






# RUN THE DRY SCHEME FOR PATCH B

python compress_patch3.py --config config/dry_get_summs.yaml --patch A --no_train 1 --custom_name _get_sc

python compress_patch3.py --config config/dry_get_summs.yaml --patch B --no_train 1 --custom_name _get_sc --summary_file summaries_vanilla2_A_get_sc.npz

python compress_patch3.py --config config/dry_get_summs.yaml --patch C --no_train 1 --custom_name _get_sc --summary_file summaries_vanilla2_B_get_sc.npz



# new run with ell cut
python compress_cls2.py --config config/dry_ell_cut.yaml 
python compress_patch3.py --config config/dry_ell_cut.yaml  --patch A


# check for relu activation in Cls
python compress_cls2.py --config ./config/dry_cls_relu.yaml
python compress_patch3.py --config config/dry_cls_relu.yaml  --patch A


