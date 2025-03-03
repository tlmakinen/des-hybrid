#!/bin/bash


# dry run through 


python compress_cls2.py --config config_dry.yaml

python compress_patch2.py --config config_dry.yaml --patch A

python compress_patch2.py --config config_dry.yaml --patch B --summary_file summaries_test_A.npz

python compress_patch2.py --config config_dry.yaml --patch C --summary_file summaries_test_B.npz

