# for OmegaM
python compress_patch_single.py --config dry_single.yaml --param 0 --patch A 

python compress_patch_single.py --config dry_single.yaml --param 0 --patch B --summary_file summaries_vanilla_A.npz

python compress_patch_single.py --config dry_single.yaml --param 0 --patch C --summary_file summaries_vanilla_B.npz