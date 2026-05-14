CUDA_VISIBLE_DEVICES=1 python tools/test_fsq.py \
--add-hand True \
--exp-name Qwen3_motioncode \
--num-layers 9 \
--nb-code 4096 \
--resume-pth FSQ/mmifsqft_274_4096/net_6000000.pth \
--vq-name VQVAE_codebook_4096_FSQ_all \
--out-dir results/output/inference/single_inference/ \
--dataname mocap \
--down-t 1 \
--depth 3 \
--quantizer FSQ \
--dilation-growth-rate 3 \
--vq-act relu \
--vq-norm LN \
--fps 30 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--pkeep 1 \
--motion_type vector_274 \

# --print-iter 1 \
# --eval-metric-iter 1 \
# --eval-loss-iter 1 

# --eval-metric-iter 5000 \
# --eval-loss-iter 2000 \