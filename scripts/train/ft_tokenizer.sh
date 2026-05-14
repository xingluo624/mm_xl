export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1200
CUDA_VISIBLE_DEVICES=1 accelerate launch  --num_processes 1 --main_process_port 29701 train_tokenizer.py \
--data-root /ssd/caoshiqin/datasets/our_mocap_data/processed_data \
--add-hand True \
--causal \
--batch-size 64 \
--lr 5e-5 \
--total-iter 3000000 \
--lr-scheduler 300000 \
--down-t 1 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir FSQ \
--dataname mocap \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name fsqft_326331417424507508 \
--quantizer FSQ \
--nb-code 4096 \
--motion_type vector_274 \
--warm-up-iter 2000 \
--num-workers 64 \
--window-size 96 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--vq-norm LN \
--eval-iter 500000 \
--save-iter 500000 \
--resume-pth /ssd/caoshiqin/FSQ/274pt_bce_4096_causal/net_2000000.pth

# --print-iter 1 \
# --eval-iter 10 \
# --save-iter 10 \
# --save-latest 1
# --resume-pth 