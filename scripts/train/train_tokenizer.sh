export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1200
CUDA_VISIBLE_DEVICES=2 accelerate launch  --num_processes 1 --main_process_port 29701 train_tokenizer.py \
--add-hand True \
--batch-size 64 \
--lr 5e-5 \
--total-iter 6000000 \
--lr-scheduler 300000 \
--down-t 1 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir FSQ \
--dataname motionmillion \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name mmfsqad_4096_causal_debug \
--quantizer FSQ \
--nb-code 4096 \
--motion_type vector_272 \
--version version1/tokenizer_96 \
--warm-up-iter 2000 \
--num-workers 64 \
--window-size 96 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--causal \
--vq-norm LN \
--eval-iter 1000000 \
--save-iter 1000000 
#--resume-pth /ssd/caoshiqin/FSQ/train_FSQ_totaliter3000000_codebook8192/net_290000.pth

# --print-iter 1 \
# --eval-iter 10 \
# --save-iter 10 \
# --save-latest 1
# --resume-pth 