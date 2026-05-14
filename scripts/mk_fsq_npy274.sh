export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1200
accelerate launch --num_processes 1 mk_fsq_npy.py \
--add-hand True \
--exp-name get_codes \
--nb-code 4096 \
--resume-pth FSQ/fsqft_326331417424507508/net_3000000.pth \
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
--motion_type vector_274 \
--causal 



