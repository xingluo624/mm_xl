export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=0 python fsq_chunk.py \
--add-hand True \
--nb-code 4096 \
--resume-pth FSQ/fsqft_326331417424507508_causal/net_3000000.pth \
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
--causal \
--motion_type vector_274 \
