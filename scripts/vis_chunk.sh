CUDA_VISIBLE_DEVICES=0 python vis_video_chunk.py \
--qwen_model_path /data_public/zjk/csq/PyProject/ft_qwenvl/logs/517_3m_2taskallidx_16kepoch_noresize \
--add-hand True \
--exp-name vis_v \
--nb-code 4096 \
--resume-pth FSQ/507508/net_2100000.pth \
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
--motion_type vector_274

