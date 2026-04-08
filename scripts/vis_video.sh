python test.py \
--qwen_model_path /gemini-2/space/zjk/csq/mocap_addv_4375_326_272 \
--exp-name infer_272 \
--num-layers 9 \
--nb-code 4096 \
--resume-pth FSQ/4096_ft326/net_6015000.pth \
--dataname motionmillion \
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
--motion_type vector_272 

