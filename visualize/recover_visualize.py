import numpy as np
import os
import visualize.plot_3d_global as plot_3d
from visualize.smplx2joints import process_smplx_data

def smpl85_2_smpl322(smpl_85_data):
    result = np.concatenate((smpl_85_data[:,:66], np.zeros((smpl_85_data.shape[0], 90)), np.zeros((smpl_85_data.shape[0], 3)), np.zeros((smpl_85_data.shape[0], 50)), np.zeros((smpl_85_data.shape[0], 100)), smpl_85_data[:,72:72+3], smpl_85_data[:,75:]), axis=-1)
    return result

def visualize_smpl_85(data, title=None, output_path='visualize_result', fps=30):
    # data: torch.Size([nframe, 85])
   
    smpl_85_data = data
    if len(smpl_85_data.shape) == 3:
       smpl_85_data = np.squeeze(smpl_85_data.cpu().numpy(), axis=0)
    
    smpl_85_data = smpl85_2_smpl322(smpl_85_data)
    vert, joints, motion, faces = process_smplx_data(smpl_85_data, norm_global_orient=False, transform=False)
    xyz = joints[:, :22, :].reshape(1, -1, 22, 3).detach().cpu().numpy()
    os.makedirs(os.path.dirname(output_path[0]), exist_ok=True)
    
    pose_vis = plot_3d.draw_to_batch(xyz, title_batch=title, outname=[f'{output_path[0]}'], fps=fps)
    return output_path