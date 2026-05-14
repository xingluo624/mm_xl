import os.path as osp
from pathlib import Path
from typing import List, Dict
from typing import Union
import os

import numpy as np
import torch
from colour import Color
from loguru import logger
from scipy.spatial.transform import Rotation as R
from torch import nn

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import create_list_chunks
from tqdm import tqdm
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
import os.path as osp
from glob import glob

import numpy as np
import torch
from loguru import logger
from human_body_prior.tools.omni_tools import get_support_data_dir

from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import render_smpl_params
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import get_support_data_dir

import cv2


class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 n_joints: int = 22,
                 kpts_colors: Union[np.ndarray, None] = None,
                 num_betas=16
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, num_betas=num_betas, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []  # self.bm.f
        self.n_joints = n_joints
        self.kpts_colors = np.array(
            [Color('grey').rgb for _ in range(n_joints)]) if kpts_colors == None else kpts_colors

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {'source_kpts': new_body.Jtr[:, :self.n_joints], 'body': new_body}


def transform_smpl_coordinate(bm_fname: Path, trans: np.ndarray,
                              root_orient: np.ndarray, betas: np.ndarray, rotxyz: Union[np.ndarray, List]) -> Dict:
    """
    rotates smpl parameters while taking into account non-zero center of rotation for smpl
    Parameters
    ----------
    bm_fname: body model filename
    trans: Nx3
    root_orient: Nx3
    betas: num_betas
    rotxyz: desired XYZ rotation in degrees

    Returns
    -------

    """
    if isinstance(rotxyz, list):
        rotxyz = np.array(rotxyz).reshape(1, 3)
    if betas.ndim == 1: betas = betas[None]
    if betas.ndim == 2 and betas.shape[0] != 1:
        logger.warning(
            f'betas should be the same for the entire sequence. 2D np.array with 1 x num_betas: {betas.shape}. taking the mean')
        betas = np.mean(betas, keepdims=True, axis=0)
    transformation_euler = np.deg2rad(rotxyz)

    coord_change_matrot = R.from_euler('XYZ', transformation_euler.reshape(1, 3)).as_matrix().reshape(3, 3)
    bm = BodyModel(bm_fname=bm_fname,
                   num_betas=betas.shape[1])
    pelvis_offset = c2c(bm(**{'betas': torch.from_numpy(betas).type(torch.float32)}).Jtr[[0], 0])

    root_matrot = R.from_rotvec(root_orient).as_matrix().reshape([-1, 3, 3])

    transformed_root_orient_matrot = np.matmul(coord_change_matrot, root_matrot.T).T
    transformed_root_orient = R.from_matrix(transformed_root_orient_matrot).as_rotvec()
    transformed_trans = np.matmul(coord_change_matrot, (trans + pelvis_offset).T).T - pelvis_offset

    return {'root_orient': transformed_root_orient.astype(np.float32),
            'trans': transformed_trans.astype(np.float32), }


def convert_motion_mp4(planned_mtoion, motion_path, save_render=True,
                    comp_device='cuda:0',
                    surface_model_type='smplx', gender='neutral', batch_size=128, verbosity=0, partial_fit = False,text:str=''):
    """
    :param skeleton_movie_fname: either a result npy file or a motion numpy array [nframes, njoints, 3]
    :param surface_model_type:
    :param gender:
    :param batch_size:
    :param verbosity: 0: silent, 1: text, 2: visual with psbody.mesh
    :return:
    """

    support_base_dir = get_support_data_dir()
    support_dir = osp.join(support_base_dir, 'dowloads/dowloads')  # '../../../support_data/dowloads'
    logger.info(f'found support_dir: {support_dir}')
    # 'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
    vposer_expr_dir = osp.join(support_dir, 'vposer_v2_05')
    bm_fname = osp.join(support_dir, f'models/{surface_model_type}/{gender}/model.npz')

    if partial_fit:
    # 'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
        pad1 = torch.zeros((planned_mtoion.shape[0], 6, 3)).type(torch.float32)
        pad2 = torch.zeros((planned_mtoion.shape[0], 11, 3)).type(torch.float32)
        motion_inp = torch.concat((planned_mtoion[:, 0].unsqueeze(1), pad1, planned_mtoion[:, [1,2]], pad2, planned_mtoion[:, [3,4]]), dim=1)
        motion = motion_inp.numpy()
        # render_out_dir = motion_path.parent / "render" / motion_path.stem
        # render_out_dir.mkdir(parents=True, exist_ok=True)
        # render_out_fname = render_out_dir / f"{motion_path.stem}.mp4"
        render_out_fname = motion_path
    else:
        motion = planned_mtoion
        render_out_fname = motion_path

    # if out_fname is None:
    #     out_fname = skeleton_movie_fname.replace('.mp4', '.npz')

    # render_out_dir = os.path.dirname(motion_path) + '/render/' + motion_path.split('/')[-1].split('.')[0] + '/'


    # if osp.exists(render_out_fname):
    #     logger.warning(f'render output already exists: {render_out_fname}. skipping...')
    #     return
    n_joints = 22
    num_betas = 16

    red = Color("red")
    blue = Color("blue")
    kpts_colors = [c.rgb for c in list(red.range_to(blue, n_joints))]

    # create source and target key points and make sure they are index aligned
    data_loss = torch.nn.MSELoss(reduction='sum')

    stepwise_weights = [
        {'data': 10., 'poZ_body': .01, 'betas': .5},
    ]

    optimizer_args = {'type': 'LBFGS', 'max_iter': 300, 'lr': 1, 'tolerance_change': 1e-4, 'history_size': 200}
    ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                          verbosity=verbosity,
                          display_rc=(2, 2),
                          data_loss=data_loss,
                          num_betas=num_betas,
                          stepwise_weights=stepwise_weights,
                          optimizer_args=optimizer_args).to(comp_device)

    all_results = {}
    batched_frames = create_list_chunks(np.arange(len(motion)), batch_size, overlap_size=0, cut_smaller_batches=False)
    if verbosity < 2:
        batched_frames = tqdm(batched_frames, desc='VPoser Advanced IK')
    for cur_frame_ids in batched_frames:

        target_pts = torch.from_numpy(motion[cur_frame_ids, :n_joints]).to(comp_device)
        source_pts = SourceKeyPoints(bm=bm_fname, n_joints=n_joints, kpts_colors=kpts_colors, num_betas=num_betas).to(
            comp_device)

        ik_res = ik_engine(source_pts, target_pts, {})

        ik_res_detached = {k: c2c(v) for k, v in ik_res.items()}
        nan_mask = np.isnan(ik_res_detached['trans']).sum(-1) != 0
        if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
        for k, v in ik_res_detached.items():
            if k not in all_results: all_results[k] = []
            all_results[k].append(v)

    d = {k: np.concatenate(v, axis=0) for k, v in all_results.items()}
    d['betas'] = np.median(d['betas'], axis=0)

    transformed_d = transform_smpl_coordinate(bm_fname=bm_fname, trans=d['trans'], root_orient=d['root_orient'],
                                              betas=d['betas'], rotxyz=[90, 0, 0])
    d.update(transformed_d)
    d['poses'] = np.concatenate([d['root_orient'], d['pose_body'], np.zeros([len(d['pose_body']), 99])], axis=1)

    d['surface_model_type'] = surface_model_type
    d['gender'] = gender
    d['mocap_frame_rate'] = 30
    d['num_betas'] = num_betas
    # breakpoint()
    # np.savez(out_fname, **d)
    # logger.success(f'created: {out_fname}')

    if save_render:
        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas)
        smpl_dict = np.load(bm_fname)
        mean_pose_hand = np.repeat(np.concatenate([smpl_dict['hands_meanl'], smpl_dict['hands_meanr']])[None], axis=0,
                                   repeats=len(motion))

        body_parms = {**d, 'betas': np.repeat(d['betas'][None], axis=0, repeats=len(motion)),
                      'pose_hand': mean_pose_hand}
        body_parms = {k: torch.from_numpy(v) for k, v in body_parms.items() if
                      k in ['root_orient', 'trans', 'pose_body', 'pose_hand']}

        img_array = render_smpl_params(bm, body_parms, [-90, 0, 0])[None, None]
        
        if partial_fit:
            for i, img in enumerate(img_array[0][0]):
                frame_path = os.path.join(render_out_dir, f"frame_{i:04d}.png")
                img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # save images
                cv2.imwrite(frame_path, img_bgr)
            fps = 1
        else:
            fps = 30
        
        imagearray2file(img_array, outpath=str(render_out_fname), fps=fps,text=text)
        logger.success(f'created: {render_out_fname}')

    logger.info(f'You can visualize these results as any amass npz file or in Blender via blender_smplx_addon.')

    # return np.concatenate((d['pose_body'].reshape(-1, 21, 3), d['root_orient'][:, np.newaxis, :]), axis=1)
    results = {
        'trans': d['trans'],
        'poses': np.concatenate((d['root_orient'], d['pose_body']), axis=1),
        'betas': d['betas'],
        'mocap_framerate': 20
    }
    return results

def transform_smpl_coordinate(bm_fname: Path, trans: np.ndarray,
                              root_orient: np.ndarray, betas: np.ndarray, rotxyz: Union[np.ndarray, List]) -> Dict:
    """
    rotates smpl parameters while taking into account non-zero center of rotation for smpl
    Parameters
    ----------
    bm_fname: body model filename
    trans: Nx3
    root_orient: Nx3
    betas: num_betas
    rotxyz: desired XYZ rotation in degrees

    Returns
    -------

    """
    if isinstance(rotxyz, list):
        rotxyz = np.array(rotxyz).reshape(1, 3)
    if betas.ndim == 1: betas = betas[None]
    if betas.ndim == 2 and betas.shape[0] != 1:
        logger.warning(
            f'betas should be the same for the entire sequence. 2D np.array with 1 x num_betas: {betas.shape}. taking the mean')
        betas = np.mean(betas, keepdims=True, axis=0)
    transformation_euler = np.deg2rad(rotxyz)

    coord_change_matrot = R.from_euler('XYZ', transformation_euler.reshape(1, 3)).as_matrix().reshape(3, 3)
    bm = BodyModel(bm_fname=bm_fname,
                   num_betas=betas.shape[1])
    pelvis_offset = c2c(bm(**{'betas': torch.from_numpy(betas).type(torch.float32)}).Jtr[[0], 0])

    root_matrot = R.from_rotvec(root_orient).as_matrix().reshape([-1, 3, 3])

    transformed_root_orient_matrot = np.matmul(coord_change_matrot, root_matrot.T).T
    transformed_root_orient = R.from_matrix(transformed_root_orient_matrot).as_rotvec()
    transformed_trans = np.matmul(coord_change_matrot, (trans + pelvis_offset).T).T - pelvis_offset

    return {'root_orient': transformed_root_orient.astype(np.float32),
            'trans': transformed_trans.astype(np.float32), }



def get_img_array(planned_mtoion, save_render=True,
                    comp_device='cuda:0',
                    surface_model_type='smplx', gender='neutral', batch_size=128, verbosity=0, partial_fit = True):
    """
    :param skeleton_movie_fname: either a result npy file or a motion numpy array [nframes, njoints, 3]
    :param surface_model_type:
    :param gender:
    :param batch_size:
    :param verbosity: 0: silent, 1: text, 2: visual with psbody.mesh
    :return:
    """

    support_base_dir = get_support_data_dir()
    support_dir = osp.join(support_base_dir, 'dowloads')  # '../../../support_data/dowloads'
    logger.info(f'found support_dir: {support_dir}')
    # 'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
    vposer_expr_dir = osp.join(support_dir, 'vposer_v2_05')
    bm_fname = osp.join(support_dir, f'models/{surface_model_type}/{gender}/model.npz')

    if partial_fit:
    # 'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
        pad1 = torch.zeros((planned_mtoion.shape[0], 6, 3)).type(torch.float32)
        pad2 = torch.zeros((planned_mtoion.shape[0], 11, 3)).type(torch.float32)
        motion_inp = torch.concat((planned_mtoion[:, 0].unsqueeze(1), pad1, planned_mtoion[:, [1,2]], pad2, planned_mtoion[:, [3,4]]), dim=1)
        motion = motion_inp.numpy()
    else:
        motion = planned_mtoion

    # if out_fname is None:
    #     out_fname = skeleton_movie_fname.replace('.mp4', '.npz')

    # render_out_dir = os.path.dirname(motion_path) + '/render/' + motion_path.split('/')[-1].split('.')[0] + '/'


    # if osp.exists(render_out_fname):
    #     logger.warning(f'render output already exists: {render_out_fname}. skipping...')
    #     return
    n_joints = 22
    num_betas = 16

    red = Color("red")
    blue = Color("blue")
    kpts_colors = [c.rgb for c in list(red.range_to(blue, n_joints))]

    # create source and target key points and make sure they are index aligned
    data_loss = torch.nn.MSELoss(reduction='sum')

    stepwise_weights = [
        {'data': 10., 'poZ_body': .01, 'betas': .5},
    ]

    optimizer_args = {'type': 'LBFGS', 'max_iter': 300, 'lr': 1, 'tolerance_change': 1e-4, 'history_size': 200}
    ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                          verbosity=verbosity,
                          display_rc=(2, 2),
                          data_loss=data_loss,
                          num_betas=num_betas,
                          stepwise_weights=stepwise_weights,
                          optimizer_args=optimizer_args).to(comp_device)

    all_results = {}
    batched_frames = create_list_chunks(np.arange(len(motion)), batch_size, overlap_size=0, cut_smaller_batches=False)
    if verbosity < 2:
        batched_frames = tqdm(batched_frames, desc='VPoser Advanced IK')
    for cur_frame_ids in batched_frames:

        target_pts = torch.from_numpy(motion[cur_frame_ids, :n_joints]).to(comp_device)
        source_pts = SourceKeyPoints(bm=bm_fname, n_joints=n_joints, kpts_colors=kpts_colors, num_betas=num_betas).to(
            comp_device)

        ik_res, kypts = ik_engine(source_pts, target_pts, {})

        ik_res_detached = {k: c2c(v) for k, v in ik_res.items()}
        nan_mask = np.isnan(ik_res_detached['trans']).sum(-1) != 0
        if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
        for k, v in ik_res_detached.items():
            if k not in all_results: all_results[k] = []
            all_results[k].append(v)

    d = {k: np.concatenate(v, axis=0) for k, v in all_results.items()}
    d['betas'] = np.median(d['betas'], axis=0)

    transformed_d = transform_smpl_coordinate(bm_fname=bm_fname, trans=d['trans'], root_orient=d['root_orient'],
                                              betas=d['betas'], rotxyz=[90, 0, 0])
    d.update(transformed_d)
    d['poses'] = np.concatenate([d['root_orient'], d['pose_body'], np.zeros([len(d['pose_body']), 99])], axis=1)

    d['surface_model_type'] = surface_model_type
    d['gender'] = gender
    d['mocap_frame_rate'] = 30
    d['num_betas'] = num_betas
    # np.savez(out_fname, **d)
    # logger.success(f'created: {out_fname}')

    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas)
    smpl_dict = np.load(bm_fname)
    mean_pose_hand = np.repeat(np.concatenate([smpl_dict['hands_meanl'], smpl_dict['hands_meanr']])[None], axis=0,
                               repeats=len(motion))

    body_parms = {**d, 'betas': np.repeat(d['betas'][None], axis=0, repeats=len(motion)),
                  'pose_hand': mean_pose_hand}
    body_parms = {k: torch.from_numpy(v) for k, v in body_parms.items() if
                  k in ['root_orient', 'trans', 'pose_body', 'pose_hand']}

    img_array = render_smpl_params(bm, body_parms, [-90, 0, 0])[None, None]


    return img_array


