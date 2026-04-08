import torch
import torch.nn as nn

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, motion_type) :
        super(ReConsLoss, self).__init__()
        
        self.motion_type = motion_type
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        
        if motion_type == 'vector_272':
            self.motion_dim = 272
            self.continuous_dim = 272
        elif motion_type == 'vector_274':
            self.motion_dim = 274
            self.continuous_dim = 272  # 前 272 维：关节运动（回归）
            self.discrete_dim = 2  
            
            self.BCELoss = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, motion_pred, motion_gt) : 
        if self.motion_type == 'vector_272':
            # 全部用回归损失
            return self.Loss(motion_pred, motion_gt)
        
        elif self.motion_type == 'vector_274':
            loss_continuous = self.Loss(motion_pred[..., :self.continuous_dim], motion_gt[..., :self.continuous_dim])
            
            loss_discrete = self.BCELoss(motion_pred[..., self.continuous_dim:], motion_gt[..., self.continuous_dim:])
            
            return loss_continuous + 0.3 *loss_discrete
            
    
    def forward_vel(self, motion_pred, motion_gt) : 
        # loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        loss = self.Loss(motion_pred[..., 8 : 8+3*self.nb_joints], motion_gt[..., 8 : 8+3*self.nb_joints])
        return loss
    
    def forward_acc(self, motion_pred, motion_gt) : 
        B, T, D = motion_pred.shape
        
        pred_vel = motion_pred[:,1:, 8 : 8+3*self.nb_joints]-motion_pred[:, :-1, 8 : 8+3*self.nb_joints]
        gt_vel = motion_gt[:,1:, 8 : 8+3*self.nb_joints]-motion_gt[:, :-1, 8 : 8+3*self.nb_joints]
        
        loss = self.Loss(pred_vel, gt_vel)
        
        return loss
    
    def forward_acc_vel(self, motion_pred, motion_gt) : 
        B, T, D = motion_pred.shape
        
        pred_vel = motion_pred[:,1:, 8 : 8+3*self.nb_joints]-motion_pred[:, :-1, 8 : 8+3*self.nb_joints]
        pred_acc = pred_vel[:,1:, :]-pred_vel[:, :-1, :]
        
        gt_vel = motion_gt[:,1:, 8 : 8+3*self.nb_joints]-motion_gt[:, :-1, 8 : 8+3*self.nb_joints]
        gt_acc = gt_vel[:,1:, :]-gt_vel[:, :-1, :]
        
        loss = self.Loss(pred_acc, gt_acc)
        
        return loss
    
    
    def forward_root(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., :8], motion_gt[..., :8])
        return loss
    