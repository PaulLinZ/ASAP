import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import Slerp
from isaac_utils.rotations import slerp

class MotionLibKuavo:
    def __init__(self, m_cfg, num_envs, device):
        self.m_cfg = m_cfg
        self.num_envs = num_envs
        self.device = device
        self._num_unique_motions = 1
        
    
    
    def load_data(self):
        """
        读取npz文件并将其中的数据存储到类属性中
        """
        # import ipdb; ipdb.set_trace()
        # 读取npz文件
        data = np.load(self.m_cfg.motion_file, allow_pickle=True)[0]
        
        # data shape: (num_frame, origin_shape)
        self.root_trans = torch.from_numpy(data["root_trans"]).to(self.device)       # (num_frame, 3)
        self.root_quat = torch.from_numpy(data["root_quat"]).to(self.device)           # (num_frame, 4)
        self.root_lin_vel = torch.from_numpy(data["root_lin_vel"]).to(self.device)     # (num_frame, 3)
        self.root_ang_vel = torch.from_numpy(data["root_ang_vel"]).to(self.device)     # (num_frame, 3)
        self.dof_pos = torch.from_numpy(data["joint_pos"]).to(self.device)             # (num_frame, 28)
        self.dof_vel = torch.from_numpy(data["joint_vel"]).to(self.device)           # (num_frame, 28)
        self.body_rigid_pos = torch.from_numpy(data["body_pos"]).to(self.device)             # (num_frame, 29, 3)
        self.body_rigid_quat = torch.from_numpy(data["body_quat"]).to(self.device)           # (num_frame, 29, 4)
        self.frame_rate = torch.tensor(data["frame_rate"]).to(self.device)         # 1
        self.frame_len = torch.tensor(data["frames"]).to(self.device)              # 1  
        self.body_rigid_vel = torch.from_numpy(data["body_lin_vel"]).to(self.device) 
        self.body_rigid_ang_vel = torch.from_numpy(data["body_ang_vel"]).to(self.device) 
        self._motion_dt = torch.tensor(1. / self.frame_rate).to(self.device) 
        self.motion_duration = torch.tensor(self._motion_dt * self.frame_len).to(self.device) 
        self.num_rigid_body = self.body_rigid_pos.shape[1]
        self.num_dof = self.dof_pos.shape[1]
        
        print(f"数据成功加载自 {self.m_cfg.motion_file}")


    def get_motion_state(self,motion_ids, motion_times, offset=None):
        """
        获取特定的运动状态数据

        :param motion_times: 运动时间点的列表或数组
        :param offset: 用于调整时间点的偏移量，默认为0
        :return: 返回运动状态的数据，格式为字典 {motion_id: motion_state}
        """
        if self.root_quat is None:
            raise ValueError("数据尚未加载。请先调用 load_data_from_npz 方法加载数据。")
        
        # import ipdb; ipdb.set_trace()

        motion_state = {}
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, self.motion_duration, self.frame_len, self._motion_dt)
        frame_idx0, frame_idx1, blend = frame_idx0[0], frame_idx1[0], blend[0]
        
        # 对位置和速度使用线性插值
        motion_state["rg_pos_t"] = (self.body_rigid_pos[frame_idx0] + blend * (self.body_rigid_pos[frame_idx1] - self.body_rigid_pos[frame_idx0])).to(torch.float32)
        motion_state["body_vel_t"] = (self.body_rigid_vel[frame_idx0] + blend * (self.body_rigid_vel[frame_idx1] - self.body_rigid_vel[frame_idx0])).to(torch.float32)
        motion_state["dof_vel"] = (self.dof_vel[frame_idx0] + blend * (self.dof_vel[frame_idx1] - self.dof_vel[frame_idx0])).to(torch.float32)
        motion_state["dof_pos"] = (self.dof_pos[frame_idx0] + blend * (self.dof_pos[frame_idx1] - self.dof_pos[frame_idx0])).to(torch.float32)
        motion_state["body_ang_vel_t"] = (self.body_rigid_ang_vel[frame_idx0] + blend * (self.body_rigid_ang_vel[frame_idx1] - self.body_rigid_ang_vel[frame_idx0])).to(torch.float32)
        motion_state["root_pos"] = (self.root_trans[frame_idx0] + blend * (self.root_trans[frame_idx1] - self.root_trans[frame_idx0])).to(torch.float32)
        motion_state["root_vel"] = (self.root_lin_vel[frame_idx0] + blend * (self.root_lin_vel[frame_idx1] - self.root_lin_vel[frame_idx0])).to(torch.float32)
        motion_state["root_rot"] = self.root_quat[frame_idx0].to(torch.float32)
        motion_state["root_ang_vel"] = (self.root_ang_vel[frame_idx0] + blend * (self.root_ang_vel[frame_idx1] - self.root_ang_vel[frame_idx0])).to(torch.float32)

        
        rg_rot_t_temp = []
        for i in range (self.num_rigid_body):
            # import ipdb; ipdb.set_trace()
            rg_rot_t_temp.append(list(slerp(self.body_rigid_quat[frame_idx0][i], self.body_rigid_quat[frame_idx1][i], blend)))
        motion_state["rg_rot_t"] = torch.tensor(rg_rot_t_temp).to(self.device).to(torch.float32)

        for key in motion_state.keys():
            # 获取原始形状
            origin_shape = motion_state[key].shape
            
            new_shape = (self.num_envs,) + origin_shape
            # 扩展张量
            motion_state[key] = motion_state[key].expand(new_shape)
            
            # print(key, motion_state[key].shape)
        # import ipdb; ipdb.set_trace()
        if offset is not None:
            motion_state["root_pos"] = motion_state["root_pos"] + offset
        
        return motion_state


    # 计算frame和插值点
    def _calc_frame_blend(self, time, len, num_frames, dt):
        # import ipdb; ipdb.set_trace()
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()              # 帧索引，在当前时间应该是第几帧
        frame_idx1 = torch.min(frame_idx0 + 1, torch.tensor(num_frames - 1, device=frame_idx0.device))
        
        # blend代表现在在两帧之间的位置比例
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend
    
    def get_motion_length(self, motion_ids):
        return self.motion_duration.expand((len(motion_ids)))

""""
1. 文件读取
2. get state 匹配
3. resample问题
"""

if __name__ == "__main__":
    # 创建 MotionLibKuavo 实例
    motion_lib = MotionLibKuavo(m_cfg=0, num_envs=4096, device="cuda:0")

    # 读取数据
    motion_lib.load_data("/home/liujunjie/amass-retargeting/amass_retargeting/result.npy")

    # 获取运动状态数据
    # motion_ids = [1, 2, 3]
    motion_times = torch.tensor(2)
    motion_state = motion_lib.get_motion_state(motion_times)
    # print(motion_lib.get_motion_length(torch.tensor([1,2,3])))

    # print(motion_state)