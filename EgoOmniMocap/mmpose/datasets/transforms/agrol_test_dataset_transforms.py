from typing import List, Union, Dict, Optional, Tuple

import torch
from mmcv.transforms import BaseTransform
# from mmpose.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS



@TRANSFORMS.register_module()
class NormalizeTrajectory(BaseTransform):
    def __init__(self, key,
                 traj_mean_path, traj_std_path,
                 control_joint_ids=(15, 20, 21),
                 output_joint_ids=(0, 15, 20, 21, 10, 11),
                 ):
        super().__init__()
        self.key = key
        self.traj_mean = torch.load(traj_mean_path)
        self.traj_std = torch.load(traj_std_path)

        self.output_joint_ids_not_in_joint_ids = []
        for i in range(len(output_joint_ids)):
            if output_joint_ids[i] not in control_joint_ids:
                self.output_joint_ids_not_in_joint_ids.append(i)

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        traj_data = results[self.key]

        normalize_traj_data = self.normalize(traj_data)
        normalize_traj_data[:, self.output_joint_ids_not_in_joint_ids, :] *= .0
        results[self.key] = normalize_traj_data
        return results

    def normalize(self, trajectory):
        traj_mean = self.traj_mean.to(trajectory.device)
        traj_std = self.traj_std.to(trajectory.device)
        trajectory = (trajectory - traj_mean) / traj_std
        return trajectory

    def inv_normalize(self, trajectory):
        traj_mean = self.traj_mean.to(trajectory.device)
        traj_std = self.traj_std.to(trajectory.device)
        trajectory = trajectory * traj_std + traj_mean
        return trajectory

