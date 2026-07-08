"""HumanML3D feature-extraction functions for the 30 fps Nymeria rebuild.

The functions `fill_nan_with_previous_frame`, `uniform_skeleton`, `process_file`,
`recover_root_rot_pos` and `recover_from_ric` are copied VERBATIM from
`/local/home/dhollidt/repos/HumanML3DFork/motion_representation_nymeria.py`
(the script that produced `processed_nymeria_scene_sub_split`), so that the
30 fps features built here follow exactly the same convention as the user's
20 fps features. Do not "improve" them.

Their dependencies (`common.skeleton`, `common.quaternion`, `paramUtil`) are
imported read-only from the HumanML3DFork repo itself to guarantee identical
math (no copies to drift).

Module globals (skeleton constants) mirror the `__main__` block of the source
script; `init_target_skeleton()` must be called once before `process_file`.
"""
import sys

HUMANML3D_FORK = '/local/home/dhollidt/repos/HumanML3DFork'
if HUMANML3D_FORK not in sys.path:
    sys.path.insert(0, HUMANML3D_FORK)

import numpy as np
import torch

from common.skeleton import Skeleton  # noqa: E402  (HumanML3DFork)
from common.quaternion import (  # noqa: E402,F401  (HumanML3DFork)
    qbetween_np, qrot_np, qmul_np, qinv_np, qfix, qrot, qinv,
    quaternion_to_cont6d_np, quaternion_to_cont6d,
)
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # noqa: E402  (HumanML3DFork)

# ---- constants, mirroring motion_representation_nymeria.py __main__ ----
# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

tgt_offsets = None  # set by init_target_skeleton()


def init_target_skeleton(example_joint_positions):
    """Set the target skeleton offsets, mirroring the source script.

    The source script derived `tgt_offsets` from frame 0 of the FIRST item of
    the sorted `joint_positions.npy` list (== line 1 of data_order.txt).
    Pass that same frame-0 data here (interpolation preserves frame 0, so the
    20 fps original is exact).
    """
    global tgt_offsets
    example_data = example_joint_positions.reshape(len(example_joint_positions), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    return tgt_offsets


# ---- verbatim copies from motion_representation_nymeria.py ----

def fill_nan_with_previous_frame(positions):
    """
    Replaces frames containing NaN values with the last valid frame.

    Args:
        positions (np.ndarray): A 3D NumPy array of shape (frames, joints, coordinates)
                                that may contain NaN values.

    Returns:
        np.ndarray: A new NumPy array with NaN frames replaced.
    """
    # Create a copy to avoid modifying the original array
    filled_positions = np.copy(positions)

    # print how many rows are containing NaNs
    num_nan_frames = np.sum(np.any(np.isnan(filled_positions), axis=(1, 2)))
    if num_nan_frames > 0:
        print(f"Number of frames with NaNs: {num_nan_frames} out of {len(filled_positions)}")

    last_valid_frame = None

    for i in range(len(filled_positions)):
        # Check if the current frame contains any NaNs across the joint and coordinate axes
        if np.any(np.isnan(filled_positions[i])):
            # If the first frame is NaN, we cannot fill it yet.
            # This check ensures we have a valid frame to copy from.
            if last_valid_frame is not None:
                # Replace the entire NaN frame with the last valid frame
                filled_positions[i] = last_valid_frame
        else:
            # If the current frame is valid, update the last_valid_frame
            last_valid_frame = filled_positions[i]

    # After the first pass, it's possible the very first frames were NaN and were not filled.
    # We can handle this by back-filling from the first valid frame.
    # Find the first valid frame index
    first_valid_frame_idx = np.where(~np.any(np.isnan(filled_positions), axis=(1, 2)))[0]

    if len(first_valid_frame_idx) > 0:
        first_valid_idx = first_valid_frame_idx[0]
        # Fill any leading NaN frames with the first valid frame
        for i in range(first_valid_idx):
            filled_positions[i] = filled_positions[first_valid_idx]

    return filled_positions


def uniform_skeleton(positions, target_offset):
    """
    Retargets a motion sequence to a skeleton with standardized bone lengths.

    This function normalizes the skeleton's proportions by scaling it to match
    a target skeleton's bone lengths. It preserves the original motion's rotational
    data (pose) and root trajectory, applying them to the new standardized skeleton.

    The scaling factor is determined by comparing the leg lengths of the source
    and target skeletons.

    Args:
        positions (np.ndarray): The input motion sequence, as a numpy array of
            joint positions with shape (seq_len, joints_num, 3).
        target_offset (torch.Tensor): The desired bone offsets for the target
            skeleton, with shape (joints_num, 3).

    Returns:
        np.ndarray: A new motion sequence with the same shape as the input, but
            with the joint positions adjusted to the target skeleton's proportions.
    """
    def check_nan_local(data, msg):
        if np.isnan(data).any():
            raise ValueError(f"NaN detected in uniform_skeleton: {msg}")

    check_nan_local(positions, "input positions")

    # Check for zero-length bones or collapsed joints in input
    # This often causes IK to fail (division by zero in normalization)
    diffs = positions[:, 1:] - positions[:, :-1]
    dists = np.linalg.norm(diffs, axis=-1)
    if (dists < 1e-6).any():
        # Find which joints/frames are collapsed
        bad_indices = np.where(dists < 1e-6)
        print(f"Warning: Collapsed joints detected at indices {bad_indices}. This may cause NaNs in IK.")

    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    check_nan_local(src_offset, "src_offset")

    tgt_offset = target_offset.numpy()
    check_nan_local(tgt_offset, "tgt_offset")

    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    if src_leg_len == 0:
        raise ValueError("src_leg_len is zero, cannot calculate scale_rt")

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    if np.isnan(scale_rt) or np.isinf(scale_rt):
        raise ValueError(f"Invalid scale_rt: {scale_rt}")

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt
    check_nan_local(tgt_root_pos, "tgt_root_pos")

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # check_nan_local(quat_params, "quat_params (after IK)")
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    # check_nan_local(new_joints, "new_joints (after FK)")

    return new_joints


def process_file(positions, feet_thre):
    """
    Processes a single motion sequence to extract a comprehensive feature representation.

    Returns:
        tuple: (data (seq_len-1, 263), global_positions, positions (root-local), l_velocity)
        See motion_representation_nymeria.py for the full 263-dim layout docs.
    """
    def check_nan(data, msg):
        if np.isnan(data).any():
            raise ValueError(f"NaN detected in {msg}")

    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    check_nan(positions, "raw_positions")
    positions = uniform_skeleton(positions, tgt_offsets)
    positions = fill_nan_with_previous_frame(positions)

    check_nan(positions, "uniform_skeleton")

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / (np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis] + 1e-8)
    check_nan(across, "across vector normalization")

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / (np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis] + 1e-8)
    check_nan(forward_init, "forward_init vector normalization")

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    check_nan(root_quat_init, "root_quat_init")

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)
    check_nan(positions, "positions after initial rotation")

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
        quat_params = qfix(quat_params)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    check_nan(cont_6d_params, "cont_6d_params")
    check_nan(r_velocity, "r_velocity (quaternion)")
    check_nan(velocity, "velocity")
    check_nan(r_rot, "r_rot")

    positions = get_rifke(positions)
    check_nan(positions, "positions after rifke")

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    # r_velocity: Root's angular velocity around the y-axis (in radians).

    # Fix: Clip r_velocity before arcsin to avoid NaN
    r_velocity_val = r_velocity[:, 2:3]
    r_velocity_val = np.clip(r_velocity_val, -1.0, 1.0)
    r_velocity = np.arcsin(r_velocity_val)
    check_nan(r_velocity, "r_velocity after arcsin")

    # l_velocity: Root's linear velocity on the xz-plane, in the root's local coordinate system.
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    check_nan(data, "final data")

    return data, global_positions, positions, l_velocity


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions
