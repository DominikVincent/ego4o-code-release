import numpy as np
import torch

from .mesh_viewer import MeshViewer

c2c = lambda tensor: tensor.detach().cpu().numpy()

colors = {
    'pink': [.7, .7, .9],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .7, .7],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [.5, .65, .9],

    'grey': [.7, .7, .7],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}

smpl_connections = [[11, 8], [8, 5], [5, 2], [2, 0], [10, 7], [7, 4], [4, 1], [1, 0],
                    [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [12, 13], [13, 16], [16, 18],
                    [18, 20], [12, 14], [14, 17], [17, 19], [19, 21]]


def render_skeleton_pyrender(imw=1080, imh=1080, fps=30, contacts=None,
                             render_joints=True, render_skeleton=True, render_ground=True, ground_plane=None,
                             use_offscreen=False, out_path=None, wireframe=False, RGBA=False,
                             joints_seq=None, joints_vel=None, follow_camera=False, vtx_list=None, points_seq=None,
                             points_vel=None,
                             static_meshes=None, camera_intrinsics=None, img_seq=None, point_rad=0.015,
                             skel_connections=smpl_connections, img_extn='png', ground_alpha=1.0, body_alpha=None,
                             mask_seq=None,
                             cam_offset=[0.0, 5.0, 1.25], ground_color0=[0.8, 0.9, 0.9], ground_color1=[0.6, 0.7, 0.7],
                             joint_rad=0.035,
                             point_color=[1.0, 1.0, 0.0],
                             joint_color=[0.0, 1.0, 0.0],
                             contact_color=[1.0, 0.0, 0.0],
                             render_bodies_static=None,
                             render_points_static=None,
                             cam_rot=None,
                             cam_traj=None
                             ):
    '''
    Visualizes the body model output of a smpl sequence.
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    '''

    if contacts is not None and torch.is_tensor(contacts):
        contacts = c2c(contacts)

    if render_joints and torch.is_tensor(joints_seq[0]):
        joints_seq = [c2c(joint_frame) for joint_frame in joints_seq]
        for joint_seq_item in joints_seq:
            if torch.is_tensor(joint_seq_item['joints']):
                joint_seq_item['joints'] = c2c(joint_seq_item['joints'])

    if joints_vel is not None and torch.is_tensor(joints_vel[0]):
        joints_vel = [c2c(joint_frame) for joint_frame in joints_vel]
    if points_vel is not None and torch.is_tensor(points_vel[0]):
        points_vel = [c2c(joint_frame) for joint_frame in points_vel]

    mv = MeshViewer(width=imw, height=imh,
                    use_offscreen=use_offscreen,
                    follow_camera=follow_camera,
                    camera_intrinsics=camera_intrinsics,
                    img_extn=img_extn,
                    default_cam_offset=cam_offset,
                    default_cam_rot=cam_rot)
    if render_joints and render_skeleton:
        for joint_seq_item in joints_seq:
            mv.add_point_seq(joint_seq_item['joints'], color=joint_seq_item['joint_color'], radius=joint_rad,
                             contact_seq=contacts,
                             connections=skel_connections, connect_color=joint_seq_item['skel_color'], vel=joints_vel,
                             contact_color=contact_color, render_static=render_points_static)
        # mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts,
        #                  connections=skel_connections, connect_color=skel_color, vel=joints_vel,
        #                  contact_color=contact_color, render_static=render_points_static)
    elif render_joints:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts, vel=joints_vel,
                         contact_color=contact_color,
                         render_static=render_points_static)

    # print(points_seq.shape)
    if points_seq is not None:
        # points_seq *= .0
        if torch.is_tensor(points_seq[0]):
            points_seq = [c2c(point_frame) for point_frame in points_seq]
        mv.add_point_seq(points_seq, color=point_color, radius=point_rad, vel=points_vel,
                         render_static=render_points_static)

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_joints:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = points_seq[0][0, :]

        mv.add_ground(ground_plane=ground_plane, xyz_orig=xyz_orig, color0=ground_color0, color1=ground_color1,
                      alpha=ground_alpha)

    mv.set_render_settings(out_path=out_path, wireframe=wireframe, RGBA=RGBA,
                           single_frame=(
                                   render_points_static is not None or render_bodies_static is not None))  # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps, cam_traj=cam_traj)
    except RuntimeError as err:
        print('Could not render properly with the error: %s' % (str(err)))

    del mv
