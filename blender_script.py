import os
import bpy
import joblib
import numpy as np
from mathutils import Matrix, Vector


path_code = "" # os.path.dirname(os.path.abspath(__file__))
result_folder = "tennis1"
wham_pose = f"/home/user1/Prathamesh/3D/Animation/WHAM/output/demo/{result_folder}/wham_output.pkl"
smplerx_pose = f"demo/results/{result_folder}/smplx/body_poses.npy"
smpl_model = '/home/user1/Prathamesh/3D/Animation/SMPLX_Model/SMPLX_56_FLAT_GROUND.fbx'
output_fbx_path = f"demo/results/{result_folder}/animation.fbx"
output_bvh_path = f"demo/results/{result_folder}/animation.bvh"


JOINT_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3"]

smplx_map = {f'bone_{i:02d}': joint for i, joint in enumerate(JOINT_NAMES)}


# Function definitions
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    r = np.array(r)
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2][0], r[1][0]],
                      [r[2][0], 0, -r[0][0]],
                      [-r[1][0], r[0][0], 0]])
    
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat



def rot2quat(rot):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot.reshape(9)
    q_abs = np.array([
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ])
    q_abs = np.sqrt(np.maximum(q_abs, 0))

    quat_by_rijk = np.vstack(
        [
            np.array([q_abs[0] ** 2, m21 - m12, m02 - m20, m10 - m01]),
            np.array([m21 - m12, q_abs[1] ** 2, m10 + m01, m02 + m20]),
            np.array([m02 - m20, m10 + m01, q_abs[2] ** 2, m12 + m21]),
            np.array([m10 - m01, m20 + m02, m21 + m12, q_abs[3] ** 2]),
        ]
    )
    flr = 0.1
    quat_candidates = quat_by_rijk / np.maximum(2.0 * q_abs[:, None], 0.1)

    idx = q_abs.argmax(axis=-1)

    quat = quat_candidates[idx]
    return quat


def deg2rad(angle):
    return -np.pi * (angle + 90) / 180.


def init_scene(scene):
    path_fbx = os.path.join(path_code, smpl_model)
    bpy.ops.import_scene.fbx(filepath=path_fbx, axis_forward='-Y', axis_up='Z', global_scale=1)
    
    obname = "SMPLX-mesh-neutral"
    ob = bpy.data.objects[obname]
    
    # arm_ob = bpy.data.objects['Armature']
    arm_ob = bpy.data.objects['SMPLX-neutral']

    # Ensure the armature is the active object
    bpy.context.view_layer.objects.active = arm_ob

    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Remove the bone named 'm_avg_root'
    bpy.context.object.data.edit_bones.remove(bpy.context.object.data.edit_bones['root'])

    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # ob.data.use_auto_smooth = False
    ob.data.shape_keys.animation_data_clear()
    # arm_ob.animation_data_clear()
    return ob, obname, arm_ob


def rotate180(rot):
    xyz_convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float32)
    return np.dot(xyz_convert.T, rot)


def convert_transl(transl):
    xyz_convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float32)
    return transl.dot(xyz_convert)


def rodrigues2bshapes(pose):
    # if pose.size == 15 * 9:
    #     rod_rots = np.asarray(pose).reshape(15, 3, 3)
    #     mat_rots = [rod_rot for rod_rot in rod_rots]
    # else:
    #     rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in pose]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return (mat_rots, bshapes)


# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, arm_ob, frame=None):
    mrots, _ = rodrigues2bshapes(pose)
    arm_ob.pose.bones['pelvis'].location = trans
    arm_ob.pose.bones['pelvis'].keyframe_insert('location', frame=frame)
    # arm_ob.pose.bones['pelvis'].rotation_quaternion.w = 0.0
    # arm_ob.pose.bones['pelvis'].rotation_quaternion.x = -1.0

    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[smplx_map['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)


# Clear the scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Initialize scene
scene = bpy.context.scene
ob, obname, arm_ob = init_scene(scene)

# Process animation frames
character = 0

# wham = joblib.load(wham_pose)
smplerx = np.load(smplerx_pose)
qtd_frames = len(smplerx)

for fframe in range(qtd_frames):
    scene.frame_set(fframe)
    tran = [0, 0, 0]
    # tran = wham[character]['trans_world'][fframe]
    # wham_pose = wham[character]['pose_world'][fframe]
    # wham_pose = wham_pose.reshape(-1, 3)
    
    smplerx_pose = smplerx[fframe]
    pose = np.zeros((52, 3)) + 1e-5
    pose[1:, :] = smplerx_pose
    # pose[:13, :] = wham_pose[:13, :]
    # pose[13:, :] = smplerx_pose[12:, :]
    # breakpoint()

    apply_trans_pose_shape(Vector(tran), pose, arm_ob, fframe)
    bpy.context.view_layer.update()

# Rename objects
# arm_ob.name = 'Finalized_Armature'
# ob.name = 'Finalized_Mesh'

# Save the result as a BVH file
bpy.ops.export_anim.bvh(filepath=output_bvh_path, frame_start=1, frame_end=qtd_frames)
bpy.ops.export_scene.fbx(filepath=output_fbx_path, axis_forward='-Y', axis_up='Z', global_scale=1.0)
print(f'Successfully saved BVH to {output_bvh_path}')
