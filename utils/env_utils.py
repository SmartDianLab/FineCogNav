import math
import numba as nb
import airsim
import numpy as np
import copy

from airsim_plugin.airsim_settings import AirsimActions, AirsimActionSettings

from src.common.param import args

from utils.logger import logger
from utils.shorest_path_sensor import EuclideanDistance3, EuclideanDistance1


class SimState:
    def __init__(self, index=-1,
                 step=0,
                 episode_info={},
                 pose=airsim.Pose(),
                 ):
        self.index = index
        self.step = step
        self.episode_info = copy.deepcopy(episode_info)

        self.pose = pose
        self.trajectory = []
        self.is_end = False

        self.SUCCESS_DISTANCE = 20

        self.DistanceToGoal = {
            '_metric': 0,
            '_previous_position': None,
        }
        self.Success = {
            '_metric': 0,
        }
        self.NDTW = {
            '_metric': 0,
            'locations': [],
            'gt_locations': (np.array(episode_info['reference_path'])[:, 0:3]).tolist(),
        }
        self.SDTW = {
            '_metric': 0,
        }
        self.PathLength = {
            '_metric': 0,
            '_previous_position': None,
        }
        self.OracleSuccess = {
            '_metric': 0,
        }
        self.StepsTaken = {
            '_metric': 0,
        }

        self.distance_data_pre_frame = {
            'distance_front_data': 0.0,
            'distance_up_data': 0.0,
            'distance_down_data': 0.0,
            'distance_left_data': 0.0,
            'distance_right_data': 0.0,
        }

        self.pre_carrot_idx = 0
        self.start_point_nearest_node_token = None
        self.end_point_nearest_node_token = None
        self.progress = 0.0
        self.waypoint = {}
        self.unique_path = None
        self.pre_action = AirsimActions.STOP
        self.is_collisioned = False


class ENV:
    def __init__(self, load_scenes: list):
        self.batch = None

    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)
        return

    def get_obs_at(self, index: int, state):
        assert self.batch is not None, 'batch is None'
        item = self.batch[index]

        teacher_action = AirsimActions.STOP
        done = state.is_end
        if not done:
            progress = 0.0
        else:
            progress = 1.0

        return (teacher_action, done, progress), state

def getPoseAfterMakeAction(pose: airsim.Pose, action):
    current_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    current_rotation = np.array([
        pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val
    ])

    if action == AirsimActions.MOVE_FORWARD:
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        pitch = 0
        roll = 0

        unit_x = 1 * math.cos(pitch) * math.cos(yaw)
        unit_y = 1 * math.cos(pitch) * math.sin(yaw)
        unit_z = 1 * math.sin(pitch) * (-1)
        unit_vector = np.array([unit_x, unit_y, unit_z])
        assert unit_z == 0

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.TURN_LEFT:
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        pitch = 0
        roll = 0

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw - math.radians(AirsimActionSettings.TURN_ANGLE)
        if float(new_yaw * 180 / math.pi) < -180:
            new_yaw = math.radians(360) + new_yaw

        new_position = current_position.copy()
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
            new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
        ]
    elif action == AirsimActions.TURN_RIGHT:
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        pitch = 0
        roll = 0

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw + math.radians(AirsimActionSettings.TURN_ANGLE)
        if float(new_yaw * 180 / math.pi) > 180:
            new_yaw = math.radians(-360) + new_yaw

        new_position = current_position.copy()
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
            new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
        ]
    elif action == AirsimActions.GO_UP:
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        pitch = 0
        roll = 0

        unit_vector = np.array([0, 0, -1])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.GO_DOWN:
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        pitch = 0
        roll = 0

        unit_vector = np.array([0, 0, -1])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE * (-1)
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.MOVE_LEFT:
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        pitch = 0
        roll = 0

        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_vector = np.array([unit_x, unit_y, 0])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE * (-1)
        new_rotation = current_rotation.copy()
    elif action == AirsimActions.MOVE_RIGHT:
        pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
        pitch = 0
        roll = 0

        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_vector = np.array([unit_x, unit_y, 0])

        new_position = np.array(current_position) + unit_vector * AirsimActionSettings.LEFT_RIGHT_STEP_SIZE
        new_rotation = current_rotation.copy()
    else:
        new_position = current_position.copy()
        new_rotation = current_rotation.copy()

    new_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            x_val=new_position[0],
            y_val=new_position[1],
            z_val=new_position[2]
        ),
        orientation_val=airsim.Quaternionr(
            x_val=new_rotation[0],
            y_val=new_rotation[1],
            z_val=new_rotation[2],
            w_val=new_rotation[3]
        )
    )
    return new_pose