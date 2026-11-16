"""Unchanged Future camera, state and environment preprocessing."""
import math
from pathlib import Path

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)


def _quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def _get_libero_env(task, resolution, seed):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def _observation(obs, prompt):
    result = {}
    for source, target in [("agentview_image", "observation/image"),
                           ("robot0_eye_in_hand_image", "observation/wrist_image")]:
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        result[target] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    result["observation/state"] = np.concatenate((obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    result["prompt"] = str(prompt)
    return result

