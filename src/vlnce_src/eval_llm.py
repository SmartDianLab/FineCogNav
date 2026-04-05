import io
import re
import textwrap
from typing import Dict, Optional, Union, List, DefaultDict
import os
import sys
from pathlib import Path

import cv2
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import gc
import tqdm
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from PIL import Image
import msgpack_numpy

from utils.logger import logger
from utils.vision import VisionClient
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.utils.common import append_text_to_image, images_to_video

from src.common.param import args
from src.vlnce_src.env import AirVLNLLMENV
from src.common.cognitive_agent import CognitiveAgentConfig, CognitiveAgent

def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    frame = egocentric_view

    return frame

def generate_video(
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    llm: str,
    metrics: Dict[str, float],
    fps: int = 5,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        llm: llm for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")
    
    llm = llm.replace('/', '-')

    video_name = f"{episode_id}-{llm}-" + "-".join(metric_strs)
    images_to_video(images, video_dir, video_name, fps=fps)

def observations_to_image(observation: Dict, draw_depth: bool = False) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb[...,[2,1,0]])

    # draw depth map if observation has depth info. resize to rgb size.
    if draw_depth and ("depth" in observation):
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    frame = egocentric_view

    return frame

def setup():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

def batch_obs(
    observations: List[DictTree]
):
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(obs[sensor])

    return batch
    
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def eval_vlnce():
    logger.info(args)

    _eval_checkpoint(
        llm=args.EVAL_LLM,
    )
    logger.info("END evaluate")


def _eval_checkpoint(
    llm: str
) -> None:
    logger.info(f"LLM: {llm}")

    train_env = AirVLNLLMENV(batch_size=args.batchSize, split=args.EVAL_DATASET)

    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'output/logs/{}/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{llm.replace('/', '-')}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return

    agent_config = CognitiveAgentConfig()
    agent_config.attention.model = args.EVAL_LLM
    agent_config.memory.model = args.EVAL_LLM
    agent_config.imagery.model = args.EVAL_LLM
    agent_config.problem_solving.model = args.EVAL_LLM
    agent_config.reasoning.model = args.EVAL_LLM
    agent_config.decision_making.model = args.EVAL_LLM
    agent_config.perception.model = args.EVAL_LLM
    agent_config.perception.vlm_model = args.EVAL_VLM
    trainer = CognitiveAgent(agent_config)

    gc.collect()

    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break
            finisheds = [[] for _ in range(train_env.batch_size)]

            rgb_frames = [[] for _ in range(train_env.batch_size)]
            rgb_depth_frames = [[] for _ in range(train_env.batch_size)]

            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]

            outputs = train_env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations)

            print(f'Batch: {batch["instruction"]}')
            
            if args.SAVE_LOG:
                SAVE_LOG_DIR = Path(args.project_prefix) / 'output/logs/{}/{}/{}'.format(args.name, args.make_dir_time, llm.replace('/', '-'))

                print(f'Episode ID: {train_env.batch[0]["episode_id"]}')

                if not os.path.exists(str(SAVE_LOG_DIR / train_env.batch[0]['episode_id'])):
                    os.makedirs(str(SAVE_LOG_DIR / train_env.batch[0]['episode_id']), exist_ok=True)

                SAVE_LOG_FOLDER = Path(SAVE_LOG_DIR) / train_env.batch[0]['episode_id']
            else:
                SAVE_LOG_DIR = None
                SAVE_LOG_FOLDER = None

            ended = False

            if args.SAVE_LOG:
                trainer.preprocess(batch, SAVE_LOG_FOLDER)
            else:
                trainer.preprocess(batch)

            for t in range(int(args.maxAction)):
                logger.info('llm:{} \t {} - {} / {}'.format(llm, idx, t, end_iter, ))

                ## Predict Next Action 
                actions, finished = trainer.act(
                    batch,
                    step=t
                )
                for i, finish in enumerate(finished):
                    finisheds[i].append(finish)

                # Make action and get the new state
                # actions = [temp[0] for temp in actions.numpy()]
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations)

                logger.info('action: {}'.format(actions))

                # reset envs and observations if necessary
                for i in range(train_env.batch_size):
                    if args.EVAL_GENERATE_VIDEO:
                        rgb_depth_frame = observations_to_image(observations[i], infos[i])
                        rgb_depth_frame = append_text_to_image(
                            rgb_depth_frame, train_env.batch[i]['instruction']['instruction_text']
                        )
                        rgb_depth_frames[i].append(rgb_depth_frame)
                        
                        rgb_frame = observations_to_image(observations[i], False)
                        rgb_frames[i].append(rgb_frame)

                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True
                    pbar.update()

                if np.array(dones).all():
                    ended = True
                    break

            for t in range(int(train_env.batch_size)):
                infos[t]['finished'] = finisheds[t]
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                EVAL_SAVE_EVERY_RESULTS_DIR = SAVE_LOG_FOLDER
                f_intermediate_result_name = os.path.join(
                    EVAL_SAVE_EVERY_RESULTS_DIR,
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                if args.EVAL_GENERATE_VIDEO:
                    EVAL_GENERATE_VIDEO_DIR = SAVE_LOG_FOLDER
                    generate_video(
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=rgb_depth_frames[t],
                        episode_id=train_env.batch[t]['episode_id'],
                        llm=llm,
                        metrics={
                            # "spl": infos[t]['spl'],
                            "ndtw": infos[t]['ndtw'],
                        }
                    )

                logger.info((
                    'result-{} \t' +
                    'distance_to_goal: {} \t' +
                    'success: {} \t' +
                    'ndtw: {} \t' +
                    'sdtw: {} \t' +
                    'path_length: {} \t' +
                    'oracle_success: {} \t' +
                    'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['success'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                ))

    # end
    pbar.close()


    EVAL_INTERMEDIATE_RESULTS_DIR = SAVE_LOG_DIR
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_llm_{llm.replace('/', '-')}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    #
    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values())
            / num_episodes
        )

    #
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_llm_{llm.replace('/', '-')}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


if __name__ == "__main__":
    setup()
    eval_vlnce()