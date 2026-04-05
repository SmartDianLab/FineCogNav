import argparse
import os
import datetime
from pathlib import Path
from utils.CN import CN
import multiprocessing


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        project_prefix = Path(str(os.getcwd())).parent.resolve()
        self.parser.add_argument('--project_prefix', type=str, default=str(project_prefix), help="project path")

        self.parser.add_argument('--name', type=str, default='default', help='experiment name')

        self.parser.add_argument('--maxInput', type=int, default=300, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=500, help='max action sequence')

        self.parser.add_argument('--batchSize', type=int, default=1)

        self.parser.add_argument('--Image_Height_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Width_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Height_DEPTH', type=int, default=256)
        self.parser.add_argument('--Image_Width_DEPTH', type=int, default=256)

        self.parser.add_argument('--ablate_instruction', action="store_true")
        self.parser.add_argument('--ablate_rgb', action="store_true")
        self.parser.add_argument('--ablate_depth', action="store_true")

        self.parser.add_argument('--EVAL_DATASET', type=str, default="val_unseen")
        self.parser.add_argument('--EVAL_LLM', type=str)
        self.parser.add_argument('--EVAL_VLM', type=str, default="qwen2.5-vl-72b-instruct")
        self.parser.add_argument("--EVAL_NUM", type=int, default=-1)
        self.parser.add_argument('--EVAL_GENERATE_VIDEO', action="store_true")
        self.parser.add_argument('--SAVE_LOG', action="store_true")

        self.parser.add_argument("--simulator_tool_port", type=int, default=30000, help="simulator_tool port")

        self.args = self.parser.parse_args()


param = Param()
args = param.args

if multiprocessing.current_process().name == "MainProcess":
    args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    os.environ['MAKE_DIR_TIME'] = str(args.make_dir_time)
else:
    args.make_dir_time = os.environ['MAKE_DIR_TIME']
args.logger_file_name = '{}/output/logs/{}/{}/{}.log'.format(args.project_prefix, args.name, args.make_dir_time, args.name)
args.log_dir = '{}/output/logs/{}/{}/'.format(args.project_prefix, args.name, args.make_dir_time)



args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
    },
]

default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()

