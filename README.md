# FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation

## 🛠️ **Getting Started**

### Install Dependencies

```bash
conda env create -f environment.yaml
pip install airsim
pip uninstall msgpack
pip uninstall msgpack-python
pip uninstall msgpack-rpc-python
pip install msgpack
pip install -U git+https://github.com/tbelhalfaoui/msgpack-rpc-python.git@fix-msgpack-dep
```

Then follow this [GitHub Issue](https://github.com/microsoft/AirSim/issues/3333) to modify the `client.py` in your `airsim` installation:

```bash
pip show airsim
# locate the "Location: <path>" field
vim <path>/airsim/client.py
```

Update the file as follows:

```python
class VehicleClient:
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        if (ip == ""):
            ip = "127.0.0.1"
        # self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value, pack_encoding = 'utf-8', unpack_encoding = 'utf-8')
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value)
```

If your server does not have a display device, you may need to install the following packages:

```bash
sudo apt install xdg-user-dirs xdg-utils
sudo apt install libegl1
sudo apt install vulkan-tools libvulkan1 mesa-vulkan-drivers
```

### Simulator & Datasets

Please follow [AerialVLN](https://github.com/AirVLN/AirVLN) to download Simulator files.

Create or modify the AirSim configuration file at: `~/Documents/AirSim/settings.json`, make sure `PhysicsEngineName` is `ExternalPhysicsEngine`.

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
    "CameraDefaults": {
      "CaptureSettings": [
        {
          "ImageType": 0,
          "Width": 1280,
          "Height": 720,
          "FOV_Degrees": 90,
          "AutoExposureSpeed": 100,
          "MotionBlurAmount": 0
        }
    ]
  },
  "SettingsVersion": 1.2,
  "PhysicsEngineName": "ExternalPhysicsEngine",
  "SimMode": "Multirotor"
}
```

### Directory Structure

Your project directory should be organized as follows:

```
.
├── DATA
│   └── data
│       └── aerialvln
│           └── ...
├── ENVs
│   └── ENVs
│   │   ├── env_1
│   │   │   ├── ...
│   │   ├── ...
└── FineCogNav
    ├── airsim_plugin
    ├── ...
    └── README.md
```

## 🔧 **Example Usage**

First, launch the simulator in one terminal: 

```bash
# make sure you are in FineCogNav/
python -u ./airsim_plugin/AirVLNSimulatorServerTool.py \
    --port 30001 \
    --gpus 0
```

You can optionally add --onscreen to enable visualization.

---

Then, run the script in another terminal:

```
# make sure you are in FineCogNav/
# ensure your API keys for LLM and VLM are set
bash ./scripts/eval_llm.sh \
    qwen3.5-397b-a17b \
    qwen3.5-397b-a17b \
    rl_4 \
    30001
```

Arguments:
+ LLM: Large Language Model name
+ VLM: Vision-Language Model name
+ DataSplit: JSON filename under `./DATA/data/aerialvln/`
+ Port: Simulator port

---

To run multiple experiments in parallel, repeat the above steps with different ports.

⚠️ Make sure the port difference between instances is greater than 2.

## 📚 **Evaluation**

Evaluation Script (`scripts/eval.py`) evaluates agent trajectories. 

```python
python eval.py <ref_type> <pred_type> <gt_folder> <pred_path>
```

The two main arguments, `<ref_type>` and `<pred_type>`, define the format of your ground truth and prediction data.

### Options

*   **`<ref_type>`**: Specifies the structure of the ground truth (reference) data.
    *   `1`: **Full-level**. The reference data contains a single path and goal for the entire episode.
    *   `2`: **Sentence-level**. The reference data is broken down into a sequence of sentences, each with its own sub-goal and path segment.

*   **`<pred_type>`**: Specifies the structure of the prediction data.
    *   `1`: **Single JSON file (dict)**. All predictions are in one dictionary-style JSON file.
    *   `2`: **Folder of JSON files (multi-dict)**. The predictions are split across multiple JSON files within a directory.
    *   `3`: **Single JSON file (single)**. The file contains the prediction for a single episode.
    *   `4`: **Folder of JSON files (folder)**. Each JSON file in the directory contains a prediction for one episode.

### Expected Folder/File Structure

#### For `<ref_type>` = 1 (Full-level)
The ground truth folder should contain one or more JSON files. Each file must have an `episodes` list, where each episode has a `reference_path` and a `goals` list.

```
gt_folder/
└── gt_data.json
```


#### For `<ref_type>` = 2 (Sentence-level)
The ground truth folder should contain one or more JSON files. Each file represents a single episode and must have a `sentence_instructions` list. Each instruction in this list should contain an `end_position` and a `reference_path`.

```
gt_folder/
├── episode_001.json
├── episode_002.json
└── ...
```


#### For `<pred_type>` = 1 (Single dict)
The prediction path is a single JSON file containing a dictionary where each key is an episode ID.

```
pred_path.json
```


#### For `<pred_type>` = 2 (Multi-dict folder)
The prediction path is a folder containing multiple JSON files, each structured as a dictionary of episodes.

```
pred_folder/
├── preds_part1.json
├── preds_part2.json
└── ...
```


#### For `<pred_type>` = 3 or 4 (Single file or Folder)
The prediction path can be either a single JSON file for one episode (`3`) or a folder where each JSON file is a single episode's prediction (`4`).

**For type 3:**
```
single_pred.json
```

**For type 4:**
```
pred_folder/
├── episode_001.json
├── episode_002.json
└── ...
```

## 📜 **Citing**
If you use FineCog-Nav in your research, please cite the following paper:

```
@misc{shao2026finecognavintegratingfinegrainedcognitive,
      title={FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation}, 
      author={Dian Shao and Zhengzheng Xu and Peiyang Wang and Like Liu and Yule Wang and Jieqi Shi and Jing Huo},
      year={2026},
      eprint={2604.16298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.16298}, 
}
```

## 🥰 **Acknowledgement**
* Some components modified from [AerialVLN](https://github.com/AirVLN/AirVLN). Thanks sincerely.
