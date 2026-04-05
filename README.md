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

TODO

## 📜 **Citing**
If you use FineCog-Nav in your research, please cite the following paper:

```
coming soon
```

## 🥰 **Acknowledgement**
* Some components modified from [AerialVLN](https://github.com/AirVLN/AirVLN). Thanks sincerely.