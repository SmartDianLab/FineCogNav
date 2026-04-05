# FineCog-Nav: 

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

根据 [GitHub Issue](https://github.com/microsoft/AirSim/issues/3333) 修改 `airism`安装中的 `client.py`:

```bash
pip show airsim
# locate the "Location: <path>" field
vim <path>/airsim/client.py
```

更新文件:

```python
class VehicleClient:
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        if (ip == ""):
            ip = "127.0.0.1"
        # self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value, pack_encoding = 'utf-8', unpack_encoding = 'utf-8')
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value)
```

如果你的服务器没有显示设备, 你可能需要安装这些包: 

```bash
sudo apt install xdg-user-dirs xdg-utils
sudo apt install libegl1
sudo apt install vulkan-tools libvulkan1 mesa-vulkan-drivers
```

### Simulator & Datasets

请参照 [AerialVLN](https://github.com/AirVLN/AirVLN) 来下载模拟器.

创建或修改AirSim配置文件: `~/Documents/AirSim/settings.json`, 确保`PhysicsEngineName`是 `ExternalPhysicsEngine`.

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

### 目录结构

你的项目目录应该长这样: 

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

## 🔧 **使用示例**

首先在终端启动模拟器: 

```bash
# make sure you are in FineCogNav/
python -u ./airsim_plugin/AirVLNSimulatorServerTool.py \
    --port 30001 \
    --gpus 0
```

你可以选择是否添加`--onscreen`来使模拟器可视化.

---

然后在另一个终端运行以下脚本: 

```
# make sure you are in FineCogNav/
# ensure your API keys for LLM and VLM are set
bash ./scripts/eval_llm.sh \
    qwen3.5-397b-a17b \
    qwen3.5-397b-a17b \
    rl_4 \
    30001
```

参数:
+ LLM: LLM 模型名称
+ VLM: VLM 模型名称
+ 数据集: `./DATA/data/aerialvln/`路径下的JSON文件名
+ 端口号: 模拟器通信端口

---

如果需要多组实验并行, 使用不同的端口号重复上述步骤. 

⚠️ 请确保不同实验间的端口号相差大于2.

## 📚 **Evaluation**

TODO

## 📜 **引用**
如果您在研究中使用了FineCog-Nav请引用以下文献:

```
coming soon
```

## 🥰 **致谢**
* 部分组件修改自 [AerialVLN](https://github.com/AirVLN/AirVLN). 衷心感谢.