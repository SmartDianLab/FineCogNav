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

## 📚 **评估说明**

评估脚本 (`scripts/eval.py`) 用于评估智能体的轨迹。

```python
python eval.py <ref_type> <pred_type> <gt_folder> <pred_path>
```

其中两个主要参数 `<ref_type>` 和 `<pred_type>` 分别定义了真值（ground truth）数据和预测（prediction）数据的格式。

### 参数选项

*   **`<ref_type>`**：指定真值（参考）数据的结构。
    *   `1`：**完整层次（Full-level）**。参考数据为整个片段（episode）提供单一的路径和目标。
    *   `2`：**句子层次（Sentence-level）**。参考数据被分解为一系列句子，每个句子都有自己的子目标和路径片段。

*   **`<pred_type>`**：指定预测数据的结构。
    *   `1`：**单个 JSON 文件（字典格式）**。所有预测结果都存放在一个字典形式的 JSON 文件中。
    *   `2`：**JSON 文件夹（多字典格式）**。预测结果分散在目录中的多个 JSON 文件里，每个文件都是一个字典。
    *   `3`：**单个 JSON 文件（单片段）**。该文件只包含一个片段的预测结果。
    *   `4`：**JSON 文件夹（单片段文件）**。目录中的每个 JSON 文件都包含一个片段的预测结果。

### 期望的文件/文件夹结构

#### 当 `<ref_type>` = 1 (完整层次)
真值文件夹应包含一个或多个 JSON 文件。每个文件必须包含一个 `episodes` 列表，其中每个片段都包含 `reference_path` 和 `goals` 列表。

```
gt_folder/
└── gt_data.json
```


#### 当 `<ref_type>` = 2 (句子层次)
真值文件夹应包含一个或多个 JSON 文件。每个文件代表一个单独的片段，并且必须包含一个 `sentence_instructions` 列表。该列表中的每条指令都应包含 `end_position` 和 `reference_path`。

```
gt_folder/
├── episode_001.json
├── episode_002.json
└── ...
```


#### 当 `<pred_type>` = 1 (单字典文件)
预测路径是一个单独的 JSON 文件，其中包含一个字典，每个键（key）对应一个片段 ID。

```
pred_path.json
```


#### 当 `<pred_type>` = 2 (多字典文件夹)
预测路径是一个文件夹，里面包含多个 JSON 文件，每个文件都是一个片段字典。

```
pred_folder/
├── preds_part1.json
├── preds_part2.json
└── ...
```


#### 当 `<pred_type>` = 3 或 4 (单文件或文件夹)
预测路径可以是单个片段的 JSON 文件 (`3`)，也可以是一个文件夹，其中每个 JSON 文件对应一个片段的预测结果 (`4`)。

**类型 3 的结构：**
```
single_pred.json
```

**类型 4 的结构：**
```
pred_folder/
├── episode_001.json
├── episode_002.json
└── ...
```

## 📜 **引用**
如果您在研究中使用了FineCog-Nav请引用以下文献:

```
coming soon
```

## 🥰 **致谢**
* 部分组件修改自 [AerialVLN](https://github.com/AirVLN/AirVLN). 衷心感谢.
