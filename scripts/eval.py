# 0404
# 1.Full-level去除计算NE-A，Sentence-level去除计算NE。
# 2.Full-level新增计算explore-ratio，计算方法为参考轨迹的起终点距离/参考轨迹的轨迹长度。 
# 3.新增一组全局指标，命名为探索度分组指标，使用explore-ratio来将轨迹分为[0,0.4] (0.4,0.6] (0.6,1.0]三组，分别称作explore-group-1/2/3，并计算、记录三个全局平均值。
# 0406
# 新增多阈值的SR和OSR指标，分别为20/30/40米的成功率和Oracle成功率
# 0408
# 新增适配原始参考轨迹，仅计算full-level指标。
# 0409
# 1. 添加json输出，修正为嵌套结构。
# 2. 打印评估统计信息。
# 3.为了便于分析，输出sentence-level的NE
# 0410
# 新增二维的SR、OSR、NE指标，计算方法为去掉z轴的欧氏距离。
# 0616
# 新增Explore_Rate指标，计算方法为预测动作数/参考轨迹动作数。 
# 0618
# 1. 新增5/10米的SR和OSR指标。
# 2. 新增Min Distance，轨迹任意一点和目标点之间的最小距离。
# 3. 新增考虑碰撞的SR、OSR指标，碰撞时SR/OSR为0，仅考虑了阈值为15、20。
# 0619
# 1.新增去重处理，按优先级保留最佳轨迹：优先SR_3D-20高，其次OSR_3D-20高，最后NE_3D低。
# 2.新增递归遍历子文件夹功能，支持从嵌套文件夹中加载数据。

import json
import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import re
from collections import defaultdict
from tqdm import tqdm

# 选择配对重采样方法：'signflip' | 'bootstrap' | 'paired-permutation'
# 注意：'perm'/'permutation' 简写也会被识别为 'paired-permutation'
PAIR_RESAMPLE_METHOD = 'bootstrap'

SAVE_DEDUP_JSON = True  # 是否保存去重后择优轨迹json

# 模式2（ref_type==2）分类工作表层级：'sentence'（按 Seg_k 单句）或 'full'（按 Full 完整轨迹）
# 仅当提供 --segcat 时生效
CAT_SHEET_LEVEL_MODE2 = 'full'  # 可选: 'sentence' | 'full'

# ================= 工具与辅助 =================
def _sanitize_sheet_name(name: str) -> str:
    """清理Excel工作表名，最长31字符，去除非法字符。"""
    if not isinstance(name, str):
        name = str(name)
    # 替换非法字符
    name = re.sub(r'[\\/*?\[\]:]', '_', name)
    # 过长截断
    if len(name) > 31:
        name = name[:31]
    # 避免空名
    return name or "Sheet"

def _parse_cli_with_flags(argv):
    """解析命令行，支持可选的 --segcat/--seg-cat/--seg-category-json。
    返回 (positional_args, options)；positional_args 不包含这些flag。
    允许两种形式：--segcat=path 或 --segcat path
    """
    options = {}
    positional = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if isinstance(arg, str) and arg.startswith('--'):
            key = None
            if arg.startswith('--segcat='):
                options['segcat'] = arg.split('=', 1)[1].strip()
            elif arg in ('--segcat', '--seg-cat', '--seg-category-json'):
                if i + 1 < len(argv):
                    options['segcat'] = argv[i+1]
                    i += 1
            else:
                # 未识别的flag，保留为位置参数以避免破坏旧用法
                positional.append(arg)
            i += 1
            continue
        positional.append(arg)
        i += 1
    return positional, options

def load_seg_categories(json_path):
    """加载多标签单句分类JSON，返回 {(episode_id:str, seg_idx:int): set(categories)} 映射。
    兼容多种字段命名：
    - episode: 'episode_id' | 'episodeId' | 'ep_id'
    - segment index: 'seg_idx' | 'segment_index' | 'sentence_index' | 'instruction_index'
    - categories: 'categories' | 'labels' | 'label' | 'tags'
    若 labels 为字典，则取 value 为 True 的键或转为列表。
    """
    mapping = {}
    if not json_path or not os.path.exists(json_path):
        return mapping
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load seg-category JSON: {e}")
        return mapping

    # 可能包裹在 data 列表下
    items = None
    if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
        items = data['data']
    elif isinstance(data, list):
        items = data
    else:
        # 直接当作单条或未知结构
        items = []

    def _norm_ep(v):
        try:
            return str(v)
        except Exception:
            return None

    for it in items:
        if not isinstance(it, dict):
            continue
        # episode id
        ep = it.get('episode_id')
        if ep is None:
            ep = it.get('episodeId', it.get('ep_id'))
        ep = _norm_ep(ep)
        # segment index
        seg_idx = (
            it.get('seg_idx') if 'seg_idx' in it else (
                it.get('segment_index', it.get('sentence_index', it.get('instruction_index', it.get('sentence_id'))))
            )
        )
        try:
            seg_idx = int(seg_idx) if seg_idx is not None else None
        except Exception:
            seg_idx = None

        # categories
        cats = None
        if 'categories' in it:
            cats = it['categories']
        elif 'tags' in it:
            cats = it['tags']
        elif 'label' in it:
            cats = it['label']
        elif 'labels' in it:
            labs = it['labels']
            if isinstance(labs, dict):
                cats = [k for k, v in labs.items() if v]
            else:
                cats = labs
        # 归一为列表
        if cats is None:
            cats = []
        elif isinstance(cats, str):
            cats = [cats]
        elif not isinstance(cats, (list, tuple, set)):
            cats = [str(cats)]
        cats = [str(c) for c in cats if c is not None]

        if ep is None or seg_idx is None:
            continue
        key = (ep, seg_idx)
        mapping.setdefault(key, set()).update(cats)

    return mapping

# ================= 统计与Bootstrap工具 =================
def _filter_valid(values):
    """过滤掉None/NaN/Inf，返回numpy数组。"""
    valid = []
    for v in values:
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if not (np.isnan(vf) or np.isinf(vf)):
            valid.append(vf)
    return np.array(valid, dtype=float)

def bootstrap_stats(values, n_boot=2000, ci=0.95, random_state=42):
    """对给定样本做非参数Bootstrap，返回点估计、样本标准差、Bootstrap标准误、置信区间。

    参数:
        values: 可迭代的数值（将自动清洗None/NaN/Inf）
        n_boot: bootstrap次数（默认2000）
        ci: 置信水平（默认0.95）
        random_state: 随机种子
    返回:
        dict: { 'n','mean','std','se_boot','ci_low','ci_high','ci_level' }
    """
    x = _filter_valid(values)
    if x.size == 0:
        return {
            'n': 0, 'mean': None, 'std': None, 'se_boot': None,
            'ci_low': None, 'ci_high': None, 'ci_level': ci
        }
    rng = np.random.default_rng(random_state)
    point_mean = float(np.mean(x))
    sample_std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = np.mean(x[idx], axis=1)
    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1 - alpha])
    se_boot = float(np.std(boot_means, ddof=1))
    return {
        'n': int(x.size),
        'mean': point_mean,
        'std': sample_std,
        'se_boot': se_boot,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'ci_level': ci
    }

def bootstrap_one_sample_p(values, null_value, n_boot=2000, random_state=42):
    """一样本双侧Bootstrap检验p值: H0: E[X] = null_value；HA: E[X] != null_value"""
    x = _filter_valid(values)
    if x.size == 0 or null_value is None:
        return None
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = np.mean(x[idx], axis=1)
    p_left = np.mean(boot_means <= null_value)
    p_right = np.mean(boot_means >= null_value)
    p_two = 2 * min(p_left, p_right)
    return float(min(1.0, max(0.0, p_two)))

def paired_signflip_p(diffs, n_perm=2000, random_state=42):
    """成对样本的符号翻转检验（双侧）p值（按分布相对 0 的双尾概率）。
    输入 diffs 为每个 episode 的成对差 A-B（已清洗）。
    H0: E[diff] = 0；HA: != 0。
    p = 2 * min(Pr(Δ* > 0), Pr(Δ* < 0))，Δ* 为符号翻转生成的均值分布。
    """
    d = _filter_valid(diffs)
    if d.size == 0:
        return None
    rng = np.random.default_rng(random_state)
    # 生成 Rademacher 符号矩阵 {-1, +1}
    signs = rng.integers(0, 2, size=(n_perm, d.size)) * 2 - 1
    perm_means = np.mean(signs * d, axis=1)
    p_left = np.mean(perm_means < 0)
    p_right = np.mean(perm_means > 0)
    p_two = 2 * min(p_left, p_right)
    return float(min(1.0, max(0.0, p_two)))

def paired_bootstrap_p(diffs, n_boot=2000, random_state=42):
    """成对样本的 Bootstrap p 值（双侧），对差分样本有放回重采样，取均值分布 Δ*。
    p = 2 * min(Pr(Δ* > 0), Pr(Δ* < 0))。
    """
    d = _filter_valid(diffs)
    if d.size == 0:
        return None
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot_means = np.mean(d[idx], axis=1)
    p_left = np.mean(boot_means < 0)
    p_right = np.mean(boot_means > 0)
    p_two = 2 * min(p_left, p_right)
    return float(min(1.0, max(0.0, p_two)))

def paired_permutation_p(a_vals, b_vals, n_perm=2000, random_state=42):
    """配对置换检验（双侧）：每对中以 0.5 概率交换 A 与 B，得到均值差分布 Δ*。
    p = 2 * min(Pr(Δ* > 0), Pr(Δ* < 0))。
    注意：在“差值法”视角下，此检验与符号翻转等价；此处显式以对内交换实现。
    """
    a = _filter_valid(a_vals)
    b = _filter_valid(b_vals)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    rng = np.random.default_rng(random_state)
    n = a.size
    perm_means = []
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=n)
        a_perm = np.where(swap == 1, b, a)
        b_perm = np.where(swap == 1, a, b)
        perm_means.append(float(np.mean(a_perm - b_perm)))
    perm_means = np.array(perm_means, dtype=float)
    p_left = np.mean(perm_means < 0)
    p_right = np.mean(perm_means > 0)
    p_two = 2 * min(p_left, p_right)
    return float(min(1.0, max(0.0, p_two)))

def safe_nanmean(values):
    """安全的计算平均值，过滤掉None、NaN和无穷大值"""
    valid = []
    for val in values:
        if val is not None and not np.isnan(val) and not np.isinf(val):
            valid.append(val)
    return np.nanmean(valid) if valid else None

def load_groundtruth_type2(gt_folder):
    """使用Fine标注参考轨迹，加载文件夹中的所有原始轨迹数据（支持递归遍历子文件夹），兼容多种结构"""
    gt_data = {}
    for root, dirs, files in os.walk(gt_folder):
        for filename in files:
            if filename.endswith('.json'):
                path = os.path.join(root, filename)
                try:
                    with open(path, encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            if 'episode_id' in data:
                                gt_data[data['episode_id']] = data
                            elif 'episodes' in data and isinstance(data['episodes'], list):
                                for episode in data['episodes']:
                                    ep_id = episode.get('episode_id')
                                    if ep_id is not None:
                                        gt_data[ep_id] = episode
                                    else:
                                        print(f"Warning: episode missing episode_id in {path}")
                            else:
                                print(f"Warning: Unrecognized gt json structure in {path}")
                        else:
                            print(f"Warning: gt json is not dict in {path}")
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
    return gt_data

def load_reference_trajectory_type1(gt_folder):
    """使用原始参考轨迹，加载参考轨迹数据（包含多个episodes的JSON文件，支持递归遍历子文件夹）"""
    gt_data = {}
    for root, dirs, files in os.walk(gt_folder):
        for filename in files:
            if filename.endswith('.json'):
                path = os.path.join(root, filename)
                try:
                    with open(path, encoding='utf-8') as f:
                        data = json.load(f)
                        for episode in data.get('episodes', []):
                            ep_id = episode['episode_id']
                            gt_data[ep_id] = episode
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
    return gt_data

def load_predictions(mode, pred_path):
    """根据模式加载预测数据，支持同episode_id多条轨迹"""
    if mode == 1:  # 字典格式包含多个轨迹
        with open(pred_path, encoding='utf-8') as f:
            data = json.load(f)
            pred_data = defaultdict(list)
            for k, v in data.items():
                ep_id = v.get('episode_id', k)
                pred_data[ep_id].append(v)
            return pred_data
    elif mode == 2:  # 多个字典文件目录模式
        pred_data = defaultdict(list)
        for root, dirs, files in os.walk(pred_path):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, encoding='utf-8') as f:
                            data = json.load(f)
                            for k, v in data.items():
                                ep_id = v.get('episode_id', k)
                                pred_data[ep_id].append(v)
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
        return pred_data
    elif mode == 3:  # 单个轨迹文件模式
        with open(pred_path, encoding='utf-8') as f:
            v = json.load(f)
            ep_id = v.get('episode_id', os.path.basename(pred_path).split('.')[0])
            pred_data = defaultdict(list)
            pred_data[ep_id].append(v)
            return pred_data
    elif mode == 4:  # 文件夹模式
        return load_prediction_folder(pred_path)
    else:
        raise ValueError("Invalid mode. Use 1 (dict), 2 (multi-dict), 3 (single) or 4 (folder)")

def load_prediction_folder(folder_path):
    """加载整个预测文件夹（支持递归遍历子文件夹），支持同episode_id多条轨迹"""
    pred_data = defaultdict(list)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, encoding='utf-8') as f:
                        data = json.load(f)
                        ep_id = data.get('episode_id', filename.split('.')[0])
                        pred_data[ep_id].append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    return pred_data

def split_segments(finished):
    """根据finished标记分割轨迹"""
    if len(finished) == 1:
        # 如果预测轨迹只有一个点，返回整个轨迹作为一个段
        return [(0, 0)]
    
    indices = [i for i, val in enumerate(finished) if val]
    segments = []
    start = 0
    for end in indices:
        segments.append((start, end))
        # 让轨迹分段重叠
        start = end
    if start < len(finished) - 1:  # 处理最后一段，同时避免最后单个点的情况
        segments.append((start, len(finished) - 1))
    return segments

# ========== 基础评估指标 ==========

def calculate_pl(points):
    """计算路径长度"""
    pl = 0.0
    for i in range(1, len(points)):
        current = points[i-1][:3]
        next_point = points[i][:3]
        pl += np.linalg.norm(np.array(next_point) - np.array(current))
    return pl

def calculate_sr(points, goal, success_distance=20.0):
    """计算成功率"""
    if not points:
        return 0.0
    final_point = np.array(points[-1][:3])
    goal = np.array(goal[:3])
    distance = np.linalg.norm(final_point - goal)
    return 1.0 if distance <= success_distance else 0.0

def calculate_sr_2d(points, goal, success_distance=20.0):
    """计算二维成功率"""
    if not points:
        return 0.0
    final_point = np.array(points[-1][:3])
    goal = np.array(goal[:3])
    distance = np.linalg.norm(final_point[:2] - goal[:2])
    return 1.0 if distance <= success_distance else 0.0

def calculate_osr(points, goal, success_distance=20.0):
    """计算Oracle成功率"""
    if not points:
        return 0.0
    points = [np.array(p[:3]) for p in points]
    goal = np.array(goal[:3])
    distances = [np.linalg.norm(p - goal) for p in points]
    return 1.0 if min(distances) <= success_distance else 0.0

def calculate_osr_2d(points, goal, success_distance=20.0):
    """计算二维Oracle成功率"""
    if not points:
        return 0.0
    points = [np.array(p[:3]) for p in points]
    goal = np.array(goal[:3])
    distances = [np.linalg.norm(p[:2] - goal[:2]) for p in points]
    return 1.0 if min(distances) <= success_distance else 0.0

def calculate_pc(pred_points, ref_points, d_th=20.0):
    """计算路径覆盖度"""
    if not pred_points or not ref_points:
        return 0.0
    pred_coords = np.array([p[:3] for p in pred_points])
    ref_coords = np.array([r[:3] for r in ref_points])
    distances = cdist(ref_coords, pred_coords)
    min_distances = np.min(distances, axis=1)
    return np.mean(np.exp(-min_distances / d_th))

def calculate_cls(pred_points, ref_points, d_th=20.0):
    """计算覆盖分数"""
    pc = calculate_pc(pred_points, ref_points, d_th)
    if pc == 0:
        return 0.0
    pl_ref = calculate_pl(ref_points)
    pl_pred = calculate_pl(pred_points)
    ls = (pc * pl_ref) / (pc * pl_ref + abs(pc * pl_ref - pl_pred))
    return pc * ls

def calculate_spl(sr, ref_pl, pred_pl):
    """计算路径加权成功率"""
    if ref_pl == 0 or pred_pl == 0:
        return 0.0
    return sr * (ref_pl / max(pred_pl, ref_pl))

# ========== 三维评估指标 ==========

def calculate_distance(p1, p2, weights=(1.0, 1.0, 1.0)):
    """三维加权欧氏距离计算"""
    dx = (p1[0] - p2[0]) * weights[0]
    dy = (p1[1] - p2[1]) * weights[1]
    dz = (p1[2] - p2[2]) * weights[2]
    return np.sqrt(dx**2 + dy**2 + dz**2)

# 三维nDTW
def calculate_ndtw_3d(pred_path, ref_path, d_h=20.0, z_weight=1.0):
    """三维归一化动态时间规整"""
    if not pred_path or not ref_path:
        return 0.0
    
    # 提取三维坐标并加权
    pred_coords = [p[:3] for p in pred_path]
    ref_coords = [r[:3] for r in ref_path]
    
    # 调整z轴权重
    weights = (1.0, 1.0, z_weight)
    
    try:
        distance, _ = fastdtw(ref_coords, pred_coords, dist=lambda x,y: calculate_distance(x,y,weights))
        return np.exp(-distance / (len(ref_coords) * d_h))
    except:
        return 0.0
    
def calculate_sdtw_3d(ne, ndtw, d_th=20.0):
    """三维加权成功率nDTW"""
    return ndtw if ne <= d_th else 0.0

def extract_sentence_endpoints(trajectory, finished):
    """从轨迹中提取每个句子分段的终点坐标"""
    segments = split_segments(finished)
    endpoints = []
    for start, end in segments:
        if end < len(trajectory):
            endpoint = trajectory[end][:3]
            endpoints.append(endpoint)
    return endpoints

def calculate_sentence_ndtw_3d(pred_path, ref_path, pred_finished, ref_finished, d_h=20.0, z_weight=1.0):
    """计算基于句子终点的nDTW
    
    Args:
        pred_path: 预测轨迹点列表
        ref_path: 参考轨迹点列表
        pred_finished: 预测轨迹的finished标记列表
        ref_finished: 参考轨迹的finished标记列表
        d_h: DTW的标准化参数
        z_weight: z轴权重
    """
    if not pred_path or not ref_path:
        return 0.0
        
    # 提取句子终点
    pred_endpoints = extract_sentence_endpoints(pred_path, pred_finished)
    ref_endpoints = extract_sentence_endpoints(ref_path, ref_finished)
    
    if not pred_endpoints or not ref_endpoints:
        return 0.0
    
    # 转换为numpy数组
    ref_coords = np.array(ref_endpoints)
    pred_coords = np.array(pred_endpoints)
    
    # 设置坐标权重
    weights = np.array([1.0, 1.0, z_weight])
    
    def calculate_distance(p1, p2, weights):
        return np.sqrt(np.sum(((p1 - p2) * weights) ** 2))
    
    # 计算DTW距离
    distance, _ = fastdtw(ref_coords, pred_coords, dist=lambda x,y: calculate_distance(x,y,weights))
    
    # 计算轨迹长度（用于归一化）
    max_len = len(ref_endpoints)
    
    # 计算nDTW
    ndtw = np.exp(-distance / (d_h * max_len))
    return ndtw

# 三维双向nDTW
def bidirectional_ndtw_3d(pred_path, ref_path, d_h=20.0, z_weight=1.0):
    forward = calculate_ndtw_3d(pred_path, ref_path, d_h, z_weight)
    backward = calculate_ndtw_3d(ref_path, pred_path, d_h, z_weight)
    return (forward + backward) / 2

# 三维LCSS
def lcss_3d(pred_path, ref_path, delta=2.0, epsilon=5, z_weight=1.0):
    """三维最长公共子序列"""
    pred = np.array([p[:3] for p in pred_path])
    ref = np.array([r[:3] for r in ref_path])
    
    dp = np.zeros((len(pred)+1, len(ref)+1))
    for i in range(1, len(pred)+1):
        for j in range(max(1, i-epsilon), min(len(ref)+1, i+epsilon)):
            if calculate_distance(pred[i-1], ref[j-1], (1,1,z_weight)) <= delta:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1] / max(len(pred), len(ref))

# 三维EDR
def edr_3d(pred_path, ref_path, delta=2.0, z_weight=1.0):
    """三维编辑距离实数序列"""
    pred = np.array([p[:3] for p in pred_path])
    ref = np.array([r[:3] for r in ref_path])
    
    dp = np.zeros((len(pred)+1, len(ref)+1))
    for i in range(len(pred)+1):
        for j in range(len(ref)+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if calculate_distance(pred[i-1], ref[j-1], (1,1,z_weight)) <= delta else 1
                dp[i][j] = min(dp[i-1][j] + 1, 
                             dp[i][j-1] + 1, 
                             dp[i-1][j-1] + cost)
    return 1 - (dp[-1][-1] / max(len(pred), len(ref)))

# 三维ERP
def erp_3d(pred_path, ref_path, delta=2.0, gap_penalty=20.0, z_weight=1.0):
    """三维带实际惩罚的编辑距离"""
    pred = np.array([p[:3] for p in pred_path])
    ref = np.array([r[:3] for r in ref_path])
    gap = np.array([gap_penalty]*3)  # 三维间隙点
    
    dp = np.zeros((len(pred)+1, len(ref)+1))
    for i in range(len(pred)+1):
        for j in range(len(ref)+1):
            if i == 0 and j == 0:
                dp[i][j] = 0
            elif i == 0:
                dp[i][j] = dp[i][j-1] + calculate_distance(ref[j-1], gap, (1,1,z_weight))
            elif j == 0:
                dp[i][j] = dp[i-1][j] + calculate_distance(pred[i-1], gap, (1,1,z_weight))
            else:
                cost = calculate_distance(pred[i-1], ref[j-1], (1,1,z_weight))
                dp[i][j] = min(dp[i-1][j] + calculate_distance(pred[i-1], gap, (1,1,z_weight)),
                             dp[i][j-1] + calculate_distance(ref[j-1], gap, (1,1,z_weight)),
                             dp[i-1][j-1] + cost)
    return np.exp(-dp[-1][-1] / (len(ref) * delta))

# 三维Hausdorff
def hausdorff_3d(pred_path, ref_path, z_weight=1.0):
    """三维Hausdorff距离"""
    pred = np.array([p[:3] for p in pred_path]) if pred_path else np.empty((0,3))
    ref = np.array([r[:3] for r in ref_path]) if ref_path else np.empty((0,3))
    
    if pred.size == 0 or ref.size == 0:
        return 0.0
    
    def _hdistance(A, B):
        d_matrix = np.array([[calculate_distance(a, b, (1,1,z_weight)) for b in B] for a in A])
        return np.max(np.min(d_matrix, axis=1))
    
    return max(_hdistance(pred, ref), _hdistance(ref, pred))

def calculate_ne_a(current_seg, current_goal, cumulative_pl_ref):
    """
    计算当前段的最后一个点与目标点之间的归一化误差。

    参数:
        current_seg (list or None): 当前段的点列表，每个点至少包含3个坐标。如果为空，则返回None。
        current_goal (list or array-like): 目标点，至少包含3个坐标。
        cumulative_pl_ref (float): 累积参考路径长度，用于归一化误差。

    返回:
        float: 归一化误差 (ne_a)。如果累积路径长度为0或无效，则返回无穷大。
    """
    ne_a = None  # 初始化归一化误差为None
    if current_seg:  # 如果当前段不为空
        final_point = np.array(current_seg[-1][:3])  # 提取当前段最后一个点的前3个坐标
        goal_point = np.array(current_goal[:3])      # 提取目标点的前3个坐标
        ne = np.linalg.norm(final_point - goal_point)  # 计算当前点与目标点之间的欧几里得距离
        # 根据累积路径长度计算归一化误差，若累积路径长度为0或无效，则返回无穷大
        ne_a = ne / cumulative_pl_ref if cumulative_pl_ref > 0 else float('inf')
    return ne_a  # 返回归一化误差

def calculate_cesr(points, goal, success_distance=20.0):
    """计算指令结束判断成功率"""
    if not points:
        return 0.0
    final_point = np.array(points[-1][:3])
    goal = np.array(goal[:3])
    distance = np.linalg.norm(final_point - goal)
    return 1.0 if distance <= success_distance else 0.0

def calculate_explore_ratio(ref_points):
    """计算探索比率：起终点直线距离 / 参考轨迹长度"""
    if len(ref_points) < 2:
        return 0.0
    start = np.array(ref_points[0][:3])
    end = np.array(ref_points[-1][:3])
    straight_dist = np.linalg.norm(end - start)
    path_length = calculate_pl(ref_points)
    return straight_dist / path_length if path_length > 0 else 0.0

# ========== 修改后的评估函数 ==========
def evaluate_trajectory(orig, pred, ref_type=1):
    # 获取轨迹的finished标记 - 支持两种数据格式
    if 'pred' in pred and pred['pred']:
        # 格式1: finished在pred数组中的每个项目里
        full_pred_finished = [p.get('finished', False) for p in pred['pred']]
    else:
        # 格式2: finished直接在顶层
        full_pred_finished = pred.get('finished', [])
    
    if ref_type == 1:
        # 类型1数据处理
        full_ref_traj = orig.get('reference_path', [])
        # 为原始参考轨迹生成finished标记
        full_ref_finished = [False] * (len(full_ref_traj)-1) + [True] if full_ref_traj else []
        if orig.get('goals'):
            full_goal = orig['goals'][0]['position']
        else:
            full_goal = [0, 0, 0]  # 默认值
        sentences = []
        sub_goals = [full_goal]
        ref_paths = [full_ref_traj]
    else:
        # 原始数据处理
        sentences = orig.get('sentence_instructions', [])
        sub_goals = [s['end_position'] for s in sentences]
        ref_paths = [s['reference_path'] for s in sentences]
        # 为每个分段的参考轨迹生成finished标记
        full_ref_finished = []
        for path in ref_paths:
            full_ref_finished.extend([False] * (len(path)-1) + [True] if path else [])
    
    full_pred_traj = pred['trajectory']
    full_ref_traj = [p for path in ref_paths for p in path]
    finished = pred.get('finished', [])
    segments = split_segments(finished)
    
    sub_metrics = []    # 存储每个子轨迹的指标
    pred_segment = []   # 当前分段的预测轨迹
    ref_segment = []   # 当前分段的参考轨迹
    cumulative_pred_traj = []  # 累计参考轨迹
    cumulative_ref_traj = []    # 累计预测轨迹
    cumulative_pl_ref = 0.0    # 累计路径长度
    total_actions = 0 #  总预测动作数
    # 计算explore rate
    pred_action_count = len(full_pred_traj)
    gt_action_count = len(full_ref_traj) if len(full_ref_traj) > 0 else 1  # 防止除0
    

    # 全局区段数据存储结构
    global_segment_data = {}  # {seg_num: metrics}
    
    if ref_type == 2:
        # 处理子指令分段
        for seg_idx in range(len(sentences)):
            # 累计预测轨迹计算
            if seg_idx < len(segments):
                start, end = segments[seg_idx]
                pred_segment = full_pred_traj[start:end+1]
                cumulative_pred_traj.extend(pred_segment)
            else:
                # 处理缺失分段
                pred_segment = []

            # 累计参考轨迹计算
            if seg_idx < len(ref_paths):
                ref_segment = ref_paths[seg_idx] if seg_idx < len(ref_paths) else []     
                cumulative_ref_traj.extend(ref_segment)
            else:
                # 处理缺失分段
                ref_segment = []

            goal = sub_goals[seg_idx] if seg_idx < len(sub_goals) else None

            # 计算统计信息
            pl_pred = calculate_pl(pred_segment)
            pl_ref = calculate_pl(ref_segment)
            cumulative_pl_pred = calculate_pl(cumulative_pred_traj)
            cumulative_pl_ref = calculate_pl(cumulative_ref_traj)
            final_point = np.array(pred_segment[-1][:3] if pred_segment else [0, 0, 0])
            goal_point = np.array(goal[:3] if goal and pred_segment else [0, 0, 0])

            # 初始化指标字典
            metrics = {
                # 基础指标
                'PL': 0.0,
                'Action_Count': 0.0,
                'SR_3D-5': 0.0,
                'SR_3D-10': 0.0,
                'SR_3D-15': 0.0,
                'SR_3D-20': 0.0,
                'SR_3D-30': 0.0,
                'SR_3D-40': 0.0,
                'OSR_3D-5': 0.0,
                'OSR_3D-10': 0.0,
                'OSR_3D-15': 0.0,
                'OSR_3D-20': 0.0,
                'OSR_3D-30': 0.0,
                'OSR_3D-40': 0.0,
                'NE_3D': 0.0,
                'NE-A': 0.0,
                'Explore_Ratio': 0.0,  # 新增探索度指标
                # 其他指标
                'PB': 0.0,
                'CESR': 0.0,
                'SPL': 0.0,
                'CLS': 0.0,
                # 三维时空指标
                'nDTW_3D': 0.0,
                'sDTW_3D': 0.0,
                'Backward-nDTW_3D': 0.0,
                'Bi-nDTW_3D': 0.0,
                'LCSS_3D': 0.0,
                'EDR_3D': 0.0,
                'ERP_3D': 0.0,
                'Hausdorff_3D': None,  # 设置为null不参与平均
                # 二维指标
                'SR_2D': 0.0,
                'OSR_2D': 0.0,
                'NE_2D': 0.0,
            }
            
            # 仅当存在预测分段时计算指标
            if pred_segment and ref_type == 2:
                # 计算指标（全部使用累计轨迹）
                metrics.update({
                    # 基础指标
                    'SR_3D-15': calculate_sr(cumulative_pred_traj, goal, 15.0),
                    'OSR_3D-15': calculate_osr(cumulative_pred_traj, goal, 15.0),
                    'PL': pl_pred,
                    'Action_Count': len(pred_segment),
                    'SR_3D-5': calculate_sr(cumulative_pred_traj, goal, 5.0),
                    'SR_3D-10': calculate_sr(cumulative_pred_traj, goal, 10.0),
                    'SR_3D-20': calculate_sr(cumulative_pred_traj, goal),
                    'SR_3D-30': calculate_sr(cumulative_pred_traj, goal, 30.0),
                    'SR_3D-40': calculate_sr(cumulative_pred_traj, goal, 40.0),
                    'OSR_3D-5': calculate_osr(cumulative_pred_traj, goal, 5.0),
                    'OSR_3D-10': calculate_osr(cumulative_pred_traj, goal, 10.0),
                    'OSR_3D-20': calculate_osr(cumulative_pred_traj, goal),
                    'OSR_3D-30': calculate_osr(cumulative_pred_traj, goal, 30.0),
                    'OSR_3D-40': calculate_osr(cumulative_pred_traj, goal, 40.0),
                    'NE_3D': np.linalg.norm(final_point - goal_point) if pred_segment and goal else float('inf'),
                    'NE-A': calculate_ne_a(pred_segment, goal, cumulative_pl_ref) if pred_segment else None,
                    # 其他指标
                    'PB': min(cumulative_pl_ref, cumulative_pl_pred) / max(cumulative_pl_ref, cumulative_pl_pred) if max(cumulative_pl_ref, cumulative_pl_pred) != 0 else 0.0,
                    'CESR': calculate_cesr(cumulative_pred_traj, goal),
                    'SPL': calculate_spl(calculate_sr(cumulative_pred_traj, goal), cumulative_pl_ref, cumulative_pl_pred),
                    'CLS': calculate_cls(cumulative_pred_traj, cumulative_ref_traj),
                    # 三维时空指标
                    'nDTW_3D': calculate_ndtw_3d(cumulative_pred_traj, cumulative_ref_traj),
                    'sDTW_3D': calculate_sdtw_3d(np.linalg.norm(final_point - goal_point), calculate_ndtw_3d(cumulative_pred_traj, cumulative_ref_traj)) if cumulative_pred_traj else 0.0,
                    'Backward-nDTW_3D': calculate_ndtw_3d(cumulative_ref_traj, cumulative_pred_traj),
                    'Bi-nDTW_3D': bidirectional_ndtw_3d(cumulative_pred_traj, cumulative_ref_traj),
                    'LCSS_3D': lcss_3d(cumulative_pred_traj, cumulative_ref_traj),
                    'EDR_3D': edr_3d(cumulative_pred_traj, cumulative_ref_traj),
                    'ERP_3D': erp_3d(cumulative_pred_traj, cumulative_ref_traj),
                    'Hausdorff_3D': hausdorff_3d(cumulative_pred_traj, cumulative_ref_traj),
                    # 二维指标
                    'SR_2D': calculate_sr_2d(cumulative_pred_traj, goal),
                    'OSR_2D': calculate_osr_2d(cumulative_pred_traj, goal),
                    'NE_2D': np.linalg.norm(final_point[:2] - goal_point[:2]) if pred_segment and goal else float('inf'),
                })
            elif ref_type == 2:
                # 用整条预测轨迹的终点代替预测区段终点来计算NE和NE-A
                    if cumulative_pred_traj and goal is not None:
                        final_point = np.array(cumulative_pred_traj[-1][:3])
                        goal_point = np.array(goal[:3])
                        metrics['NE_3D'] = np.linalg.norm(final_point - goal_point)
                        metrics['NE-A'] = calculate_ne_a(full_pred_traj, goal, cumulative_pl_ref)
                    else:
                        metrics['NE_3D'] = float('inf')
                        metrics['NE-A'] = None
            
            # 存储到按区段分组的字典
            global_segment_data[seg_idx] = {
                'SR_3D-5': metrics.get('SR_3D-5', 0.0),
                'SR_3D-10': metrics.get('SR_3D-10', 0.0),
                'SR_3D-15': metrics.get('SR_3D-15', 0.0),
                'SR_3D-20': metrics.get('SR_3D-20', 0.0),
                'OSR_3D-5': metrics.get('OSR_3D-5', 0.0),
                'OSR_3D-10': metrics.get('OSR_3D-10', 0.0),
                'OSR_3D-15': metrics.get('OSR_3D-15', 0.0),
                'OSR_3D-20': metrics.get('OSR_3D-20', 0.0),
                'SPL': metrics.get('SPL', 0.0),
                'NE_3D': metrics.get('NE_3D', float('inf')),
                'Min_Distance': metrics.get('Min_Distance', float('inf')),
                'NE-A': metrics.get('NE-A', None)
            }

            # 存储当前分段的指标
            sub_metrics.append(metrics)
            total_actions += len(pred_segment)
    
    # 完整轨迹计算

    full_metrics = {
        'segment_metrics': global_segment_data,  # 存储全局区段数据
        'PL': calculate_pl(full_pred_traj),
        'PB': min(calculate_pl(full_ref_traj), calculate_pl(full_pred_traj)) / max(calculate_pl(full_ref_traj), calculate_pl(full_pred_traj)),
        'SR_3D-5': calculate_sr(full_pred_traj, sub_goals[-1], 5.0),
        'SR_3D-10': calculate_sr(full_pred_traj, sub_goals[-1], 10.0),
        'SR_3D-15': calculate_sr(full_pred_traj, sub_goals[-1], 15.0),
        'SR_3D-20': calculate_sr(full_pred_traj, sub_goals[-1]),
        'SR_3D-30': calculate_sr(full_pred_traj, sub_goals[-1], 30.0),
        'SR_3D-40': calculate_sr(full_pred_traj, sub_goals[-1], 40.0),
        'OSR_3D-5': calculate_osr(full_pred_traj, sub_goals[-1], 5.0),
        'OSR_3D-10': calculate_osr(full_pred_traj, sub_goals[-1], 10.0),
        'OSR_3D-15': calculate_osr(full_pred_traj, sub_goals[-1], 15.0),
        'OSR_3D-20': calculate_osr(full_pred_traj, sub_goals[-1]),
        'OSR_3D-30': calculate_osr(full_pred_traj, sub_goals[-1], 30.0),
        'OSR_3D-40': calculate_osr(full_pred_traj, sub_goals[-1], 40.0),
        'SPL': calculate_spl(calculate_sr(full_pred_traj, sub_goals[-1]), calculate_pl(full_ref_traj), calculate_pl(full_pred_traj)),
        'CLS': calculate_cls(full_pred_traj, full_ref_traj),
        'NE_3D': np.linalg.norm(np.array(full_pred_traj[-1][:3]) - np.array(sub_goals[-1][:3])) if full_pred_traj else None,
        'NE-A': calculate_ne_a(full_pred_traj, sub_goals[-1], calculate_pl(full_ref_traj)) if full_pred_traj else None,
        'nDTW_3D': calculate_ndtw_3d(full_pred_traj, full_ref_traj),
        'Sentence-nDTW_3D': calculate_sentence_ndtw_3d(full_pred_traj, full_ref_traj, full_pred_finished, full_ref_finished) if (full_pred_traj and full_ref_traj and full_pred_finished and full_ref_finished) else 0.0,
        'sDTW_3D': calculate_sdtw_3d(np.linalg.norm(np.array(full_pred_traj[-1][:3]) - np.array(sub_goals[-1][:3])), calculate_ndtw_3d(full_pred_traj, full_ref_traj)) if full_pred_traj else 0.0,
        'Backward-nDTW_3D': calculate_ndtw_3d(full_ref_traj, full_pred_traj),
        'Bi-nDTW_3D': bidirectional_ndtw_3d(full_pred_traj, full_ref_traj),
        'LCSS_3D': lcss_3d(full_pred_traj, full_ref_traj),
        'EDR_3D': edr_3d(full_pred_traj, full_ref_traj),
        'ERP_3D': erp_3d(full_pred_traj, full_ref_traj),
        'Hausdorff_3D': hausdorff_3d(full_pred_traj, full_ref_traj),
        'Action_Count': len(full_pred_traj) if ref_type == 1 else total_actions,
        'Collision': 1 if pred.get('is_collisioned', False) else 0,
        'Explore_Ratio': calculate_explore_ratio(full_ref_traj),  # 新增探索度指标
        'Explore_Rate': pred_action_count / gt_action_count,  # 新增explore rate
        # 二维指标
        'SR_2D': calculate_sr_2d(full_pred_traj, sub_goals[-1]),
        'OSR_2D': calculate_osr_2d(full_pred_traj, sub_goals[-1]),
        'NE_2D': np.linalg.norm(np.array(full_pred_traj[-1][:3])[:2] - np.array(sub_goals[-1][:3])[:2]) if full_pred_traj else None,
    }
    
    # 添加 Min_Distance 计算，使用错误处理
    try:
        if full_pred_traj and sub_goals:
            full_metrics['Min_Distance'] = np.min([np.linalg.norm(np.array(p[:3]) - np.array(sub_goals[-1][:3])) for p in full_pred_traj])
        else:
            full_metrics['Min_Distance'] = float('inf')
    except Exception:
        full_metrics['Min_Distance'] = float('inf')
    
    # 计算CESR
    success_count = sum(1 for sub in sub_metrics if sub['CESR'] == 1.0)
    full_metrics['CESR'] = success_count / len(sub_metrics) if len(sub_metrics) > 0 else 0.0
    
    return sub_metrics, full_metrics

def main():
    # 支持三种调用：
    # 1) 单方法:                     eval_1013.py <ref_type_A> <pred_type_A> <gt_A> <pred_A>
    # 2) 成对(同GT同模式):           eval_1013.py <ref_type_A> <pred_type_A> <gt_A> <pred_A> <pred_type_B> <pred_B>
    # 3) 成对(可不同GT/模式):        eval_1013.py <ref_type_A> <pred_type_A> <gt_A> <pred_A> <ref_type_B> <pred_type_B> <gt_B> <pred_B>
    # 先解析可选flag
    pos_argv, opts = _parse_cli_with_flags(sys.argv)
    # 兼容原有位置参数数量校验（不含脚本名，故应为 4/6/8）
    if len(pos_argv) not in (4, 6, 8):
        print("Usage:")
        print("  python eval_1013.py <ref_type_A> <pred_type_A> <gt_A> <pred_A> [--segcat <json>]")
        print("  python eval_1013.py <ref_type_A> <pred_type_A> <gt_A> <pred_A> <pred_type_B> <pred_B> [--segcat <json>]")
        print("  python eval_1013.py <ref_type_A> <pred_type_A> <gt_A> <pred_A> <ref_type_B> <pred_type_B> <gt_B> <pred_B> [--segcat <json>]")
        print("  ref_type: 1 (Full-level) | 2 (Sentence-level/Fine-level)")
        print("  pred_type: 1 (dict) | 2 (multi-dict) | 3 (single) | 4 (folder)")
        sys.exit(1)

    # 使用解析后的纯位置参数
    ref_type = int(pos_argv[0])
    mode = int(pos_argv[1])
    gt_folder = pos_argv[2]
    pred_path = pos_argv[3]
    segcat_json_path = opts.get('segcat')

    has_pair = False
    if len(pos_argv) == 6:
        # 旧用法（同GT/同ref_type），B只给 pred_type_B 与 pred_B
        ref_type_b = ref_type
        mode_b = int(pos_argv[4])
        gt_folder_b = gt_folder
        pred_path_b = pos_argv[5]
        has_pair = True
    elif len(pos_argv) == 8:
        # 新用法（允许不同GT/不同ref_type）
        ref_type_b = int(pos_argv[4])
        mode_b = int(pos_argv[5])
        gt_folder_b = pos_argv[6]
        pred_path_b = pos_argv[7]
        has_pair = True

    # 生成安全的基础名（不含后缀）；目录则取目录名，文件则取“上级目录_文件名”
    def get_safe_base_name(path: str) -> str:
        try:
            if os.path.isdir(path):
                base_name = os.path.basename(os.path.normpath(path))
            else:
                folder_name = os.path.basename(os.path.dirname(path.rstrip('/')))
                file_name = os.path.splitext(os.path.basename(path))[0]
                base_name = f"{folder_name}_{file_name}"
        except Exception:
            base_name = os.path.splitext(os.path.basename(path))[0]
        # 清理Windows不允许的字符
        return re.sub(r'[\\/*?:"<>|]', "_", base_name)

    # 根据是否进行成对显著性检验来决定输出文件名
    if has_pair:
        base_a = get_safe_base_name(pred_path)
        base_b = get_safe_base_name(pred_path_b)
        gt_base_a = get_safe_base_name(gt_folder)
        gt_base_b = get_safe_base_name(gt_folder_b)
        if gt_base_a == gt_base_b:
            output_file = f"{base_a}__VS__{base_b}_ON_{gt_base_a}_evaluation_results_v20251013.xlsx"
        else:
            output_file = f"{base_a}_ON_{gt_base_a}__VS__{base_b}_ON_{gt_base_b}_evaluation_results_v20251013.xlsx"
    else:
        base_a = get_safe_base_name(pred_path)
        gt_base = get_safe_base_name(gt_folder)
        output_file = f"{base_a}_ON_{gt_base}_evaluation_results_v20251013.xlsx"

    # 参数验证与加载 GT(A)
    if mode == 3 and not os.path.isdir(pred_path):
        print("Error: Mode 3 requires a folder path")
        sys.exit(1)
    if ref_type == 1:
        gt_data = load_reference_trajectory_type1(gt_folder)
    else:
        gt_data = load_groundtruth_type2(gt_folder)
    pred_data = load_predictions(mode, pred_path)

    # 若指定了单句分段分类JSON，仅在 ref_type==2 生效
    segcat_mapping = {}
    if segcat_json_path:
        segcat_mapping = load_seg_categories(segcat_json_path)
        if segcat_mapping:
            print(f"Loaded segment categories: {len(segcat_mapping)} keyed entries from {segcat_json_path}")
        else:
            print(f"Warning: No usable segment categories parsed from {segcat_json_path}")

    # 记录原始轨迹数量
    original_trajectory_count = sum(len(preds) for preds in pred_data.values())

    # 去重处理：按优先级保留最佳轨迹
    # 优先级：1) SR_3D-20高 2) OSR_3D-20高 3) NE_3D低
    print("开始去重处理...")
    
    # 步骤1：按episode_id分组收集所有轨迹
    episode_groups = {}
    original_count = 0
    for ep_id, preds in pred_data.items():
        for pred in preds:
            if ep_id not in episode_groups:
                episode_groups[ep_id] = []
            episode_groups[ep_id].append({
                'key': ep_id,
                'pred': pred,
                'ep_id': ep_id
            })
            original_count += 1
    
    print(f"收集到 {len(episode_groups)} 个不同的episode，共 {original_count} 条轨迹")
    
    # 步骤2：计算每组轨迹的指标并选择最佳的
    temp_results = {}
    duplicate_count = 0
    
    for ep_id, trajectories in tqdm(episode_groups.items(), desc="处理每组轨迹"):
        if ep_id not in gt_data:
            print(f"Warning: Missing ground truth for {ep_id}, 跳过该组")
            continue
        
        if len(trajectories) > 1:
            duplicate_count += len(trajectories) - 1
            print(f"发现重复轨迹 {ep_id}，共 {len(trajectories)} 条")
        
        best_trajectory = None
        best_metrics = None
        
        # 计算每条轨迹的指标并找到最佳的
        for i, traj_data in enumerate(trajectories):
            try:
                # 计算该轨迹的指标
                sub_metrics, full_metrics = evaluate_trajectory(gt_data[ep_id], traj_data['pred'], ref_type)
                
                # 获取用于比较的关键指标
                sr_3d_20 = full_metrics.get('SR_3D-20', 0)
                osr_3d_20 = full_metrics.get('OSR_3D-20', 0)
                ne_3d = full_metrics.get('NE_3D', float('inf'))
                
                current_metrics = {
                    'sr_3d_20': sr_3d_20,
                    'osr_3d_20': osr_3d_20,
                    'ne_3d': ne_3d,
                    'sub_metrics': sub_metrics,
                    'full_metrics': full_metrics,
                    'traj_data': traj_data,
                    'index': i
                }
                
                # 如果是第一条轨迹或者当前轨迹更好，则更新最佳轨迹
                if best_trajectory is None or is_better_trajectory(current_metrics, best_metrics):
                    if best_trajectory is not None and best_metrics is not None and len(trajectories) > 1:
                        # 打印替换原因
                        reason = compare_trajectories(current_metrics, best_metrics)
                        print(f"  轨迹 {i+1} 更好：{reason}")
                    best_trajectory = traj_data
                    best_metrics = current_metrics
                elif len(trajectories) > 1 and best_metrics is not None:
                    # 打印保留原因
                    reason = compare_trajectories(best_metrics, current_metrics)
                    print(f"  保留轨迹 {best_metrics['index']+1}：{reason}")
                    
            except Exception as e:
                print(f"计算轨迹 {ep_id}[{i}] 时出错: {str(e)}")
                continue
        
        # 保存最佳轨迹
        if best_trajectory is not None and best_metrics is not None:
            temp_results[ep_id] = {
                'pred': best_trajectory['pred'],
                'sr_3d_20': best_metrics['sr_3d_20'],
                'osr_3d_20': best_metrics['osr_3d_20'],
                'ne_3d': best_metrics['ne_3d'],
                'sub_metrics': best_metrics['sub_metrics'],
                'full_metrics': best_metrics['full_metrics']
            }
    
    # 更新pred_data为去重后的数据
    pred_data = {ep_id: data['pred'] for ep_id, data in temp_results.items()}

    # 新增：可选保存去重后择优轨迹json
    if SAVE_DEDUP_JSON:
        dedup_json_path = os.path.splitext(output_file)[0] + '_dedup.json'
        with open(dedup_json_path, 'w', encoding='utf-8') as f:
            json.dump(pred_data, f, ensure_ascii=False, indent=2)
        print(f"去重后择优轨迹已保存至 {dedup_json_path}")

    print(f"去重完成，原始轨迹数: {original_trajectory_count}, 去重后轨迹数: {len(pred_data)}")

    # 加载数据后添加统计变量
    total_gt = len(gt_data)
    total_pred = len(pred_data)
    computed = 0  # 实际计算的轨迹数量    # 执行评估 - 使用已计算的结果
    results = {}
    for ep_id in tqdm(pred_data.keys(), desc="构建最终结果"):
        if ep_id not in gt_data:
            print(f"Warning: Missing ground truth for {ep_id}")
            continue
        
        # 使用之前计算好的结果
        if ep_id in temp_results:
            results[ep_id] = {
                'sub_metrics': temp_results[ep_id]['sub_metrics'],
                'full_metrics': temp_results[ep_id]['full_metrics']
            }
        else:
            # 如果之前计算过程中出错，重新计算
            try:
                sub_metrics, full_metrics = evaluate_trajectory(gt_data[ep_id], pred_data[ep_id], ref_type)
                results[ep_id] = {
                    'sub_metrics': sub_metrics,
                    'full_metrics': full_metrics
                }
            except Exception as e:
                print(f"Error evaluating {ep_id}: {str(e)}")
                continue

        # 有效轨迹计数
        computed += 1


    # 生成结果报告
    if results:
        rows = []
        seg_avg_rows = []
        all_full_metrics = []
        all_sub_metrics = []
        global_segment_store = defaultdict(lambda: {
            'SR_3D-5': [],
            'SR_3D-10': [],
            'SR_3D-15': [],
            'SR_3D-20': [],
            'SR_3D-30': [],
            'SR_3D-40': [],
            'OSR_3D-5': [],
            'OSR_3D-10': [],
            'OSR_3D-15': [],
            'OSR_3D-20': [],
            'OSR_3D-30': [],
            'OSR_3D-40': [],
            'SPL': [],
            'NE_3D': [],
            'NE-A': []
        })

        # 新增：SR/OSR 15、20考虑碰撞的列
        def sr_osr_collision(val, collision):
            return 0.0 if collision==1 else val

        # 列顺序
        columns_order = [
            'Episode ID', 'Type',
            'PL','Action_Count', 'Explore_Rate', 'SR_3D-5', 'SR_3D-10', 'SR_3D-15', 'SR_3D-15-Collision', 'SR_3D-20', 'SR_3D-20-Collision', 'SR_3D-30', 'SR_3D-40',
            'OSR_3D-5', 'OSR_3D-10', 'OSR_3D-15', 'OSR_3D-15-Collision', 'OSR_3D-20', 'OSR_3D-20-Collision', 'OSR_3D-30', 'OSR_3D-40',
            'NE_3D', 'Min_Distance', 'NE-A', 'nDTW_3D', 'Sentence-nDTW_3D', 'sDTW_3D', 'SPL', 'CESR', 'PB', 
            'CLS', 'Backward-nDTW_3D', 'Bi-nDTW_3D','LCSS_3D', 'EDR_3D', 'ERP_3D', 'Hausdorff_3D', 'Collision', 'SR_2D', 'OSR_2D', 'NE_2D', 'Explore_Ratio'
        ]

        # 构建数据行
        for ep_id, metrics in results.items():
            # 收集完整轨迹指标
            full = metrics['full_metrics']
            all_full_metrics.append(full)
            
            # 收集子轨迹指标
            sub_list = metrics['sub_metrics']
            all_sub_metrics.extend(sub_list)

            # 收集各轨迹的区段指标
            seg_metrics = metrics['full_metrics']['segment_metrics']
            for seg_num, seg_data in seg_metrics.items():
                for metric in ['SR_3D-5', 'SR_3D-10', 'SR_3D-15', 'SR_3D-20', 'SR_3D-30', 'SR_3D-40', 'OSR_3D-5', 'OSR_3D-10', 'OSR_3D-15', 'OSR_3D-20', 'OSR_3D-30', 'OSR_3D-40', 'SPL', 'NE_3D', 'NE-A']:
                    # 跳过无效值
                    value = seg_data.get(metric, None)
                    if value is not None and not np.isnan(value):
                        global_segment_store[seg_num][metric].append(value)

            # 完整轨迹指标
            rows.append({
                'Episode ID': ep_id,
                'Type': 'Full',
                'PL': full['PL'],
                'Action_Count': full['Action_Count'],
                'SR_3D-5': full['SR_3D-5'],
                'SR_3D-10': full['SR_3D-10'],
                'SR_3D-15': full['SR_3D-15'],
                'SR_3D-15-Collision': sr_osr_collision(full['SR_3D-15'], full['Collision']),
                'SR_3D-20': full['SR_3D-20'],
                'SR_3D-20-Collision': sr_osr_collision(full['SR_3D-20'], full['Collision']),
                'SR_3D-30': full['SR_3D-30'],
                'SR_3D-40': full['SR_3D-40'],
                'OSR_3D-5': full['OSR_3D-5'],
                'OSR_3D-10': full['OSR_3D-10'],
                'OSR_3D-15': full['OSR_3D-15'],
                'OSR_3D-15-Collision': sr_osr_collision(full['OSR_3D-15'], full['Collision']),
                'OSR_3D-20': full['OSR_3D-20'],
                'OSR_3D-20-Collision': sr_osr_collision(full['OSR_3D-20'], full['Collision']),
                'OSR_3D-30': full['OSR_3D-30'],
                'OSR_3D-40': full['OSR_3D-40'],
                'NE_3D': full['NE_3D'],
                'Min_Distance': full.get('Min_Distance', None),
                'Explore_Rate': full.get('Explore_Rate', None),
                # 'NE-A': full['NE-A'],
                'SPL': full['SPL'],
                'CESR': full['CESR'],
                'PB': full['PB'],
                'CLS': full['CLS'],
                'nDTW_3D': full['nDTW_3D'],
                'Sentence-nDTW_3D': full['Sentence-nDTW_3D'],
                'Backward-nDTW_3D': full['Backward-nDTW_3D'],
                'Bi-nDTW_3D': full['Bi-nDTW_3D'],
                'sDTW_3D': full['sDTW_3D'],
                'LCSS_3D': full['LCSS_3D'],
                'EDR_3D': full['EDR_3D'],
                'ERP_3D': full['ERP_3D'],
                'Hausdorff_3D': full['Hausdorff_3D'],
                'Collision': full['Collision'],
                # 二维指标
                'SR_2D': full['SR_2D'],
                'OSR_2D': full['OSR_2D'],
                'NE_2D': full['NE_2D'],
                'Explore_Ratio': full.get('Explore_Ratio', None)
            })

            # 子轨迹指标
            for seg_idx, sub in enumerate(metrics['sub_metrics']):
                rows.append({
                    'Episode ID': ep_id,
                    'Type': f'Seg_{seg_idx}',
                    'PL': sub['PL'],
                    'Action_Count': sub['Action_Count'],
                    'SR_3D-5': sub['SR_3D-5'],
                    'SR_3D-10': sub['SR_3D-10'],
                    'SR_3D-15': sub['SR_3D-15'],
                    'SR_3D-20': sub['SR_3D-20'],
                    'SR_3D-30': sub['SR_3D-30'],
                    'SR_3D-40': sub['SR_3D-40'],
                    'OSR_3D-5': sub['OSR_3D-5'],
                    'OSR_3D-10': sub['OSR_3D-10'],
                    'OSR_3D-15': sub['OSR_3D-15'],
                    'OSR_3D-20': sub['OSR_3D-20'],
                    'OSR_3D-30': sub['OSR_3D-30'],
                    'OSR_3D-40': sub['OSR_3D-40'],
                    'NE_3D': sub['NE_3D'], # 为便于分析，记录NE
                    'NE-A': sub['NE-A'],
                    # 二维指标（用于分类工作表）
                    'SR_2D': sub.get('SR_2D'),
                    'OSR_2D': sub.get('OSR_2D'),
                    'NE_2D': sub.get('NE_2D'),
                    'SPL': sub['SPL'],
                    'CESR': sub['CESR'],
                    'PB': sub['PB'],
                    'CLS': sub['CLS'],
                    'nDTW_3D': sub['nDTW_3D'],
                    'Backward-nDTW_3D': sub['Backward-nDTW_3D'],
                    'Bi-nDTW_3D': sub['Bi-nDTW_3D'],
                    'sDTW_3D': sub['sDTW_3D'],
                    'LCSS_3D': sub['LCSS_3D'],
                    'EDR_3D': sub['EDR_3D'],
                    'ERP_3D': sub['ERP_3D'],
                    'Hausdorff_3D': sub['Hausdorff_3D'],
                    'Collision': None,  # 子轨迹无碰撞指标
                })

            # 子轨迹平均（仅计算有效指标）
            if metrics['sub_metrics'] and ref_type == 2:
                avg_sub = {
                    'PL': safe_nanmean([s['PL'] for s in metrics['sub_metrics'] if s['PL'] is not None]),
                    'Action_Count': safe_nanmean([s['Action_Count'] for s in metrics['sub_metrics'] if s['Action_Count'] is not None]),
                    'SR_3D-5': safe_nanmean([s['SR_3D-5'] for s in metrics['sub_metrics'] if s['SR_3D-5'] is not None]),
                    'SR_3D-10': safe_nanmean([s['SR_3D-10'] for s in metrics['sub_metrics'] if s['SR_3D-10'] is not None]),
                    'SR_3D-15': safe_nanmean([s['SR_3D-15'] for s in metrics['sub_metrics'] if s['SR_3D-15'] is not None]),
                    'SR_3D-20': safe_nanmean([s['SR_3D-20'] for s in metrics['sub_metrics'] if s['SR_3D-20'] is not None]),
                    'SR_3D-30': safe_nanmean([s['SR_3D-30'] for s in metrics['sub_metrics'] if s['SR_3D-30'] is not None]),
                    'SR_3D-40': safe_nanmean([s['SR_3D-40'] for s in metrics['sub_metrics'] if s['SR_3D-40'] is not None]),
                    'OSR_3D-5': safe_nanmean([s['OSR_3D-5'] for s in metrics['sub_metrics'] if s['OSR_3D-5'] is not None]),
                    'OSR_3D-10': safe_nanmean([s['OSR_3D-10'] for s in metrics['sub_metrics'] if s['OSR_3D-10'] is not None]),
                    'OSR_3D-15': safe_nanmean([s['OSR_3D-15'] for s in metrics['sub_metrics'] if s['OSR_3D-15'] is not None]),
                    'OSR_3D-20': safe_nanmean([s['OSR_3D-20'] for s in metrics['sub_metrics'] if s['OSR_3D-20'] is not None]),
                    'OSR_3D-30': safe_nanmean([s['OSR_3D-30'] for s in metrics['sub_metrics'] if s['OSR_3D-30'] is not None]),
                    'OSR_3D-40': safe_nanmean([s['OSR_3D-40'] for s in metrics['sub_metrics'] if s['OSR_3D-40'] is not None]),
                    'NE_3D': safe_nanmean([s['NE_3D'] for s in metrics['sub_metrics'] if s['NE_3D'] is not None]),
                    'NE-A': safe_nanmean([s['NE-A'] for s in metrics['sub_metrics'] if s['NE-A'] is not None]),
                    # 二维指标（子轨迹平均）
                    'SR_2D': safe_nanmean([s.get('SR_2D') for s in metrics['sub_metrics'] if s.get('SR_2D') is not None]),
                    'OSR_2D': safe_nanmean([s.get('OSR_2D') for s in metrics['sub_metrics'] if s.get('OSR_2D') is not None]),
                    'NE_2D': safe_nanmean([s.get('NE_2D') for s in metrics['sub_metrics'] if s.get('NE_2D') is not None]),
                    'SPL': safe_nanmean([s['SPL'] for s in metrics['sub_metrics'] if s['SPL'] is not None]),
                    'CESR': safe_nanmean([s['CESR'] for s in metrics['sub_metrics'] if s['CESR'] is not None]),
                    'PB': safe_nanmean([s['PB'] for s in metrics['sub_metrics'] if s['PB'] is not None]),
                    'CLS': safe_nanmean([s['CLS'] for s in metrics['sub_metrics'] if s['CLS'] is not None]),
                    'nDTW_3D': safe_nanmean([s['nDTW_3D'] for s in metrics['sub_metrics'] if s['nDTW_3D'] is not None]),
                    'Backward-nDTW_3D': safe_nanmean([s['Backward-nDTW_3D'] for s in metrics['sub_metrics'] if s['Backward-nDTW_3D'] is not None]),
                    'Bi-nDTW_3D': safe_nanmean([s['Bi-nDTW_3D'] for s in metrics['sub_metrics'] if s['Bi-nDTW_3D'] is not None]),
                    'sDTW_3D': safe_nanmean([s['sDTW_3D'] for s in metrics['sub_metrics'] if s['sDTW_3D'] is not None]),
                    'LCSS_3D': safe_nanmean([s['LCSS_3D'] for s in metrics['sub_metrics'] if s['LCSS_3D'] is not None]),
                    'EDR_3D': safe_nanmean([s['EDR_3D'] for s in metrics['sub_metrics'] if s['EDR_3D'] is not None]),
                    'ERP_3D': safe_nanmean([s['ERP_3D'] for s in metrics['sub_metrics'] if s['ERP_3D'] is not None]),
                    'Hausdorff_3D': safe_nanmean([s['Hausdorff_3D'] for s in metrics['sub_metrics'] if s['Hausdorff_3D'] is not None])
                }
                # 添加平均行
                rows.append({
                    'Episode ID': ep_id,
                    'Type': 'Sub_Avg',
                    'Collision': None,   # 子轨迹无碰撞指标
                    **avg_sub
                })

        # ========== 全局均值统计 ==========
        def calculate_global(values, key):
            valid = []
            for v in values:
                val = v.get(key)
                if val is not None and not np.isnan(val) and not np.isinf(val):
                    valid.append(val)
            return np.nanmean(valid) if valid else None

        # 1. 全局完整轨迹平均
        if all_full_metrics:
            global_full = {
                'Episode ID': 'Global',
                'Type': 'Full_Avg',
                'PL': calculate_global(all_full_metrics, 'PL'),
                'SPL': calculate_global(all_full_metrics, 'SPL'),
                'PB': calculate_global(all_full_metrics, 'PB'),
                'SR_3D-5': calculate_global(all_full_metrics, 'SR_3D-5'),
                'SR_3D-10': calculate_global(all_full_metrics, 'SR_3D-10'),
                'SR_3D-15': calculate_global(all_full_metrics, 'SR_3D-15'),
                'SR_3D-15-Collision': calculate_global(all_full_metrics, 'SR_3D-15') if all_full_metrics[0].get('Collision', 0) == 0 else 0.0,
                'SR_3D-20': calculate_global(all_full_metrics, 'SR_3D-20'),
                'SR_3D-20-Collision': calculate_global(all_full_metrics, 'SR_3D-20') if all_full_metrics[0].get('Collision', 0) == 0 else 0.0,
                'SR_3D-30': calculate_global(all_full_metrics, 'SR_3D-30'),
                'SR_3D-40': calculate_global(all_full_metrics, 'SR_3D-40'),
                'OSR_3D-5': calculate_global(all_full_metrics, 'OSR_3D-5'),
                'OSR_3D-10': calculate_global(all_full_metrics, 'OSR_3D-10'),
                'OSR_3D-15': calculate_global(all_full_metrics, 'OSR_3D-15'),
                'OSR_3D-15-Collision': calculate_global(all_full_metrics, 'OSR_3D-15') if all_full_metrics[0].get('Collision', 0) == 0 else 0.0,
                'OSR_3D-20': calculate_global(all_full_metrics, 'OSR_3D-20'),
                'OSR_3D-20-Collision': calculate_global(all_full_metrics, 'OSR_3D-20') if all_full_metrics[0].get('Collision', 0) == 0 else 0.0,
                'OSR_3D-30': calculate_global(all_full_metrics, 'OSR_3D-30'),
                'OSR_3D-40': calculate_global(all_full_metrics, 'OSR_3D-40'),
                'CLS': calculate_global(all_full_metrics, 'CLS'),
                'NE_3D': calculate_global(all_full_metrics, 'NE_3D'),
                'Min_Distance': calculate_global(all_full_metrics, 'Min_Distance'),
                # 'NE-A': calculate_global(all_full_metrics, 'NE-A'),
                'Explore_Rate': calculate_global(all_full_metrics, 'Explore_Rate'),
                'Explore_Ratio': calculate_global(all_full_metrics, 'Explore_Ratio'),
                'nDTW_3D': calculate_global(all_full_metrics, 'nDTW_3D'),
                'Sentence-nDTW_3D': calculate_global(all_full_metrics, 'Sentence-nDTW_3D'),
                'sDTW_3D': calculate_global(all_full_metrics, 'sDTW_3D'),
                'Backward-nDTW_3D': calculate_global(all_full_metrics, 'Backward-nDTW_3D'),
                'Bi-nDTW_3D': calculate_global(all_full_metrics, 'Bi-nDTW_3D'),
                'LCSS_3D': calculate_global(all_full_metrics, 'LCSS_3D'),
                'EDR_3D': calculate_global(all_full_metrics, 'EDR_3D'),
                'ERP_3D': calculate_global(all_full_metrics, 'ERP_3D'),
                'Hausdorff_3D': calculate_global(all_full_metrics, 'Hausdorff_3D'),
                'Action_Count': calculate_global(all_full_metrics, 'Action_Count'),
                'Collision': calculate_global(all_full_metrics, 'Collision'),
                'CESR': calculate_global(all_full_metrics, 'CESR'),
                # 二维指标
                'SR_2D': calculate_global(all_full_metrics, 'SR_2D'),
                'OSR_2D': calculate_global(all_full_metrics, 'OSR_2D'),
                'NE_2D': calculate_global(all_full_metrics, 'NE_2D')
            }
            rows.append(global_full)

        # 2. 全局子轨迹平均
        if all_sub_metrics and ref_type == 2:
            global_sub = {
                'Episode ID': 'Global',
                'Type': 'Avg_of_All_Sub',
                'PL': calculate_global(all_sub_metrics, 'PL'),
                'SPL': calculate_global(all_sub_metrics, 'SPL'),
                'PB': calculate_global(all_sub_metrics, 'PB'),
                'SR_3D-5': calculate_global(all_sub_metrics, 'SR_3D-5'),
                'SR_3D-10': calculate_global(all_sub_metrics, 'SR_3D-10'),
                'SR_3D-15': calculate_global(all_sub_metrics, 'SR_3D-15'),
                'SR_3D-20': calculate_global(all_sub_metrics, 'SR_3D-20'),
                'SR_3D-30': calculate_global(all_sub_metrics, 'SR_3D-30'),
                'SR_3D-40': calculate_global(all_sub_metrics, 'SR_3D-40'),
                'OSR_3D-5': calculate_global(all_sub_metrics, 'OSR_3D-5'),
                'OSR_3D-10': calculate_global(all_sub_metrics, 'OSR_3D-10'),
                'OSR_3D-15': calculate_global(all_sub_metrics, 'OSR_3D-15'),
                'OSR_3D-20': calculate_global(all_sub_metrics, 'OSR_3D-20'),
                'OSR_3D-30': calculate_global(all_sub_metrics, 'OSR_3D-30'),
                'OSR_3D-40': calculate_global(all_sub_metrics, 'OSR_3D-40'),
                'CLS': calculate_global(all_sub_metrics, 'CLS'),
                'NE_3D': calculate_global(all_sub_metrics, 'NE_3D'),
                'NE-A': calculate_global(all_sub_metrics, 'NE-A'),
                'nDTW_3D': calculate_global(all_sub_metrics, 'nDTW_3D'),
                'sDTW_3D': calculate_global(all_sub_metrics, 'sDTW_3D'),
                'Backward-nDTW_3D': calculate_global(all_sub_metrics, 'Backward-nDTW_3D'),
                'Bi-nDTW_3D': calculate_global(all_sub_metrics, 'Bi-nDTW_3D'),
                'LCSS_3D': calculate_global(all_sub_metrics, 'LCSS_3D'),
                'EDR_3D': calculate_global(all_sub_metrics, 'EDR_3D'),
                'ERP_3D': calculate_global(all_sub_metrics, 'ERP_3D'),
                'Hausdorff_3D': calculate_global(all_sub_metrics, 'Hausdorff_3D'),
                'Action_Count': calculate_global(all_sub_metrics, 'Action_Count'),
                'Collision': None,  # 子轨迹无碰撞
                # 二维指标
                'SR_2D': calculate_global(all_sub_metrics, 'SR_2D'),
                'OSR_2D': calculate_global(all_sub_metrics, 'OSR_2D'),
                'NE_2D': calculate_global(all_sub_metrics, 'NE_2D'),
                'CESR': calculate_global(all_sub_metrics, 'CESR')
            }
            rows.append(global_sub)

        # 3. 全局子轨迹平均值平均
        if len(rows) > 0 and ref_type == 2:
            # 收集所有轨迹的子轨迹平均值
            sub_avg_rows = [row for row in rows if row['Type'] == 'Sub_Avg']
            
            if sub_avg_rows:
                sub_avg_mean = {
                    'Episode ID': 'Global',
                    'Type': 'Avg_of_All_Sub_Avg',
                    'PL': safe_nanmean([r['PL'] for r in sub_avg_rows]),
                    'Action_Count': safe_nanmean([r['Action_Count'] for r in sub_avg_rows]),
                    'SR_3D-5': safe_nanmean([r['SR_3D-5'] for r in sub_avg_rows]),
                    'SR_3D-10': safe_nanmean([r['SR_3D-10'] for r in sub_avg_rows]),
                    'SR_3D-15': safe_nanmean([r['SR_3D-15'] for r in sub_avg_rows]),
                    'SR_3D-20': safe_nanmean([r['SR_3D-20'] for r in sub_avg_rows]),
                    'SR_3D-30': safe_nanmean([r['SR_3D-30'] for r in sub_avg_rows]),
                    'SR_3D-40': safe_nanmean([r['SR_3D-40'] for r in sub_avg_rows]),
                    'OSR_3D-5': safe_nanmean([r['OSR_3D-5'] for r in sub_avg_rows]),
                    'OSR_3D-10': safe_nanmean([r['OSR_3D-10'] for r in sub_avg_rows]),
                    'OSR_3D-15': safe_nanmean([r['OSR_3D-15'] for r in sub_avg_rows]),
                    'OSR_3D-20': safe_nanmean([r['OSR_3D-20'] for r in sub_avg_rows]),
                    'OSR_3D-30': safe_nanmean([r['OSR_3D-30'] for r in sub_avg_rows]),
                    'OSR_3D-40': safe_nanmean([r['OSR_3D-40'] for r in sub_avg_rows]),
                    'NE_3D': safe_nanmean([r['NE_3D'] for r in sub_avg_rows]),
                    'NE-A': safe_nanmean([r['NE-A'] for r in sub_avg_rows]),
                    'SPL': safe_nanmean([r['SPL'] for r in sub_avg_rows]),
                    'CESR': safe_nanmean([r['CESR'] for r in sub_avg_rows]),
                    'PB': safe_nanmean([r['PB'] for r in sub_avg_rows]),
                    'CLS': safe_nanmean([r['CLS'] for r in sub_avg_rows]),
                    'nDTW_3D': safe_nanmean([r['nDTW_3D'] for r in sub_avg_rows]),
                    'Backward-nDTW_3D': safe_nanmean([r['Backward-nDTW_3D'] for r in sub_avg_rows]),
                    'Bi-nDTW_3D': safe_nanmean([r['Bi-nDTW_3D'] for r in sub_avg_rows]),
                    'sDTW_3D': safe_nanmean([r['sDTW_3D'] for r in sub_avg_rows]),
                    'LCSS_3D': safe_nanmean([r['LCSS_3D'] for r in sub_avg_rows]),
                    'EDR_3D': safe_nanmean([r['EDR_3D'] for r in sub_avg_rows]),
                    'ERP_3D': safe_nanmean([r['ERP_3D'] for r in sub_avg_rows]),
                    'Hausdorff_3D': safe_nanmean([r['Hausdorff_3D'] for r in sub_avg_rows]),
                }
                rows.append(sub_avg_mean)

        # ========== 全局区段均值统计 ==========
        seg_avg_rows = []
        if ref_type == 2:
            for seg_num in sorted(global_segment_store.keys()):
                metrics = global_segment_store[seg_num]
                seg_avg_rows.append({
                    'Episode ID': 'Global',
                    'Type': f'Seg{seg_num}_Avg',
                    'SR_3D-5': safe_nanmean(metrics['SR_3D-5']) if metrics['SR_3D-5'] else None,
                    'SR_3D-10': safe_nanmean(metrics['SR_3D-10']) if metrics['SR_3D-10'] else None,
                    'SR_3D-15': safe_nanmean(metrics['SR_3D-15']) if metrics['SR_3D-15'] else None,
                    'SR_3D-20': safe_nanmean(metrics['SR_3D-20']) if metrics['SR_3D-20'] else None,
                    'SR_3D-30': safe_nanmean(metrics['SR_3D-30']) if metrics['SR_3D-30'] else None,
                    'SR_3D-40': safe_nanmean(metrics['SR_3D-40']) if metrics['SR_3D-40'] else None,
                    'OSR_3D-5': safe_nanmean(metrics['OSR_3D-5']) if metrics['OSR_3D-5'] else None,
                    'OSR_3D-10': safe_nanmean(metrics['OSR_3D-10']) if metrics['OSR_3D-10'] else None,
                    'OSR_3D-15': safe_nanmean(metrics['OSR_3D-15']) if metrics['OSR_3D-15'] else None,
                    'OSR_3D-20': safe_nanmean(metrics['OSR_3D-20']) if metrics['OSR_3D-20'] else None,
                    'OSR_3D-30': safe_nanmean(metrics['OSR_3D-30']) if metrics['OSR_3D-30'] else None,
                    'OSR_3D-40': safe_nanmean(metrics['OSR_3D-40']) if metrics['OSR_3D-40'] else None,
                    'SPL': safe_nanmean(metrics['SPL']) if metrics['SPL'] else None,
                    'NE_3D': safe_nanmean(metrics['NE_3D']) if metrics['NE_3D'] else None,
                    'NE-A': safe_nanmean(metrics['NE-A']) if metrics['NE-A'] else None
                })

        # ========== 探索度分组均值统计 ==========
        explore_groups = {
            'explore-group-1': [],
            'explore-group-2': [],
            'explore-group-3': []
        }

        # 分类轨迹到探索度组别
        if ref_type == 2:
            for ep_id, data in results.items():
                ratio = data['full_metrics']['Explore_Ratio']
                if ratio <= 0.4:
                    explore_groups['explore-group-1'].append(data)
                elif ratio <= 0.6:
                    explore_groups['explore-group-2'].append(data)
                else:
                    explore_groups['explore-group-3'].append(data)

            # 计算各组的全局指标
            group_metrics = []
            for group_name, group_data in explore_groups.items():
                if not group_data:
                    continue

                # 提取完整轨迹指标
                full_metrics_list = [d['full_metrics'] for d in group_data]
                sub_metrics_list = [m for d in group_data for m in d['sub_metrics']]

                # 计算平均值
                def safe_mean(values, key):
                    valid = [v[key] for v in values if v[key] is not None]
                    return np.nanmean(valid) if valid else None

                # 计算完整轨迹指标平均值
                group_full_avg = {
                    'Group': group_name,
                    'Type': 'Full_Avg',
                    'Count': len(full_metrics_list),
                    'PL': safe_mean(full_metrics_list, 'PL'),
                    'Action_Count': safe_mean(full_metrics_list, 'Action_Count'),
                    'SR_3D-20': safe_mean(full_metrics_list, 'SR_3D-20'),
                    'SR_3D-30': safe_mean(full_metrics_list, 'SR_3D-30'),
                    'SR_3D-40': safe_mean(full_metrics_list, 'SR_3D-40'),
                    'OSR_3D-20': safe_mean(full_metrics_list, 'OSR_3D-20'),
                    'OSR_3D-30': safe_mean(full_metrics_list, 'OSR_3D-30'),
                    'OSR_3D-40': safe_mean(full_metrics_list, 'OSR_3D-40'),
                    'NE_3D': safe_mean(full_metrics_list, 'NE_3D'),
                    # 'NE-A': safe_mean(full_metrics_list, 'NE-A'),
                    'SPL': safe_mean(full_metrics_list, 'SPL'),
                    'CESR': safe_mean(full_metrics_list, 'CESR'),
                    'PB': safe_mean(full_metrics_list, 'PB'),
                    'CLS': safe_mean(full_metrics_list, 'CLS'),
                    'nDTW_3D': safe_mean(full_metrics_list, 'nDTW_3D'),
                    'sDTW_3D': safe_mean(full_metrics_list, 'sDTW_3D'),
                    'Backward-nDTW_3D': safe_mean(full_metrics_list, 'Backward-nDTW_3D'),
                    'Bi-nDTW_3D': safe_mean(full_metrics_list, 'Bi-nDTW_3D'),
                    'LCSS_3D': safe_mean(full_metrics_list, 'LCSS_3D'),
                    'EDR_3D': safe_mean(full_metrics_list, 'EDR_3D'),
                    'ERP_3D': safe_mean(full_metrics_list, 'ERP_3D'),
                    'Hausdorff_3D': safe_mean(full_metrics_list, 'Hausdorff_3D'),
                    'Collision': safe_mean(full_metrics_list, 'Collision'),
                    'Explore_Ratio': np.mean([d['full_metrics']['Explore_Ratio'] for d in group_data]),
                    'Explore_Rate': np.mean([d['full_metrics']['Explore_Rate'] for d in group_data])
                }
                group_metrics.append(group_full_avg)

                # 计算子轨迹指标平均值的平均值
                if sub_metrics_list:
                    group_sub_avg = {
                        'Group': group_name,
                        'Type': 'Sub_Avg',
                        'Count': len(sub_metrics_list),
                        'PL': safe_mean(sub_metrics_list, 'PL'),
                        'Action_Count': safe_mean(sub_metrics_list, 'Action_Count'),
                        'SR_3D-20': safe_mean(sub_metrics_list, 'SR_3D-20'),
                        'SR_3D-30': safe_mean(sub_metrics_list, 'SR_3D-30'),
                        'SR_3D-40': safe_mean(sub_metrics_list, 'SR_3D-40'),
                        'OSR_3D-20': safe_mean(sub_metrics_list, 'OSR_3D-20'),
                        'OSR_3D-30': safe_mean(sub_metrics_list, 'OSR_3D-30'),
                        'OSR_3D-40': safe_mean(sub_metrics_list, 'OSR_3D-40'),
                        'NE_3D': safe_mean(sub_metrics_list, 'NE_3D'),
                        'NE-A': safe_mean(sub_metrics_list, 'NE-A'),
                        'SPL': safe_mean(sub_metrics_list, 'SPL'),
                        'CESR': safe_mean(sub_metrics_list, 'CESR'),
                        'PB': safe_mean(sub_metrics_list, 'PB'),
                        'CLS': safe_mean(sub_metrics_list, 'CLS'),
                        'nDTW_3D': safe_mean(sub_metrics_list, 'nDTW_3D'),
                        'Backward-nDTW_3D': safe_mean(sub_metrics_list, 'Backward-nDTW_3D'),
                        'Bi-nDTW_3D': safe_mean(sub_metrics_list, 'Bi-nDTW_3D'),
                        'sDTW_3D': safe_mean(sub_metrics_list, 'sDTW_3D'),
                        'LCSS_3D': safe_mean(sub_metrics_list, 'LCSS_3D'),
                        'EDR_3D': safe_mean(sub_metrics_list, 'EDR_3D'),
                        'ERP_3D': safe_mean(sub_metrics_list, 'ERP_3D'),
                        'Hausdorff_3D': safe_mean(sub_metrics_list, 'Hausdorff_3D'),
                        'Collision': None,  # 子轨迹无碰撞指标
                        'Explore_Ratio': None,  # 子轨迹无探索度指标
                        'Explore_Rate': None  # 子轨迹无探索率指标
                    }
                    group_metrics.append(group_sub_avg)

            # 创建分组指标DataFrame
            group_df = pd.DataFrame(group_metrics, columns=[
                'Group', 'Type', 'Count', 'Explore_Rate', 'PL', 'Action_Count',
                'SR_3D-5', 'SR_3D-10', 'SR_3D-15', 'SR_3D-15-Collision', 'SR_3D-20', 'SR_3D-20-Collision',
                'OSR_3D-5', 'OSR_3D-10', 'OSR_3D-15', 'OSR_3D-15-Collision', 'OSR_3D-20', 'OSR_3D-20-Collision',
                'NE_3D', 'NE-A', 'Min_Distance', 'SPL', 'CESR', 'PB',
                'CLS', 'nDTW_3D', 'Backward-nDTW_3D', 'Bi-nDTW_3D', 'sDTW_3D',
                'LCSS_3D', 'EDR_3D', 'ERP_3D', 'Hausdorff_3D', 'Collision', 'Explore_Ratio'
            ])

        # ====== 排序逻辑 ======
        # 将seg_avg_rows合并到主rows列表
        rows += seg_avg_rows

        # 分离全局行和非全局行
        global_rows = [row for row in rows if row["Episode ID"] == "Global"]
        non_global_rows = [row for row in rows if row["Episode ID"] != "Global"]

        # 重新组合列表（全局行在前）
        sorted_rows = global_rows + non_global_rows

        # 创建最终的DataFrame
        df = pd.DataFrame(sorted_rows, columns=columns_order)

        # ========== （可选）方法间显著性（成对） ==========
        significance_summary = []
        significance_json = {}
        if 'has_pair' in locals() and has_pair:
            # 加载B方法，允许不同GT与ref_type
            if ref_type_b == 1:
                gt_data_b = load_reference_trajectory_type1(gt_folder_b)
            else:
                gt_data_b = load_groundtruth_type2(gt_folder_b)
            pred_data_b = load_predictions(mode_b, pred_path_b)

            # 按episode择优（与A相同逻辑）
            episode_groups_b = {}
            for ep_id_b, preds_b in pred_data_b.items():
                for pb in preds_b:
                    episode_groups_b.setdefault(ep_id_b, []).append({'key': ep_id_b, 'pred': pb, 'ep_id': ep_id_b})

            temp_results_b = {}
            for ep_id_b, trajectories_b in tqdm(episode_groups_b.items(), desc="处理每组轨迹[B]"):
                if ep_id_b not in gt_data_b:
                    continue
                best_traj_b = None
                best_metrics_b = None
                for i_b, traj_b in enumerate(trajectories_b):
                    try:
                        sub_b, full_b = evaluate_trajectory(gt_data_b[ep_id_b], traj_b['pred'], ref_type_b)
                        cur_b = {
                            'sr_3d_20': full_b.get('SR_3D-20', 0),
                            'osr_3d_20': full_b.get('OSR_3D-20', 0),
                            'ne_3d': full_b.get('NE_3D', float('inf')),
                            'sub_metrics': sub_b,
                            'full_metrics': full_b,
                            'traj_data': traj_b,
                            'index': i_b
                        }
                        if best_traj_b is None or is_better_trajectory(cur_b, best_metrics_b):
                            best_traj_b = traj_b
                            best_metrics_b = cur_b
                    except Exception:
                        continue
                if best_traj_b is not None and best_metrics_b is not None:
                    temp_results_b[ep_id_b] = best_metrics_b

            # 构造成对集合（A、B共有的episode）并在完整episode层次比较
            overlap_eps = sorted(set(results.keys()).intersection(set(temp_results_b.keys())))
            if overlap_eps:
                # 显著性比较指标（不含 PL 与 Steps）
                metrics_to_eval = ['SR_2D', 'SR_3D-20', 'OSR_3D-20', 'NE_3D', 'nDTW_3D', 'sDTW_3D']
                # 复用bootstrap设置
                try:
                    n_boot = int(os.environ.get('EVAL_BOOTSTRAP_N', '2000'))
                except Exception:
                    n_boot = 2000
                try:
                    ci_level = float(os.environ.get('EVAL_CI_LEVEL', '0.95'))
                except Exception:
                    ci_level = 0.95

                name_map = {}  # 无需重命名
                pair_method = str(PAIR_RESAMPLE_METHOD).lower()
                for m in metrics_to_eval:
                    a_vals = [results[e]['full_metrics'].get(m) for e in overlap_eps]
                    b_vals = [temp_results_b[e]['full_metrics'].get(m) for e in overlap_eps]
                    paired = [(a, b) for a, b in zip(a_vals, b_vals) if (a is not None and b is not None and not np.isnan(a) and not np.isnan(b) and not np.isinf(a) and not np.isinf(b))]
                    if not paired:
                        continue
                    a_arr = np.array([p[0] for p in paired], dtype=float)
                    b_arr = np.array([p[1] for p in paired], dtype=float)
                    diffs = a_arr - b_arr
                    stat_diff = bootstrap_stats(diffs, n_boot=n_boot, ci=ci_level)
                    if pair_method == 'bootstrap':
                        p_val = paired_bootstrap_p(diffs, n_boot=n_boot)
                        p_method = 'Bootstrap'
                    elif pair_method in ('perm', 'permutation', 'paired-permutation'):
                        p_val = paired_permutation_p(a_arr, b_arr, n_perm=n_boot)
                        p_method = 'Paired-Permutation'
                    else:
                        p_val = paired_signflip_p(diffs, n_perm=n_boot)
                        p_method = 'Sign-Flip'
                    disp_metric = name_map.get(m, m)
                    row = {
                        'Metric': disp_metric,
                        'N_pairs': stat_diff['n'],
                        'Mean_A': float(np.mean(a_arr)),
                        'Mean_B': float(np.mean(b_arr)),
                        'Mean_Diff(A-B)': stat_diff['mean'],
                        'Std_Diff': stat_diff['std'],
                        'SE_Boot_Diff': stat_diff['se_boot'],
                        'CI_Diff_Low': stat_diff['ci_low'],
                        'CI_Diff_High': stat_diff['ci_high'],
                        'CI_Level': stat_diff['ci_level'],
                        'P_Method': p_method,
                        'P_Value': p_val,
                        'Direction': 'higher-better' if m in ['SR_2D','SR_3D-20','OSR_3D-20','nDTW_3D','sDTW_3D'] else 'lower-better',
                        'Episodes_Overlap': len(overlap_eps)
                    }
                    significance_summary.append(row)
                    significance_json[disp_metric] = row

        # 嵌套JSON结构
        def convert_to_nested_json(df):
            """将平铺的DataFrame转换为嵌套的JSON结构"""
            nested = {"Global": {}, "episodes": {}}
            
            # 处理全局统计
            global_rows = df[df['Episode ID'] == 'Global']
            for _, row in global_rows.iterrows():
                metrics = row.dropna().to_dict()
                metric_type = metrics.pop('Type')
                nested["Global"][metric_type] = metrics
            
            # 处理每个Episode
            episode_rows = df[df['Episode ID'] != 'Global']
            for ep_id, group in episode_rows.groupby('Episode ID'):
                ep_dict = {}
                for _, row in group.iterrows():
                    row_dict = row.dropna().to_dict()
                    metric_type = row_dict.pop('Type')
                    ep_dict[metric_type] = row_dict
                nested["episodes"][ep_id] = ep_dict
            
            return nested

        # 转换为嵌套结构
        nested_data = convert_to_nested_json(df)
        if significance_summary:
            nested_data['Global']['Significance'] = significance_json

        # ========== 计算不确定性（Bootstrap置信区间/标准差/显著性） ==========
        # 从环境变量获取配置
        try:
            n_boot = int(os.environ.get('EVAL_BOOTSTRAP_N', '2000'))
        except Exception:
            n_boot = 2000
        try:
            ci_level = float(os.environ.get('EVAL_CI_LEVEL', '0.95'))
        except Exception:
            ci_level = 0.95

        null_map = None
        try:
            env_nulls = os.environ.get('EVAL_NULLS')
            if env_nulls:
                null_map = json.loads(env_nulls)
        except Exception:
            null_map = None

        # 要评估统计不确定性的指标（完整轨迹级别，按episode聚合）
        # 新增：PL 与 Steps（Action_Count）
        metrics_to_eval = ['SR_2D', 'SR_3D-20', 'OSR_3D-20', 'NE_3D', 'nDTW_3D', 'sDTW_3D', 'PL', 'Action_Count']

        def collect_metric(values, key):
            arr = []
            for v in values:
                if key in v:
                    arr.append(v.get(key))
            return arr

        uncertainty_summary = []
        uncertainty_json = {}
        name_map_unc = {'Action_Count': 'Steps'}
        for m in metrics_to_eval:
            vals = collect_metric(all_full_metrics, m)
            stats = bootstrap_stats(vals, n_boot=n_boot, ci=ci_level)
            p_val = None
            if null_map and m in null_map:
                try:
                    p_val = bootstrap_one_sample_p(vals, null_map[m], n_boot=n_boot)
                except Exception:
                    p_val = None
            disp_m = name_map_unc.get(m, m)
            row = {
                'Metric': disp_m,
                'N': stats['n'],
                'Mean': stats['mean'],
                'Std': stats['std'],
                'SE_Boot': stats['se_boot'],
                'CI_Low': stats['ci_low'],
                'CI_High': stats['ci_high'],
                'CI_Level': stats['ci_level'],
                'Null': (null_map[m] if (null_map and m in null_map) else None),
                'P_Value_TwoSided': p_val
            }
            uncertainty_summary.append(row)
            uncertainty_json[disp_m] = row

        # 合并到嵌套JSON
        nested_data.setdefault('Global', {})
        nested_data['Global']['Uncertainty'] = uncertainty_json

        # 保存为嵌套结构JSON
        json_output_path = os.path.splitext(output_file)[0] + '_structured.json'
        
        def json_serializer(obj):
            if isinstance(obj, np.generic):
                return float(obj.item())
            if isinstance(obj, float):
                if np.isinf(obj) or np.isnan(obj):
                    return None
            return obj
            
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(nested_data, f, indent=4, default=json_serializer, ensure_ascii=False)

        print(f"结构化JSON结果已保存至 {json_output_path}")

        # 将Results工作表保存为JSON格式
        # json_output_path = os.path.splitext(output_file)[0] + '_results.json'
        # df.to_json(json_output_path, orient='records', indent=4)
        # print(f"JSON格式的结果已保存至 {json_output_path}")
        
        # 创建Excel写入器
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Results')
        if ref_type == 2:
            group_df.to_excel(writer, index=False, sheet_name='Explore_Group_Results')

        # ========== 模式1：分类分工作表（按 Full 行，episode 任一分段包含此分类即计入） ==========
        if ref_type == 1 and segcat_mapping:
            # 将 (ep, seg_idx)->cats 映射转为 cat->episodes set
            cat_to_eps = defaultdict(set)
            for (ep, _seg), cats in segcat_mapping.items():
                for c in cats:
                    cat_to_eps[str(c)].add(str(ep))

            full_rows_df = df[df['Type'] == 'Full'].copy()
            # 数值列（用于计算均值）
            numeric_cols = [
                'PL','Action_Count','Explore_Rate',
                'SR_3D-5','SR_3D-10','SR_3D-15','SR_3D-20','SR_3D-30','SR_3D-40',
                'OSR_3D-5','OSR_3D-10','OSR_3D-15','OSR_3D-20','OSR_3D-30','OSR_3D-40',
                'NE_3D','NE-A','nDTW_3D','sDTW_3D','SPL','CESR','PB','CLS',
                'Backward-nDTW_3D','Bi-nDTW_3D','LCSS_3D','EDR_3D','ERP_3D','Hausdorff_3D',
                'SR_2D','OSR_2D','NE_2D','Explore_Ratio','Min_Distance'
            ]
            created_sheets_full = 0
            total_full_rows = 0
            category_columns_order_full = ['Episode ID', 'Type'] + [c for c in columns_order if c not in ('Episode ID', 'Type')]
            for cat_name, eps in cat_to_eps.items():
                if not eps:
                    continue
                cat_df = full_rows_df[full_rows_df['Episode ID'].astype(str).isin(eps)].copy()
                if cat_df.empty:
                    continue
                # 计算该分类的全局均值行
                avg_row: dict[str, object] = {'Episode ID': 'Global', 'Type': f'{cat_name}_Full_Avg'}
                for c in numeric_cols:
                    if c in cat_df.columns:
                        try:
                            vals = [v for v in cat_df[c].values if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
                            avg_row[c] = float(np.mean(vals)) if vals else None
                        except Exception:
                            avg_row[c] = None
                try:
                    cat_df_sorted = cat_df.sort_values(by=['Episode ID'], ascending=[True], kind='mergesort', na_position='last')
                except Exception:
                    cat_df_sorted = cat_df
                cat_df_out = pd.concat([pd.DataFrame([avg_row]), cat_df_sorted], ignore_index=True, sort=False)
                ordered_cols = [c for c in category_columns_order_full if c in cat_df_out.columns]
                ordered_cols += [c for c in cat_df_out.columns if c not in ordered_cols]
                cat_df_out = cat_df_out.reindex(columns=ordered_cols)
                sheet = _sanitize_sheet_name(f"CAT_{cat_name}_FULL")
                cat_df_out.to_excel(writer, index=False, sheet_name=sheet)
                ws = writer.sheets[sheet]
                ws.freeze_panes(1, 0)
                for col_num, value in enumerate(cat_df_out.columns.values):
                    try:
                        width = max(len(str(value)), cat_df_out[value].astype(str).map(len).max()) + 2
                    except Exception:
                        width = max(len(str(value)), 12)
                    ws.set_column(col_num, col_num, width)
                created_sheets_full += 1
                total_full_rows += len(cat_df)
            if created_sheets_full > 0:
                print(f"Full-level category worksheets created: {created_sheets_full}, total rows assigned: {total_full_rows}")

        # ========== 分类分工作表（仅 ref_type==2 且提供 segcat 映射） ==========
        if ref_type == 2 and segcat_mapping:
            # 通用数值列集合
            numeric_cols_common = [
                'PL','Action_Count','Explore_Rate',
                'SR_3D-5','SR_3D-10','SR_3D-15','SR_3D-20','SR_3D-30','SR_3D-40',
                'OSR_3D-5','OSR_3D-10','OSR_3D-15','OSR_3D-20','OSR_3D-30','OSR_3D-40',
                'NE_3D','NE-A','nDTW_3D','sDTW_3D','SPL','CESR','PB','CLS',
                'Backward-nDTW_3D','Bi-nDTW_3D','LCSS_3D','EDR_3D','ERP_3D','Hausdorff_3D',
                'SR_2D','OSR_2D','NE_2D','Explore_Ratio','Min_Distance'
            ]
            if CAT_SHEET_LEVEL_MODE2 == 'sentence':
                # 仅使用单句层次行（Type == 'Seg_<k>')
                sub_rows_df = df[df['Type'].astype(str).str.startswith('Seg_')].copy()
                # 解析 seg_idx
                def _parse_seg_idx(val):
                    try:
                        if isinstance(val, str) and val.startswith('Seg_'):
                            return int(val.split('_',1)[1])
                    except Exception:
                        return None
                    return None
                sub_rows_df['Seg_Idx'] = sub_rows_df['Type'].apply(_parse_seg_idx)
                # 将每行分配到多个类别
                cat_buckets = defaultdict(list)  # name -> list of dict rows
                for _, row in sub_rows_df.iterrows():
                    ep = str(row['Episode ID'])
                    seg_i = row['Seg_Idx']
                    if seg_i is None:
                        continue
                    cats = segcat_mapping.get((ep, seg_i), set())
                    for cat in cats:
                        cat_buckets[cat].append(row.drop(labels=['Seg_Idx']).to_dict())

                total_cat_rows = 0
                created_sheets = 0
                # 分类工作表列顺序：Episode ID、Segment、Type，然后参考总表顺序（去掉 Episode ID/Type，避免重复）
                category_columns_order = ['Episode ID', 'Segment', 'Type'] + [c for c in columns_order if c not in ('Episode ID', 'Type')]
                for cat_name, rows_list in cat_buckets.items():
                    if not rows_list:
                        continue
                    # 规范化每行：添加 Segment，移除 Seg_Idx（保留 Type）
                    processed_rows = []
                    for r in rows_list:
                        d = dict(r)
                        seg_val = None
                        if 'Seg_Idx' in d and d['Seg_Idx'] is not None:
                            seg_val = d['Seg_Idx']
                        else:
                            t = d.get('Type')
                            if isinstance(t, str) and t.startswith('Seg_'):
                                try:
                                    seg_val = int(t.split('_', 1)[1])
                                except Exception:
                                    seg_val = None
                        d['Segment'] = seg_val
                        d.pop('Seg_Idx', None)
                        processed_rows.append(d)
                    cat_df = pd.DataFrame(processed_rows)
                    # 添加该类别的全局均值行
                    avg_row = {'Episode ID': 'Global', 'Segment': None, 'Type': f'{cat_name}_Sub_Avg'}
                    for c in numeric_cols_common:
                        if c in cat_df.columns:
                            try:
                                vals = [v for v in cat_df[c].values if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
                                avg_row[c] = float(np.mean(vals)) if vals else None
                            except Exception:
                                avg_row[c] = None
                    # 将均值行置顶；其余按 Episode ID、Segment 升序稳定排序
                    try:
                        cat_df_sorted = cat_df.sort_values(by=['Episode ID','Segment'], ascending=[True, True], kind='mergesort', na_position='last')
                    except Exception:
                        cat_df_sorted = cat_df
                    cat_df_out = pd.concat([pd.DataFrame([avg_row]), cat_df_sorted], ignore_index=True, sort=False)
                    ordered_cols = [c for c in category_columns_order if c in cat_df_out.columns]
                    ordered_cols += [c for c in cat_df_out.columns if c not in ordered_cols]
                    cat_df_out = cat_df_out.reindex(columns=ordered_cols)
                    sheet = _sanitize_sheet_name(f"CAT_{cat_name}")
                    cat_df_out.to_excel(writer, index=False, sheet_name=sheet)
                    ws = writer.sheets[sheet]
                    ws.freeze_panes(1, 0)
                    for col_num, value in enumerate(cat_df_out.columns.values):
                        try:
                            width = max(len(str(value)), cat_df_out[value].astype(str).map(len).max()) + 2
                        except Exception:
                            width = max(len(str(value)), 12)
                        ws.set_column(col_num, col_num, width)
                    total_cat_rows += len(rows_list)
                    created_sheets += 1
                if created_sheets > 0:
                    print(f"Segment category worksheets created: {created_sheets}, total rows assigned: {total_cat_rows}")
            elif CAT_SHEET_LEVEL_MODE2 == 'full':
                # 构建 cat -> episode_ids 映射（任一分段命中该分类即计入）
                cat_to_eps = defaultdict(set)
                for (ep, _seg), cats in segcat_mapping.items():
                    for c in cats:
                        cat_to_eps[str(c)].add(str(ep))
                full_rows_df = df[df['Type'] == 'Full'].copy()
                created_sheets_full = 0
                total_full_rows = 0
                category_columns_order_full = ['Episode ID', 'Type'] + [c for c in columns_order if c not in ('Episode ID', 'Type')]
                for cat_name, eps in cat_to_eps.items():
                    if not eps:
                        continue
                    cat_df = full_rows_df[full_rows_df['Episode ID'].astype(str).isin(eps)].copy()
                    if cat_df.empty:
                        continue
                    # 全局均值行
                    avg_row: dict[str, object] = {'Episode ID': 'Global', 'Type': f'{cat_name}_Full_Avg'}
                    for c in numeric_cols_common:
                        if c in cat_df.columns:
                            try:
                                vals = [v for v in cat_df[c].values if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
                                avg_row[c] = float(np.mean(vals)) if vals else None
                            except Exception:
                                avg_row[c] = None
                    try:
                        cat_df_sorted = cat_df.sort_values(by=['Episode ID'], ascending=[True], kind='mergesort', na_position='last')
                    except Exception:
                        cat_df_sorted = cat_df
                    cat_df_out = pd.concat([pd.DataFrame([avg_row]), cat_df_sorted], ignore_index=True, sort=False)
                    ordered_cols = [c for c in category_columns_order_full if c in cat_df_out.columns]
                    ordered_cols += [c for c in cat_df_out.columns if c not in ordered_cols]
                    cat_df_out = cat_df_out.reindex(columns=ordered_cols)
                    sheet = _sanitize_sheet_name(f"CAT_{cat_name}")
                    cat_df_out.to_excel(writer, index=False, sheet_name=sheet)
                    ws = writer.sheets[sheet]
                    ws.freeze_panes(1, 0)
                    for col_num, value in enumerate(cat_df_out.columns.values):
                        try:
                            width = max(len(str(value)), cat_df_out[value].astype(str).map(len).max()) + 2
                        except Exception:
                            width = max(len(str(value)), 12)
                        ws.set_column(col_num, col_num, width)
                    created_sheets_full += 1
                    total_full_rows += len(cat_df)
                if created_sheets_full > 0:
                    print(f"Full-level category worksheets (mode2) created: {created_sheets_full}, total rows assigned: {total_full_rows}")
        # 写入不确定性工作表
        uncertainty_df = pd.DataFrame(uncertainty_summary, columns=[
            'Metric', 'N', 'Mean', 'Std', 'SE_Boot', 'CI_Low', 'CI_High', 'CI_Level', 'Null', 'P_Value_TwoSided'
        ])
        uncertainty_df.to_excel(writer, index=False, sheet_name='Uncertainty')
        # 写入显著性工作表
        if significance_summary:
            significance_df = pd.DataFrame(significance_summary, columns=[
                'Metric','N_pairs','Mean_A','Mean_B','Mean_Diff(A-B)','Std_Diff','SE_Boot_Diff','CI_Diff_Low','CI_Diff_High','CI_Level','P_Method','P_Value','Direction','Episodes_Overlap'
            ])
            significance_df.to_excel(writer, index=False, sheet_name='Significance')
        
        # 格式优化
        workbook = writer.book

        # 优化 "Results" 工作表
        results_worksheet = writer.sheets['Results']
        results_worksheet.freeze_panes(1, 0)  # 冻结第一行
        for col_num, value in enumerate(df.columns.values):
            col_width = max(len(str(value)), df[value].astype(str).map(len).max()) + 2
            results_worksheet.set_column(col_num, col_num, col_width)

        # 优化 "Uncertainty" 工作表
        uncertainty_ws = writer.sheets['Uncertainty']
        uncertainty_ws.freeze_panes(1, 0)
        for col_num, value in enumerate(uncertainty_df.columns.values):
            col_width = max(len(str(value)), uncertainty_df[value].astype(str).map(len).max()) + 2
            uncertainty_ws.set_column(col_num, col_num, col_width)
        # 优化 "Significance" 工作表
        if 'Significance' in writer.sheets:
            sig_ws = writer.sheets['Significance']
            sig_ws.freeze_panes(1, 0)
            cols = ['Metric','N_pairs','Mean_A','Mean_B','Mean_Diff(A-B)','Std_Diff','SE_Boot_Diff','CI_Diff_Low','CI_Diff_High','CI_Level','P_Method','P_Value','Direction','Episodes_Overlap']
            for i, name in enumerate(cols):
                sig_ws.set_column(i, i, max(len(name), 14))

        if ref_type == 2:
            # 优化 "Explore_Group_Results" 工作表
            group_worksheet = writer.sheets['Explore_Group_Results']
            group_worksheet.freeze_panes(1, 0)  # 冻结第一行
            for col_num, value in enumerate(group_df.columns.values):
                col_width = max(len(str(value)), group_df[value].astype(str).map(len).max()) + 2
                group_worksheet.set_column(col_num, col_num, col_width)

    # 保存文件
        writer.close()
        print(f"\n评估结果已保存至 {output_file}")

        # 在命令行输出统计信息（在生成Excel前）
        print("\n" + "="*50)
        print(f"真实轨迹总数: {total_gt}")
        print(f"预测轨迹总数: {total_pred}")
        print(f"成功计算轨迹数: {computed} (覆盖率: {computed/total_pred:.1%})")
        print("="*50 + "\n")
    else:
        print("\n无有效结果可导出")

def is_better_trajectory(current_metrics, best_metrics):
    """
    判断当前轨迹是否比最佳轨迹更好
    优先级：1) SR_3D-20高 2) OSR_3D-20高 3) NE_3D低
    """
    if best_metrics is None:
        return True
    if current_metrics['sr_3d_20'] > best_metrics['sr_3d_20']:
        return True
    elif current_metrics['sr_3d_20'] < best_metrics['sr_3d_20']:
        return False
    if current_metrics['osr_3d_20'] > best_metrics['osr_3d_20']:
        return True
    elif current_metrics['osr_3d_20'] < best_metrics['osr_3d_20']:
        return False
    if current_metrics['ne_3d'] < best_metrics['ne_3d']:
        return True
    return False

def compare_trajectories(better_metrics, worse_metrics):
    """
    返回轨迹优选的原因描述
    """
    if better_metrics['sr_3d_20'] > worse_metrics['sr_3d_20']:
        return f"SR_3D-20更高 ({better_metrics['sr_3d_20']:.3f} > {worse_metrics['sr_3d_20']:.3f})"
    elif better_metrics['osr_3d_20'] > worse_metrics['osr_3d_20']:
        return f"OSR_3D-20更高 ({better_metrics['osr_3d_20']:.3f} > {worse_metrics['osr_3d_20']:.3f})"
    elif better_metrics['ne_3d'] < worse_metrics['ne_3d']:
        return f"NE_3D更低 ({better_metrics['ne_3d']:.3f} < {worse_metrics['ne_3d']:.3f})"
    else:
        return "指标相近"

if __name__ == "__main__":
    main()