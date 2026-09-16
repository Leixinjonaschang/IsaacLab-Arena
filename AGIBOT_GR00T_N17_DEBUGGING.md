# AgiBot Stack Bowls × GR00T N1.7 评测排查日志

最后更新：2026-09-09（UTC）

本文档持续记录 `checkpoint-10000` 在 Isaac Lab-Arena `agibot_stack_bowls` 任务中的评测、故障定位、修复和反证。后续排查应继续更新此文件，不要只把结论留在终端输出或聊天记录中。

## 1. 维护约定

每次新增实验至少记录以下内容：

1. **排查点**：要验证或排除的具体假设。
2. **排查方式**：输入、对照组、指标、命令或脚本、随机种子。
3. **证据**：关键数值、日志、视频或 trace 路径。
4. **结论**：使用下面五种状态之一。
5. **影响/动作**：保留了什么修复，回滚了什么方案，下一步是什么。

状态定义：

- **已证实**：有隔离变量实验、oracle replay 或回归测试支持。
- **已排除**：实验结果明确不支持该假设。
- **部分成立**：确实存在并已改善，但不足以解释最终失败。
- **待验证**：目前只有现象或推断，没有足够的控制变量证据。
- **无效实验**：配置或输入有误，结果不得用于决策。

新增记录时应保留历史，不要把被否定的方案删除；被否定的尝试能防止后续重复走弯路。临时产物位于 `/tmp`，可能被系统清理；重要结果应迁移到 `results/` 或在此文档中保存关键汇总。

## 2. 当前结论摘要

截至 2026-09-06，结论如下：

| Pipeline 层级 | 当前判断 | 主要证据 |
|---|---|---|
| Docker / USD 资源加载 | 已解决 | 显式设置 `ARENA_LOCAL_ASSET_DIR` 后环境可以构建并进入 rollout |
| checkpoint 输入输出 contract | 已核对 | README、processor config、dataset metadata 与 Arena 请求的 key/shape/FPS 一致 |
| EEF 坐标与左右控制帧 | 已发现并修复 bug | 左臂固定 frame offset 与 tool-offset Jacobian 修复后，oracle EEF tracking 达毫米级 |
| action 解码与执行控制器 | 基本通过 | exact joint / exact EEF oracle 都可将碗抬起约 165 mm |
| gripper drive | 已修复 | close target 预载后，oracle placement error 从约 102 mm 降到约 3.5 mm |
| server seed / denoising options | 已修复 | reset 后相同 observation 可复现；请求的 8 步 denoise 得到 server 确认 |
| FPS / rollout phase | 15 Hz 正确；startup phase 是重要放大因素 | warmup 10 的首个 query 对应 row 10，标签动作尚未开始；改为 row 26 后共同阶段左臂误差均值下降 71% |
| state delay | 保留 1 step | dataset sidecar 与 proprioception 有 1-step offset；成对实验中 delayed state 更接近训练输入 |
| RTC | 已排除为当前解法 | 会把不准确的旧 chunk 尾部传播到下一 chunk 前缀，实测不利于抓取 |
| percentile clipping | 部分成立 | 能限制越界 relative XYZ，但不能消除 query-to-query drift |
| chunk 首帧平移跳变 | 已修复 | translation anchor 将每个新 chunk 的第 0 帧 XYZ 锚定到 observation state |
| camera key/order/processor contract | 基本通过 | 三路 view 顺序、RGB/uint8、crop/resize、token shape 和 state tensor 均已逐项核对 |
| 当前闭环失败的主导因素 | **startup phase + 视觉输入分布偏移/敏感性** | warmup 26 降低 71% 误差；再将 live camera 换成 dataset camera 后降低 69%，两者合计相对旧基线降低约 91% |
| 最终任务成功率 | 仍未解决 | 修复前的完整 episode 均失败；最新修改只做过 300-step diagnostics，尚无最新完整 episode |

因此当前不应继续把主要精力放在 IK 增益或坐标转换上。camera 的字段和 processor contract 已基本通过；下一阶段应优先恢复**训练数据生成时的实际 renderer/camera runtime provenance**，尤其是 right wrist view，并判断剩余像素差异是可修复的 eval pipeline mismatch，还是 checkpoint 对合法的小渲染差异和闭环扰动缺乏鲁棒性。

## 3. 系统与基线

### 3.1 代码、模型和数据

- Workspace：`/home/ubuntu/projects/isaaclab_arena_proj`
- Arena repo：`/home/ubuntu/projects/isaaclab_arena_proj/IsaacLab-Arena`
- Arena 基准 commit：`d819b42307ef9d6b84b24d3eb93950cacc3b8fa8`，工作树包含本文记录的未提交修改。
- Isaac-GR00T 子模块基准 commit：`23ace64f17aa5015259b8609d371eb61a357c776`，子模块包含 server/RTC/denoise/clip 的未提交修改。
- Docker container：`isaaclab_arena-latest`
- Docker image：`sha256:a8f53bf18a06204ae71cf1ae77e858be4148bde0e102512e27aee05d525be113`
- Checkpoint：`/home/ubuntu/projects/isaaclab_arena_proj/checkpoints/gr00t_n17_sfted_stack_bowls_with_ee_pose/checkpoint-10000`
- 训练数据：`/home/ubuntu/projects/isaaclab_arena_proj/data_process/data/stack_bowls`
- 主要分析 episode：`episode_000078`；也使用 episode 0 和整个 200-episode 数据集做统计。
- 正式 experiment config：`isaaclab_arena_environments/experiment_configs/agibot_stack_bowls_gr00t_n17_experiment.yaml`
- Arena-side policy config：`isaaclab_arena_gr00t/policy/config/agibot_n17_gr00t_closedloop_config.yaml`

数据目录只用于读取和比较，排查过程中没有改写 parquet 或原始视频。

### 3.2 Checkpoint contract

从 checkpoint `README.md`、`processor_config.json` 和 `experiment_cfg/config.yaml` 核对出的 contract：

- 数据频率：15 Hz。
- 视频输入：当前时刻的 `ego_view`、`left_wrist_view`、`right_wrist_view`。
- 原始视频：`512 × 512 × 3`、`uint8`、RGB。
- state：38 维，包括双臂 EEF 9D、双手和双臂关节。
- EEF state：`base_link` 下的绝对 `XYZ + rot6d`。
- action：`40 × 24`，即 40 个未来控制步。
- EEF action：相对当前 EEF pose 训练，由 processor 解码回绝对 `base_link` pose。
- hand action：绝对目标。
- language key：`annotation.human.action.task_description`。
- 训练语言：`Stack the bowls together.`
- checkpoint 内部默认 denoising steps 为 4；后续实验发现 8 更稳定。
- 训练只更新 projector、diffusion action head 和 VLLN；视觉 backbone 冻结。

### 3.3 当前推荐 Arena 配置

- `action_mode: gr00t_diffik`
- `action_horizon: 40`
- `action_chunk_length: 16`
- `action_sample_count: 5`，逐元素 median。
- `denoising_steps: 8`
- `action_chunk_translation_anchor_alpha: 1.0`
- `initial_camera_warmup_steps: 10`（当前正式配置基线；26-step 候选已在短对照中明显改善，但尚未晋升为正式默认值）
- 使用从 demonstration startup 计算出的双臂 ready pose。
- `state_delay_steps: 1`
- `rtc_enabled: false`
- `seed: 10`
- server：`127.0.0.1:5557`

注意：截至本次更新，端口 5557 的长期运行 server 是部分 server patch 之前启动的。正式复测前必须重启，否则进程不会加载磁盘上的 seed、denoise、RTC 和 percentile clipping 修改。

## 4. 评测启动与资源问题

### 4.1 `robodojo_table.usda` 无法打开

**排查点**

评测在环境构建阶段失败：

```text
Failed to open layer @/home/ubuntu/playground/objects/arena_local/robodojo_table/robodojo_table.usda@
```

**排查方式**

- 沿 traceback 检查 `agibot_stack_bowls_environment.py -> background_library.py -> Object -> Usd.Stage.Open`。
- 检查 `isaaclab_arena/assets/local_objects.py`，确认默认路径是宿主机风格的 `/home/ubuntu/playground/objects/arena_local`。
- 在 eval 容器内检查目标 USD 是否存在，以及容器实际挂载的 local asset 根目录。
- 使用环境变量覆盖默认路径，再进行环境 build smoke test。

**结论：已证实并解决**

这不是 policy 或 Isaac USD parser 的问题，而是容器中的 local asset 路径没有覆盖默认宿主路径。当前运行约定：

```bash
ARENA_LOCAL_ASSET_DIR=/tmp/isaaclab_arena_agibot_assets/arena_local
```

并确保该目录在容器里包含：

```text
robodojo_table/robodojo_table.usda
```

设置后环境可以完成构建并进入 rollout。

日志中的以下 warning 不是此次失败原因：

- `pxr.Semantics is deprecated`
- `omni.hydra was already registered`
- Matplotlib 对 `/home/ubuntu/.config/matplotlib` 无写权限
- ONNX Runtime 无法从 `/sys/class/drm/card0` 读取 vendor

Matplotlib warning 使用下面的可写目录消除：

```bash
MPLCONFIGDIR=/tmp/matplotlib
```

### 4.2 容器内 camera-video eval 命令

在 `/workspaces/isaaclab_arena` 中运行：

```bash
ACCEPT_EULA=Y \
OMNI_KIT_ACCEPT_EULA=YES \
ARENA_LOCAL_ASSET_DIR=/tmp/isaaclab_arena_agibot_assets/arena_local \
HF_HOME=/tmp/isaaclab_arena_agibot_assets/hf \
MPLCONFIGDIR=/tmp/matplotlib \
./submodules/IsaacLab/_isaac_sim/python.sh \
  isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/experiment_configs/agibot_stack_bowls_gr00t_n17_experiment.yaml \
  --headless \
  --record_camera_video \
  --output_base_dir /eval/agibot_eval_output
```

`--record_camera_video` 只会为完成的 episode 产出正式 MP4。使用 `rollout_limit.num_steps: 300` 的短诊断没有完成 episode，因此通常只有 report/trace，没有 episode camera video。

## 5. 排查过程

### 5.1 Pipeline contract 与 action/state 字段核查

**排查点**

Arena 是否向 server 发送了错误的 modality key、shape、dtype、相机顺序、语言或 action 维度。

**排查方式**

- 阅读 checkpoint `README.md`、`processor_config.json`、`statistics.json` 和训练配置。
- 对照 Arena 的 `Gr00tRemoteClosedloopPolicy`、joint mapping YAML 和 `build_gr00t_action_tensor`。
- 在 remote-client 边界保存 inference trace，记录发送给 server 的 video/state/language、原始 action samples、处理后的 action 和 sim tensor。
- 为 AgiBot contract 添加单元测试，固定三路相机顺序、state/action keys、shape、dtype 和 20D sim action layout。

**结论：已核对**

当前 key、shape、dtype、语言和相机排列与 checkpoint 一致。server 返回 40×24，Arena 将其转换为 20D sim action：

```text
[left XYZ + quat xyzw (7), left hand (3), right XYZ + quat xyzw (7), right hand (3)]
```

这一步没有发现能解释闭环灾难性退化的字段错位。

### 5.2 直接用训练数据请求 server（teacher forcing）

**排查点**

模型/checkpoint 本身是否能在训练分布上的 observation 产生合理 action；以及应如何比较 prediction 和 dataset action。

**排查方式**

- 从 parquet 读取同一时刻 state/action，从对应 MP4 解码三路 RGB。
- 构造与真实 PolicyClient 相同的 observation 请求 server。
- 不把所有 24 维简单混成一个 MSE；分别计算：
  - 左/右 EEF XYZ 的逐步欧氏误差；
  - 左/右 rotation geodesic error；
  - 左/右 hand MAE；
  - prefix 1、8、16、24、40 的误差；
  - 多个 diffusion sample 的方差。
- 对 episode 0、episode 78 和抓取关键阶段做检查。

**证据**

- checkpoint README 自带的 open-loop sanity：unnormalized MSE 约 `1e-4`，MAE 约 `0.006–0.008`。
- exact episode-78、denoise=8、5 samples median：
  - prefix 8：左/右位置误差 `6.20 / 5.38 mm`；
  - prefix 16：`12.53 / 6.32 mm`；
  - prefix 40：`19.56 / 8.78 mm`。
- denoise=4 的 prefix-8 左/右误差约 `9.12 / 10.98 mm`，比 denoise=8 差。
- 20 个同 observation sample 的位置不确定性随 horizon 增长：左臂 step 0/7/39 标准差约 `0.95 / 6.37 / 12.21 mm`；右臂约 `4.05 / 5.47 / 6.22 mm`。

**结论：部分成立**

checkpoint 在训练 observation 上并非失效，短前缀输出大体合理；但 diffusion 有明显随机性，尤其左臂长 horizon 尾部不稳定。因此闭环策略不能盲目执行全部 40 步，且需要固定 seed、增加 denoise、median sampling 或缩短执行 chunk。

相关临时结果：

- `/tmp/agibot_teacher_forcing_20260905/teacher_forcing.json`
- `/tmp/agibot_teacher_forcing_20260905/critical_phase_teacher_forcing.json`
- `/tmp/agibot_teacher_forcing_20260905/action_stochasticity_trace0.json`

### 5.3 EEF 坐标约定错误

**排查点**

训练数据中的 EEF frame、仿真 observation frame 和 controller command frame 是否是同一坐标约定。

**排查方式**

- 从同一 demonstration 的 joint state 做 FK，并与 parquet 中 `left_eef_9d` / `right_eef_9d` 比较。
- 比较 raw USD `gripper_center` frame 和实际 controller frame。
- 对 `XYZ + rot6d -> quaternion -> XYZ + rot6d` 做 round-trip。
- 使用 dataset exact joint action、dataset exact EEF action 分别做 oracle replay；如果 joint replay 正确而 EEF replay 错误，则问题位于 EEF 变换或 Cartesian controller，而不是数据轨迹。

**发现**

- 最终训练数据保存的是 raw USD gripper-center orientation。
- 左臂 controller frame 相对 raw frame 有固定 offset：`[0, -0.7071, 0, 0.7071]`（xyzw）。
- 右臂不需要该 offset。
- 原 pipeline 未在 observation/action 两个方向对左臂做成对转换。
- upstream DiffIK 对带 rotational body offset 的 Jacobian 处理也不正确：它旋转了 angular rows；刚性连接两点在同一 base 表达下 angular velocity 相同，平移 offset 只应修正 linear rows。

**修复**

- 新增 `isaaclab_arena_gr00t/utils/agibot_eef.py`：
  - `control_pose_to_training_eef_9d`
  - `training_eef_9d_to_control_pose`
- observation 端将 runtime control pose 转回 training raw frame。
- action 端将 checkpoint EEF-9D 转为 controller frame。
- 自定义 DiffIK 的 offset Jacobian：

```text
J_linear(frame) = J_linear(body) - skew(r) @ J_angular(body)
J_angular(frame) = J_angular(body)
```

**验证证据**

- episode-78 exact joint oracle：
  - 左 EEF mean error `4.482 mm`；
  - joint RMS error `0.462°`；
  - bowl lift `166 mm`。
- episode-78 exact EEF oracle：
  - 左 EEF mean / p95 / max error：`1.506 / 3.809 / 36.083 mm`；
  - rotation error：`0.441°`；
  - bowl lift：`164.7 mm`。
- 坐标 round-trip、action tensor layout 和 translation-anchor tests 均通过。

**结论：已证实并修复**

这是最早阶段的主要真实 pipeline bug。修复后 Cartesian oracle 能重现 demonstration 的抓取/抬升，证明坐标变换和 EEF 执行方向已经一致。但模型闭环仍失败，因此还有独立问题。

### 5.4 Reset pose、startup motion 与首帧 camera freshness

**排查点**

模型第一次看到的 state/video 是否是训练中不存在的组合。

**排查方式**

- 统计全部 demonstration 的最初若干帧。
- 对比 sim reset pose、dataset episode-initial joint reference 和模型第一个有效 observation。
- 检查 mounted wrist camera 在 articulation reset 后是否仍返回 reset 前缓存帧。
- 分别测试无 warmup、纯 hold warmup、移动到 demonstration ready EEF 后 warmup。

**发现**

- demonstration 包含确定性的 startup transition：左 EEF 在 frame 3 前移动约 `44 mm`，双臂约在 frame 9 后稳定。
- 单纯在 raw reset pose 等待相机更新，会形成训练中没有出现的“新图像 + raw pose”组合。
- mounted camera 需要在 reset 后显式 render/update，不能直接信任 reset 返回的第一帧。

**修复**

- 为环境增加 `policy_ready_arm_joint_positions`，在 reset event 中写入记录的 14 个双臂绝对关节角。
- 保留 articulation 原始 default joint reference，使 `joint_pos_rel` 仍与数据采集约定一致；不能把 ready pose 直接改成新的 default。
- policy 增加 `initial_camera_warmup_steps: 10` 和 `initial_agibot_ready_eef_9d`。
- 每个控制步记录 video history；相机从 post-reset pose 强制刷新。

**结论：部分成立并已修复**

初始 observation contract 得到明显改善，是正确的 pipeline 修复；但它不能阻止第 2–3 次 replan 后左臂预测发散。

### 5.5 Video/state 时间对齐

**排查点**

数据集 sidecar video 和 parquet state 是否存在一个控制步的偏移；在线是否应该使用 current state 或 delayed state。

**排查方式**

- 用 wrist 图像运动量与 EEF translation/rotation motion 做 lag cross-correlation。
- 在相同 live frame、相同 server seed/noise 下，只替换 delayed/current state。
- 保存同一 inference 的 `policy__*`（实际送 server 的 delayed state）和 `current_policy__*`（当前 observation），直接计算它们与 dataset frame 的距离。

**证据**

- dataset episode-0 的两侧 wrist sidecar 相对记录 state 呈 `+1` step，相关系数左/右约 `0.842 / 0.917`。
- 成对 live probe：
  - delayed state：左 prefix 8/40 error `16.8 / 224.7 mm`；
  - current state：`18.4 / 226.3 mm`，更差；
  - 与 dataset state 的距离：左 delayed/current `7.9 / 9.5 mm`，右 `2.32 / 5.51 mm`。

**结论：保留 `state_delay_steps: 1`**

一个控制步 delay 符合训练数据契约，并且 A/B 中没有导致现有发散。它不是当前主要 bug。

相关 trace：`/tmp/agibot_gr00t_n17_delay_pair_probe/traces`。

### 5.6 控制器与 action execution 隔离

**排查点**

模型 action 是否合理，但 RMPFlow/IK/actuator 执行错误。

**排查方式**

- 将模型完全移出回路，使用 exact dataset joint trajectory 和 exact EEF trajectory 做 oracle replay。
- 比较 command EEF、measured EEF、joint branch、bowl height、gripper state。
- 对 RMPFlow、低通 RMPFlow、DiffIK 做对照。
- 用同一组“错误但固定”的模型 chunk 比较不同 nullspace/joint-limit 策略；再用 expert trajectory 检查策略是否损伤正确轨迹。

**发现和修复**

- stock RMPFlow 的 nominal-cspace attractor 会选择不同的冗余关节分支；内部 target 快速反转也会让 arm drive 震动。
- 新增 control-rate DiffIK：每个 15 Hz control step 只解一次 IK，在 120 Hz physics substeps 间插值并持有该 branch。
- joint correction 使用逐关节 clamp；统一缩放整个向量会让一个饱和关节拖慢其他关节，已否定。
- nullspace 使用真正的 Moore–Penrose projector `I - J⁺J`。
- 只对第 7 个冗余关节施加 reference bias；对全部 7 关节施加 nullspace posture cost 会显著损伤 expert trajectory，已否定。

**证据**

- 旧 controller 在固定错误模型 action 上：左 EEF mean/p95 `9.91 / 52.01 mm`，第 3 query error `42.26 mm`，joint 3 可翻转约 `160°`。
- joint-7-only nullspace 后，同一错误 action 的第 3 query error 降到 `19.93 mm`。
- 对 expert trajectory：左 EEF mean/p95 `1.557 / 5.248 mm`，bowl lift `164.3 mm`，没有破坏正确执行。

**结论：执行链路基本通过**

控制器原本确有 branch continuity 问题，现有 DiffIK 修复有效。oracle 可以完成抓取/抬升，所以后续模型闭环失败不能主要归因于执行器。

### 5.7 Gripper close target

**排查点**

模型输出 hand close 后，position drive 是否保持足够夹持力。

**排查方式**

- 用同一 demonstration replay，对比 raw close target `0.0` 与物理限位之外的小负 target。
- 测量 bowl 在 placement 时相对目标轨迹的误差。

**证据**

- close target 为 `0.0` 时 placement error 约 `102 mm`。
- 将 `[0, 0.994]` 模型输出映射为 `[-0.1, 0.994]` drive target 后，placement error 约 `3.5 mm`。
- 物理 joint 仍停在 0；负 target 只让 drive 在抓住物体后持续预载。

**结论：已证实并修复**

保留 gripper preload mapping。该修复保证 oracle 抓取稳定，但无法让模型在错误视觉轨迹上主动进入抓取阶段。

### 5.8 Server 随机性、reset seed 与 denoising steps

**排查点**

相同输入为何跨 rollout 或跨 query 给出明显不同的 diffusion action；Arena 配置的 denoise 是否真正传到 N1.7 action head。

**排查方式**

- 对完全相同的 serialized observation 连续请求多次，统计 sample-to-median 和 horizon-wise std。
- 在 client reset 前后重复相同请求。
- 检查 `PolicyClient.get_action(options=...)` 到 `Gr00tPolicy._get_action` 再到 `Gr00tN1d7ActionHead.get_action` 的调用链。
- 要求 server 在 info 中回传实际使用的 `num_inference_timesteps`，Arena 对其做 assert。

**发现**

- flow matching 从 `torch.randn` 开始；长期运行 server 不 reset RNG 会让相同 rollout 不可复现。
- 原 N1.7 action head 使用构造时的默认步数，没有消费每次请求的 denoise override。

**修复**

- `Gr00tPolicy.reset(options={"seed": ...})` 调用 `torch.manual_seed`。
- Arena policy reset 将 seed 发给 server。
- action head 读取 `options["num_inference_timesteps"]` 并校验为正数。
- server info 返回有效 denoise steps；Arena 未收到确认就 fail fast。

**结论：已证实并修复**

reset 后相同 observation 的输出可以确定性复现。denoise=8 的 teacher-forcing prefix 比 4 步更好，因此正式配置使用 8。

### 5.9 多样本 median

**排查点**

是否可以通过多个 diffusion sample 抑制单次采样的异常值。

**排查方式**

- 同 observation 请求多次。
- 对每个 action key、每个 horizon、每个维度做 elementwise median。
- 对比 1 sample 与 5 samples teacher forcing error，并确保 dtype 保持 `float32`。

**结论：部分成立，保留**

5-sample median 能降低训练分布上的短前缀误差，因此正式配置保留 `action_sample_count: 5`。它提高推理成本，也无法修复 live visual input 触发的系统性错误轨迹。

### 5.10 RTC（Real-Time Chunking）

**排查点**

把旧 action horizon 未执行尾部作为下一次 diffusion inpainting prior，是否能提高 chunk 连续性。

**排查方式**

- 在 Arena request 中加入旧 action。
- 在 server processor 中把旧绝对 action 相对当前 state 重新编码。
- 因 checkpoint 每个 horizon index 的 normalization statistics 不同，对 source/destination timestep 做重新对齐。
- 对 frozen prefix 在 normalized space 和 decoded physical space分别检查 max error。
- 比较 RTC frozen-8、非 frozen RTC 和 fresh chunk。

**发现**

- 不做 per-timestep stats 对齐时，数值上复制 normalized tail 并不保持相同的物理 action。
- 即使修好 contract，当前 checkpoint 的长 horizon 尾部本身不准确；RTC 会把该错误尾部传播到下一次可执行前缀。
- RTC rollout 没有改善抓取，反而更容易延续错误方向。

**结论：实现已修正，但作为当前策略已排除**

保留 RTC 实现与 diagnostic config，正式 AgiBot stack-bowls 配置设为 `rtc_enabled: false`。

### 5.11 Relative EEF percentile clipping

**排查点**

异常轨迹是否主要来自 diffusion relative XYZ 超过训练 q01/q99 envelope。

**排查方式**

- 在 server 的 normalized action 解码前，只 clip `ActionRepresentation.RELATIVE + ActionType.EEF` 的 XYZ。
- 每个 key、每个 horizon index 使用 checkpoint `statistics.json` 的 q01/q99，并转换到对应 normalized range。
- 不 clip rotation、hand 或 absolute action。
- 记录被 clip 的比例，并跑短闭环对照。

**结论：部分成立，但不是根因**

clipping 能防止单步输出离开训练 envelope，属于安全边界；但每个 query 都可以在 envelope 内朝同一错误方向累积，因此无法阻止跨 query drift。

### 5.12 Chunk 长度选择

**排查点**

执行 40、24、16 或 8 步后重新推理，哪个能在 prediction accuracy 和任务 phase progress 之间取得平衡。

**排查方式**

- teacher forcing 比较不同 prefix。
- live rollout 对比 chunk 8、16、40。
- 观察每次 query endpoint、下一个 query state、hand close 是否出现以及 bowl 是否移动。

**证据**

- 40-step 尾部误差明显大于短前缀，不适合整段执行。
- 正确的 chunk-8 + anchor run：左臂 query endpoint error 约 `3.0 -> 5.7 -> 29.3 -> 64.7 mm`，之后继续发散；hand minimum `0.9707`，没有进入有效闭合，碗未移动。
- chunk-16 能更快推进 demonstration phase，同时避开最差的长尾；仍会在第 2–3 query 后因视觉闭环偏移发散。

**结论：保留 chunk 16**

chunk 8 并没有解决根因，且频繁 replan 让模型一直停留在 open-hand phase。chunk 16 是当前较合理的折中，不代表任务已解决。

**无效实验记录**

第一次临时 chunk-8 YAML 将左臂 joint 6 的 ready pose 符号写错。该 run 的所有结果已标记无效，不用于上述判断。只有修正后的 `/tmp/agibot_gr00t_n17_joint7_clip_anchor_chunk8_300_corrected` 可用。

### 5.13 Action chunk 首帧平移锚定

**排查点**

训练 label 的 action[0] 是否通常等于当前 state；模型每次 replan 的整体 XYZ bias 是否造成不必要的首帧跳变和累积 drift。

**排查方式**

- 扫描 200 个 stack-bowls parquet，共 202,189 帧，统计 `action[t, 0].XYZ - state[t].XYZ`。
- 对 teacher forcing action chunk 做 alpha sweep：从每个 horizon 的 XYZ 中减去 `alpha × (predicted_xyz[0] - observed_xyz)`。
- alpha=1 时保留所有 within-chunk translation delta、rotation、hand 和 raw server samples，只移动整条 XYZ 曲线的原点。
- 在真实 300-step closed loop 中记录 query-to-query target tracking。

**数据集统计**

| Arm | Mean | Median | P95 | P99 | `≤ 1 mm` |
|---|---:|---:|---:|---:|---:|
| Left | 2.891 mm | 0.186 mm | 13.715 mm | 40.687 mm | 73.24% |
| Right | 2.896 mm | 0.202 mm | 14.462 mm | 20.543 mm | 73.11% |

**Teacher-forcing alpha sweep（position error，mm）**

| Alpha | Left 8/16/40 | Right 8/16/40 |
|---:|---:|---:|
| 0.0 | 6.20 / 12.53 / 19.56 | 5.38 / 6.32 / 8.78 |
| 0.5 | 5.04 / 11.42 / 18.89 | 3.05 / 3.97 / 6.46 |
| 1.0 | 5.74 / 11.57 / 19.00 | 1.93 / 2.47 / 4.64 |

**真实 chunk-16 + anchor 300-step 结果**

- 每个新 chunk 的第一个 XYZ jump：按定义为 0。
- 右臂跨 query tracking：median `0.262 mm`、p95 `0.914 mm`、max `1.419 mm`。
- 右臂 chunk endpoint error median：`6.79 mm`。
- 左臂 chunk endpoint error：median `29.46 mm`、p95 `313.5 mm`、max `420.8 mm`。
- 左臂跨 query target tracking：median `7.89 mm`、p95 `249.9 mm`、max `268.7 mm`。
- left hand 最低到 `0.252`，但碗仍未移动。

**结论：首帧偏置已修复，但只部分解决问题**

保留 `action_chunk_translation_anchor_alpha: 1.0`。它显著修复了右臂和每次 replan 的首帧不连续，但左臂 within-chunk trajectory 本身仍会被错误视觉输入推向错误方向，anchor 不能改变轨迹形状。

相关 trace：`/tmp/agibot_gr00t_n17_joint7_clip_anchor_chunk16_300/traces`。

### 5.14 Dataset/live observation 精确对齐

**排查点**

闭环第几次 query 开始离开 demonstration；偏差先出现在 state、camera、model prediction 还是 controller tracking。

**排查方式**

- 使用 episode-78 的相同初始 scene/reset。
- 每次 live query 找到预期 dataset row，比较 state、三路 raw uint8 camera、prediction prefix 和 measured trajectory。
- 固定 server seed，并确保为了得到 q1 使用与真实 q0 相同的 RNG advance。

**证据**

- q0 live state 与 dataset 精确一致，左臂 prefix-8 prediction error `2.3 mm`。
- q1 live 左 EEF state 只偏离约 `8 mm`，但 prediction 已变为 prefix-8 `19.2 mm`、full-40 `230.1 mm`。
- q2 state 偏离约 `190 mm`，此后进入完全 out-of-distribution 状态。
- q1 raw uint8 image MAE：head `5.09`、left wrist `10.11`、right wrist `3.53`。

**结论：已证实发散顺序**

控制器并非先失控。第一个 chunk 后只有有限 state/image 偏差，第二个 query 的模型长轨迹先发生灾难性变化，执行该轨迹后 state 才大幅偏离。

### 5.15 Video/state modality ablation

**排查点**

q1 prediction 变化主要由 video 还是 state 引起；哪一路 camera 最敏感。

**排查方式**

- 固定 q1 的目标 dataset action、server seed 和 diffusion noise。
- 混合 dataset/live inputs：三路 video 全换、state 全换、只换 EEF state、只换 joint/hand state、逐路换 camera。
- 记录左/右 prefix-8 与 full-40 position error。

**关键结果（q1，mm）**

| Input variant | Left 8/40 | Right 8/40 |
|---|---:|---:|
| Dataset video + dataset state | 1.8 / 39.0 | 2.2 / 7.0 |
| Live video + live state | 19.2 / 230.1 | 5.2 / 10.7 |
| Live video + dataset state | 11.1 / 226.6 | 1.8 / 8.7 |
| Dataset video + live state | 9.4 / 44.1 | 5.8 / 9.9 |
| Dataset inputs，仅换 live EEF state | 9.3 / 43.9 | — |
| Dataset inputs，仅换 live joint/hand state | 1.8 / 39.1 | — |
| Dataset inputs，仅换 live head | 1.9 / 34.0 | — |
| Dataset inputs，仅换 live left wrist | 2.3 / 29.3 | — |
| Dataset inputs，仅换 live right wrist | 3.0 / 86.3 | — |

“—”表示该轮汇总没有保存足够可靠的对应右臂简表值，不应补猜。

**结论：已证实 video 是长 horizon 发散的主导触发因素**

- live state 会放大左臂短前缀误差，主要来自 EEF state，而不是 joint/hand state。
- 但 full-40 从约 39 mm 增至约 230 mm 主要由 live video 触发。
- 三路 camera 都有影响，且不能只凭单视角结果断言某一路单独是根因；right-wrist-only 在该 query 对左臂 long horizon 影响尤其大。

相关结果：`/tmp/agibot_oracle_live_modality_ablation_20260906.json`。

### 5.16 FPS 与 phase mismatch

**排查点**

prediction 看似错误是否只是把正确动作预测到了错误的时间 phase；或 sim/action 控制频率不等于训练 15 Hz。

**排查方式**

- 检查 dataset `meta/info.json` 的 fps。
- 检查 sim `dt = 1/120` 和 decimation 8，得到 control rate `120/8 = 15 Hz`。
- 对 live q1 三路 camera 与 dataset 全时序搜索最近帧。
- 将预测的 40-step path 与 episode-78 所有可能 phase 做 nearest-trajectory search，而不是只与预期 row 26 比较。

**证据**

- dataset 和 sim 都是 15 Hz。
- live q1 的三路 camera 最近 dataset frames 为 23–29，覆盖预期 frame 26，未显示固定大 phase shift。
- predicted 40-step path 即使允许匹配 episode-78 的任意 phase，combined error 仍约 `102 mm`。
- 左 predicted displacement 随 horizon 快速增长：step 0 约 0，step 7 约 23 mm，step 15 约 160 mm，step 29 约 450 mm。

**结论：部分排除**

当前不是简单的 FPS、action stride 或全轨迹固定 phase 偏移。模型产生的是 demonstration 中找不到的长距离轨迹；但 5.20–5.21 后续证明 startup ready-pose 的 label phase aliasing 会显著放大闭环误差。

### 5.17 完整 episode 与短诊断

**排查方式**

- 完整 episode：1800 control steps，开启三路 camera video。
- 快速诊断：300 steps，开启 inference NPZ trace 和 bowl state trace。

**已有完整 episode**

以下 run 均 `success: false`，并且发生在最新 translation anchor 之前，因此作为失败基线保存，不能代表最新代码已经通过或失败：

1. `results/agibot_gr00t_n17_diffik_chunk16_median5_20260906/2026-09-06_07-38-14`
2. `results/agibot_gr00t_n17_final_seeded_denoise8_20260906/2026-09-06_08-47-06`

第二个 run 的 head video：

```text
results/agibot_gr00t_n17_final_seeded_denoise8_20260906/2026-09-06_08-47-06/
  agibot_gr00t_n17_diffik_chunk16_median5_final/
  robot-cam-rebuild0-env0-head_cam_rgb-episode-0.mp4
```

同目录还包含 left/right wrist MP4。

**最新短诊断**

- chunk-16 + anchor：`/tmp/agibot_gr00t_n17_joint7_clip_anchor_chunk16_300`
- corrected chunk-8 + anchor：`/tmp/agibot_gr00t_n17_joint7_clip_anchor_chunk8_300_corrected`
- state-delay paired probe：`/tmp/agibot_gr00t_n17_delay_pair_probe`

这些 run 因 `num_steps` 截断，没有完成 episode，因此没有新 episode MP4。

**当前结论**

尚未在“当前全部 Arena 修改 + 重启后的当前 GR00T server”组合上跑过完整 episode。不得宣称最终 eval 已修好。

### 5.18 Camera factorial：同 state 下逐路替换三路图像

**排查点**

此前的 live/dataset 对照仍混有轻微 state drift。这里要在几乎逐数值相同的 robot/object state 下回答：

1. 图像差异本身能否改变 action；
2. 哪一路 camera 的主效应最大；
3. “完全 exact state 的重新渲染”与“已经 drift 的 live rollout”是否有不同影响。

**排查方式**

- Dataset：episode 78，row 26。
- Exact live trace：`/tmp/agibot_episode78_live_oracle_model_probe_joint_20260906/traces/inference_0001.npz`。
- Drift live trace：`/tmp/agibot_gr00t_n17_joint7_clip_anchor_chunk16_300/traces/inference_0001.npz`。
- Exact control 的 14 个 arm joints 相对 dataset row 26 的 RMS 为 `1.01e-7 rad`；左右 EEF XYZ error 分别为 `0.00987 / 0.00986 mm`。
- scene 中三只碗的 root pose 与 dataset/oracle setup 一致。
- 固定 dataset row-26 state、target、seed 10、denoise 8；每个输入连续采样 5 次取 elementwise median，再做 translation anchor alpha 1。
- 对三路 camera 做 `2^3` factorial。`E/L/R=1` 表示用 exact 或 drift live camera 替换对应 dataset camera。
- 脚本：`isaaclab_arena_gr00t/scripts/probe_agibot_visual_contract.py --suite camera_matrix`。

**证据**

Anchored 左臂平均位置误差如下：

| 图像来源 | Mask | prefix-8 | full-40 |
|---|---:|---:|---:|
| dataset | E0 L0 R0 | 2.0 mm | 43.3 mm |
| exact | E0 L0 R1 | 3.6 mm | 104.3 mm |
| exact | E0 L1 R0 | 2.2 mm | 70.4 mm |
| exact | E0 L1 R1 | 2.4 mm | 103.9 mm |
| exact | E1 L0 R0 | 1.9 mm | 37.2 mm |
| exact | E1 L0 R1 | 3.7 mm | 119.0 mm |
| exact | E1 L1 R0 | 1.7 mm | 59.9 mm |
| exact | E1 L1 R1 | 3.6 mm | 140.2 mm |
| drift | E0 L0 R1 | 4.3 mm | 138.0 mm |
| drift | E0 L1 R0 | 2.4 mm | 27.9 mm |
| drift | E0 L1 R1 | 14.5 mm | 239.5 mm |
| drift | E1 L0 R0 | 1.8 mm | 38.0 mm |
| drift | E1 L0 R1 | 4.0 mm | 161.7 mm |
| drift | E1 L1 R0 | 2.4 mm | 24.0 mm |
| drift | E1 L1 R1 | 10.6 mm | 177.4 mm |

Exact cube 对 full-40 的平均主效应：

- ego：`+8.6 mm`；
- left wrist：`+17.6 mm`；
- right wrist：`+64.1 mm`。

Drift cube 中 right wrist 主效应增至 `+145.9 mm`，并与 left wrist 存在强非线性交互。全 dataset、全 exact、全 drift 的左臂误差随 horizon 为：

| 输入 | step 1 | prefix-8 | prefix-16 | full-40 |
|---|---:|---:|---:|---:|
| dataset | 0.2 mm | 2.0 mm | 7.9 mm | 43.3 mm |
| joint-exact live | 0.2 mm | 3.6 mm | 14.7 mm | 140.2 mm |
| drift live | 0.2 mm | 10.6 mm | 43.0 mm | 177.4 mm |

Exact live 与 dataset decoded frame 的 processor-spatial 图像指标：

| View | MAE（uint8） | PSNR | SSIM |
|---|---:|---:|---:|
| ego | 3.6985 | 31.58 dB | 0.8827 |
| left wrist | 2.1070 | 36.68 dB | 0.9566 |
| right wrist | 2.6556 | 30.98 dB | 0.9495 |

同一 ready interval 的 dataset row 9 vs 26 本身也有 RTX/编码变化：MAE 为 `1.89 / 1.20 / 1.59`；两次独立 live exact render 的 MAE 约 `0.84–0.90`。

**结论：已证实**

- 即使 state 达到 `1e-7 rad` / `0.01 mm` 级一致，重新渲染的像素仍足以将 full-40 action error 从 `43.3 mm` 放大到 `140.2 mm`。
- right wrist 是该状态下最大的单路主效应，但三路存在交互，不能把其余两路视为无关。
- prefix-8 仍较好而 long horizon 先恶化，解释了为何 rollout 开始看起来正常、随后迅速离开分布。

**证据与产物**

- `diagnostics/agibot_visual_contract_ep78_row26_joint_exact_20260906/results.json`
- 同目录的三路 comparison montage 和 `actions.npz`
- 早先 EEF-exact 对照保留在 `diagnostics/agibot_visual_contract_ep78_row26_20260906/`，但 joint-exact 目录是本节权威结果。

### 5.19 实际 processor 与图像预处理排查

**排查点**

Action 差异是否来自 Arena 使用了错误的 view order、RGB/BGR、resize/crop、state packing，或 dataset MP4 的 YUV/H.264 流程没有在 live 输入上复现。

**排查方式**

- 直接加载 checkpoint 对应的 `Gr00tN1d7Processor`，而不是仅凭 README 推断。
- 对 dataset row 26 与 joint-exact live observation 比较 processor 的 `input_ids`、attention mask、归一化 state、`pixel_values` 和 `image_grid_thw`。
- 将 exact live RGB 分别经过 YUV420、H.264、逐通道 histogram matching、ECC 亚像素配准和 ECC+histogram，再用完全相同的 server seed/采样设置请求 action。
- 无 dataset reference 的变换与使用 dataset reference 的诊断变换分开解释；后者不能作为生产方案。

**实际 processor contract**

图像变换顺序为：

```text
LetterBoxPad
-> SmallestMaxSize(256, INTER_AREA)
-> FractionalCenterCrop(0.95)
-> SmallestMaxSize(256)
```

- view order 精确为 `ego_view, left_wrist_view, right_wrist_view`；
- 每路 `image_grid_thw = [1, 16, 16]`；
- 拼接后 `pixel_values.shape = (768, 1536)`；
- `input_ids`、attention mask、归一化 state 完全 bit-equal；state tensor 为 `(1, 1, 132)`。

Dataset vs joint-exact 的 processor pixel 指标：

| 范围 | MAE | RMSE | Cosine |
|---|---:|---:|---:|
| 全部 | 0.02212 | 0.04771 | — |
| ego | 0.02901 | 0.05270 | 0.993835 |
| left wrist | 0.01653 | 0.02931 | 0.998681 |
| right wrist | 0.02083 | 0.05650 | 0.994581 |

预处理 probe 的 anchored 左臂误差：

| 输入 | prefix-8 | full-40 |
|---|---:|---:|
| dataset all | 2.0 mm | 43.3 mm |
| exact raw all | 3.6 mm | 140.2 mm |
| exact raw right-only | 3.6 mm | 104.3 mm |
| exact YUV420 all | 4.0 mm | 145.6 mm |
| exact YUV420 right-only | 3.5 mm | 97.7 mm |
| exact H.264 all | 2.4 mm | 94.3 mm |
| exact H.264 right-only | 3.5 mm | 103.1 mm |
| exact histogram all | 3.0 mm | 125.6 mm |
| exact histogram right-only | 3.3 mm | 97.4 mm |
| exact ECC all | 2.0 mm | 68.7 mm |
| exact ECC right-only | 2.4 mm | 79.1 mm |
| exact ECC+hist all | 2.0 mm | 88.7 mm |
| exact ECC+hist right-only | 2.0 mm | 69.1 mm |

ECC 找到的 raw-512 刚体配准量很小：

- ego：`dx=-0.198 px, dy=+0.265 px, angle=-0.033°`；
- left wrist：`dx=-0.029 px, dy=+0.059 px, angle=-0.030°`；
- right wrist：`dx=-0.243 px, dy=-0.155 px, angle=+0.032°`。

**结论：部分成立**

- 已排除 camera key/order、RGB/BGR、dtype/range、processor crop/resize 和 state packing bug。
- 单纯 YUV420 不改善；H.264 有帮助但仍离 dataset baseline 很远；histogram 不是根因。
- ECC/resampling 改善最大，说明亚像素几何或采样差异很重要，但 ECC 本身包含插值/平滑效应，不能据此直接断言 extrinsics 错误。
- ECC 和 histogram 都使用 dataset reference，只能用于定位，不能在真实 eval 中启用。
- 当前 repo 与 `data_process` checkout 的相机 intrinsics、mount、look-at/update 代码一致；离线 rerender 数学也一致。但这还不能证明**生成数据时实际运行的 commit/container/render settings**完全一致。

**证据与产物**

- `diagnostics/agibot_visual_preprocessing_ep78_row26_20260906/results.json`
- 同目录的 H.264 round-trip MP4 和 action arrays。

### 5.20 近静止 startup 区间的 observation/label phase aliasing

**排查点**

Ready pose 前后许多 observation 几乎相同，但 40-step label 中“何时开始移动”可能不同。若 checkpoint 无法从单帧辨认时间，相同画面会对应互相冲突的 action onset。

**排查方式**

- 固定使用 episode 78 的 dataset camera 和 dataset state，不引入 live renderer 或控制误差。
- Probe rows `9, 10, 14, 18, 22, 26`；以 row 26 为 state/image reference。
- 每个 row 使用 seed 10、denoise 8、5-sample median、translation anchor。
- 以左 EEF 相对 action 首帧超过 5 mm 的第一个 horizon index 定义 action onset。
- 脚本：`isaaclab_arena_gr00t/scripts/probe_agibot_stationary_phase.py`。

**证据**

| Dataset row | State RMS vs row 26 | 左臂 prefix-8 | prefix-16 | full-40 | Pred onset | Target onset |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | `3.83e-5` | 2.6 mm | 4.3 mm | 15.6 mm | 10 | 26 |
| 10 | `4.75e-5` | 2.7 mm | 4.5 mm | 14.8 mm | 10 | 25 |
| 14 | `2.78e-7` | 2.6 mm | 4.5 mm | 26.4 mm | 6 | 21 |
| 18 | `4.51e-7` | 2.5 mm | 3.8 mm | 47.5 mm | 10 | 17 |
| 22 | `3.67e-7` | 1.6 mm | 10.6 mm | 72.1 mm | 10 | 13 |
| 26 | `0` | 2.0 mm | 7.9 mm | 43.3 mm | 9 | 9 |

训练 label 的 onset 在近乎相同 observation 间移动了 17 steps，而模型通常选择 horizon 6–10 开始移动。

**结论：已证实**

- 这是 observation/label phase aliasing：单帧 checkpoint 没有足够信息区分 ready pose 已保持了多久。
- 当前 warmup 10 的第一次 query 对应 row 10；target 到 step 25 才移动，但模型 step 10 左右便移动。chunk length 16 因而会执行一段过早动作。
- 这并不推翻 5.16：全轨迹 nearest-phase search 仍表明错误 action 不是一个可由固定时间偏移解释的正确轨迹；startup phase 只是进入闭环发散的重要放大器。

**证据与产物**

- `diagnostics/agibot_stationary_phase_ep78_20260906/results.json`

### 5.21 Warmup 10 vs 26 短闭环 A/B

**排查点**

将首次 query 从冲突最大的 row 10 推迟到 label onset 已自洽的 row 26，是否能在真实 live-camera 闭环中减缓发散；以及残余误差来自 server action 还是 controller tracking。

**排查方式**

- 两组使用相同 environment seed 42、scene、ready pose、live cameras、port 5558 server、server seed 10、denoise 8、median 5、chunk 16、state delay 1、translation anchor 1。
- 唯一变量：`ISAACLAB_ARENA_GR00T_WARMUP_STEPS=10` 或 `26`。
- 每组运行 110 control steps；query stride 为 16。
- warmup 10 的 nominal dataset rows 为 `10, 26, 42, 58, 74, 90, 106`；warmup 26 为 `26, 42, 58, 74, 90, 106`。
- 下一 query 使用 1-step delayed state，因此其位置应与上一 action 的 index 14 对齐。将误差分解为：model command 到 dataset target 的误差，以及实际 EEF 到 model command 的 controller tracking 误差。
- 可复现配置：`diagnostics/agibot_warmup26_ab_20260906/experiment_110_steps.yaml`。
- 分析脚本：`isaaclab_arena_gr00t/scripts/analyze_agibot_warmup_ab.py`。

**共同 dataset phase 的状态误差**

| Dataset row | Warmup 10 左 / 右 | Warmup 26 左 / 右 |
|---:|---:|---:|
| 26 | 12.7 / 3.7 mm | 0.01 / 0.01 mm |
| 42 | 109.6 / 10.5 mm | 27.5 / 5.3 mm |
| 58 | 557.1 / 15.5 mm | 146.2 / 14.3 mm |
| 74 | 548.3 / 13.2 mm | 141.5 / 21.5 mm |
| 90 | 445.2 / 23.0 mm | 152.9 / 22.8 mm |
| 106 | 457.2 / 27.1 mm | 140.2 / 26.2 mm |

排除共同 phase 的第一个初始点后，rows 42–106：

- 左臂均值从 `423.5 mm` 降至 `121.7 mm`，下降 `71.3%`；
- 右臂均值 `17.87 -> 17.99 mm`，基本不变；任务主运动臂是左臂。

**模型/控制器误差分解**

| 组别与 transition | 左 model command error | 左 controller tracking error | 下一 state 总误差 |
|---|---:|---:|---:|
| warmup 10，row 10 -> 26 | 12.4 mm | 0.3 mm | 12.7 mm |
| warmup 10，row 26 -> 42 | 108.4 mm | 2.2 mm | 109.6 mm |
| warmup 26，row 26 -> 42 | 27.2 mm | 1.2 mm | 27.5 mm |
| warmup 26，row 42 -> 58 | 169.2 mm | 24.8 mm | 146.2 mm |

Warmup 26 首次 query 的左 action error 为 `2.1 / 10.5 / 125.6 mm`（prefix 8 / prefix 16 / full 40），predicted/target onset 为 `5 / 9`。

两组的初始 live-vs-dataset processor-spatial MAE 几乎相同：

| View | Warmup 10 | Warmup 26 |
|---|---:|---:|
| ego | 3.791 | 3.787 |
| left wrist | 2.268 | 2.123 |
| right wrist | 3.024 | 2.673 |

因此 improvement 来自 query phase，而不是多等 16 steps 让 renderer 明显“热起来”。

**结论：部分成立**

- Warmup 26 是强而明确的 mitigation，使同 phase 左臂误差平均下降 71%。
- 它没有解决 live-camera 闭环：第二个 query 后左臂仍达到 `146 mm` 误差。
- 发散初期 controller 精确跟随错误 command，说明首因仍是模型输入/预测；偏差很大后 controller 饱和才成为次生误差。
- 110-step run 中三只碗最大位移小于 `0.00007 mm`，尚未进入抓取阶段；该 A/B 只评价 approach trajectory。
- 因此暂不把 26 写入正式默认配置，先保留为下一次 300-step/full-episode 的候选变量。

**证据与产物**

- `diagnostics/agibot_warmup26_ab_20260906/results.json`
- `diagnostics/agibot_warmup26_ab_20260906/warmup10_traces/`
- `diagnostics/agibot_warmup26_ab_20260906/warmup26_traces/`
- 两次 run 均因 `num_steps=110` 结束，没有 episode MP4。

### 5.22 Dataset-camera closed-loop 因果实验

**排查点**

在 phase 已改为 row 26 后，live 与 dataset 之间的小像素差是否真的足以造成剩余闭环发散，还是只有离线单 query 指标敏感而真实执行无影响。

**排查方式**

- 固定 warmup 26 和 5.21 的全部 state/control/server 参数。
- State、场景物体和执行器始终来自 live simulator；只在发给 GR00T 前替换 video tensor。
- 三组：全部 live、仅 right wrist 使用 episode-78 对应 dataset frame、三路都使用 dataset frame。
- 每个 control step 按 row 读取 MP4，故 inference queries 对应 rows `26, 42, 58, 74, 90, 106`。
- 事后逐 trace 解码核对：dataset-all 的三路输入在全部六个 query 上与预期 MP4 row 的 MAE 都严格为 `0`；right-only 组只有 right wrist 严格为 `0`，其余两路仍为 live。
- Dataset 替换是 oracle diagnostic，只用于因果定位，不是可部署方案。
- Diagnostic policy：`isaaclab_arena_gr00t/policy/agibot_dataset_video_closedloop_probe_policy.py`。

**首次 query：state 完全相同**

三组 q0 的 left/right EEF 均只与 dataset 相差约 `0.01 mm`，arm-joint RMS 为 `1.26e-7 rad`。

| Camera input | 左 prefix-8 | 左 prefix-16 | 左 full-40 | Pred/target onset | 下一 row-42 左 state error |
|---|---:|---:|---:|---:|---:|
| 全 live | 2.1 mm | 10.5 mm | 125.6 mm | 5 / 9 | 27.5 mm |
| 仅 right wrist 为 dataset | 1.8 mm | 12.2 mm | 53.9 mm | 8 / 9 | 28.6 mm |
| 三路 dataset | 2.0 mm | 7.9 mm | 43.3 mm | 9 / 9 | 19.1 mm |

Right-wrist-only 会显著修正 long-horizon plan 和 onset，但没有改善第一个实际执行 prefix 的终点；这说明 camera 之间存在 horizon-dependent interaction，不能仅凭 full-40 指标预测第一 chunk。

**110-step 累积闭环结果**

| Dataset row | 全 live 左误差 | Dataset right-only | Dataset all |
|---:|---:|---:|---:|
| 42 | 27.5 mm | 28.6 mm | 19.1 mm |
| 58 | 146.2 mm | 67.8 mm | 51.9 mm |
| 74 | 141.5 mm | 50.0 mm | 13.1 mm |
| 90 | 152.9 mm | 25.6 mm | 17.3 mm |
| 106 | 140.2 mm | 125.9 mm | 86.2 mm |
| rows 42–106 mean | **121.7 mm** | **59.6 mm** | **37.5 mm** |

- Right-wrist-only 相对全 live 降低 `51.0%`；
- 三路 dataset 相对全 live 降低 `69.2%`；
- warmup 26 + 三路 dataset 相对旧 warmup-10 live 基线的共同 phase 均值合计降低约 `91.1%`。

三路 dataset 组的前几个 transition 中，左 controller tracking error 通常只有 `0.8–6.4 mm`，而 model command error 为 `19.8–49.2 mm`；后续仍是模型命令误差占主导。三组碗的最大位移同样低于 `0.00007 mm`。

**结论：已证实**

- 视觉输入差异在真实闭环中具有因果作用，不是离线 metric artifact。
- Right wrist 是最高优先级 camera，但只修这一视角不足以完全恢复；三路共同对齐效果最好。
- 字段/processor contract 已通过，而差异集中在像素生成域。现有证据仍不能区分：
  1. 生成训练视频与 eval 的 runtime renderer/camera settings 存在可修复 mismatch；
  2. 两者都是合法渲染，但 checkpoint 对很小像素变化严重过拟合。
- Dataset frames 与真实偏离后的 scene 并不完全一致，所以本实验不能当成任务成功率，只能证明 camera causality。

**实验有效性记录**

- 第一次启动因 diagnostic class 间接继承 registered policy，被 loader 拒绝：`must directly inherit PolicyBase[ConcretePolicyCfg]`；未进入 rollout，标记为**无效实验**。改为直接继承 `PolicyBase` 并组合 production policy。
- 第二次启动因 wrapper 初始化顺序错误，访问尚不存在的 `self.modality_configs`；未进入 rollout，标记为**无效实验**。调整 inner policy 初始化顺序后重跑。
- 随后的 dataset-all 和 dataset-right-only run 均完成 110 steps，无 exception，trace 数量各 6。

**证据与产物**

- `diagnostics/agibot_warmup26_ab_20260906/dataset_video_results.json`
- `diagnostics/agibot_warmup26_ab_20260906/dataset_right_wrist_results.json`
- `diagnostics/agibot_warmup26_ab_20260906/dataset_video_traces/`
- `diagnostics/agibot_warmup26_ab_20260906/dataset_right_wrist_traces/`
- Dataset-all report：`/eval/agibot_eval_output_dataset_video/2026-09-06_13-24-09/`（容器内）。
- Right-only report：`/eval/agibot_eval_output_dataset_right_wrist/2026-09-06_13-26-51/`（容器内）。
- 两者均为 step-limited run，没有 episode MP4。

### 5.23 EEF 坐标错误的 Git 历史归因

**排查点**

用 `git log` / `git blame` 判断训练 raw EEF frame 被直接当作 AgiBot control frame 的错误由哪个 commit 引入。

**排查方式**

- 对 AgiBot embodiment、GR00T N1.7 policy、AgiBot action adapter 和 DiffIK offset-Jacobian 分别运行 `git log -S`、`git blame` 和历史 diff。
- 对比 `HEAD=d819b42` 与当前工作树，并检查独立的数据处理 checkout。

**证据**

- `HEAD` 中没有 `AGIBOT_BIMANUAL_MANIPULATION`、AgiBot–GR00T action adapter 或 AgiBot N1.7 experiment config；这些内容均为当前未提交修改/未跟踪文件。因此“将 checkpoint raw EEF 直接送给带左臂 control-frame offset 的控制器”没有对应 commit hash，无法通过 Git 归责到某一次已提交变更。
- `e89bd1b`（`Add support for GR00T N1.7.`）只增加 DROID N1.7 支持，没有 AgiBot 代码，不是本问题的引入提交。
- AgiBot 左臂 offset 最早随 `40cd404`（`adds new embodiment of Agibot A2D (#292)`）出现；`1ca44bf`（当前分支对应 cherry-pick 为 `5959500`）将其明确写为 xyzw 并加入 dual-arm。该 offset 是为抵消左右 tool USD frame 差异而设计的正确 control-frame 定义，本身不是 bug。
- 数据 enrichment 工具 `tools/enrich_agibot_eef9d.py` 在独立 checkout 中也是未跟踪文件，没有可 blame 的 commit。最终数据的 `meta/eef_9d.json` 记录其旋转于 2026-08-31 被修正为 raw USD frame。
- DiffIK 对 rotational body offset 错误地旋转 angular Jacobian rows 的上游代码，可精确追到 Isaac Lab commit `cf7a65f35df23e8d4e4a02265ec747a96dd951aa`（2023-12-20，`Fixes the inverse kinematics example for Franka robot (#319)`）。该缺陷一直存在于当前 pinned Isaac Lab；但它只在本地未提交的 AgiBot `gr00t_diffik` 模式开始使用 rotational body offset 后影响本任务。

**结论：部分可归因**

- 主要的 train-frame/control-frame adapter 遗漏是在 `d819b42` 之后的未提交 AgiBot–GR00T 集成过程中引入，Git 历史不能进一步定位到某个 commit。
- 可独立归因的 offset-Jacobian 子 bug 来自上游 Isaac Lab commit `cf7a65f35df23e8d4e4a02265ec747a96dd951aa`。
- 不应把 `e89bd1b`、`40cd404` 或 `5959500` 标记为主要坐标 bug 的引入提交。

## 6. 已保留、已否定和仅用于诊断的改动

### 6.1 当前保留

- AgiBot bimanual task mode 和 20D action adapter。
- 双向 training/control EEF frame conversion。
- control-rate DiffIK。
- 正确的 translated-frame Jacobian。
- 逐关节 delta clamp。
- 只约束 joint 7 的 Moore–Penrose nullspace bias。
- gripper negative preload。
- policy-ready reset pose 和当前正式基线的 10-step camera warmup；26-step 候选尚待正式复测。
- 1-step state delay。
- server reset seed。
- request-level denoising steps，当前为 8。
- 5-sample elementwise median。
- relative EEF percentile clipping 能力。
- action chunk translation anchor，alpha=1。
- inference diagnostics：原始/median/anchored action、delayed/current state、camera 和 sim action trace。

### 6.2 已否定，不应重新启用为默认值

- 直接执行全部 40-step action horizon。
- chunk 8 作为解决方案。
- RTC frozen-tail 作为当前默认策略。
- 对全部 7 个 arm joints 做 nullspace posture bias。
- 对整个 joint delta vector 做统一比例缩放。
- 认为只要做 percentile clipping 就能消除闭环 drift。
- 认为 state delay=0 或 FPS mismatch 是当前主因。

### 6.3 仅用于诊断

- `isaaclab_arena_gr00t/policy/agibot_oracle_probe_policy.py`
- `isaaclab_arena_gr00t/policy/agibot_dataset_video_closedloop_probe_policy.py`
- `isaaclab_arena_gr00t/scripts/probe_agibot_visual_contract.py`
- `isaaclab_arena_gr00t/scripts/probe_agibot_stationary_phase.py`
- `isaaclab_arena_gr00t/scripts/analyze_agibot_warmup_ab.py`
- `diagnostics/agibot_warmup26_ab_20260906/` 下的 warmup 和 dataset-camera A/B configs/traces/results。
- RTC frozen-8 diagnostic config。
- `/tmp` 下的 teacher-forcing、camera ablation、oracle replay 和短 rollout YAML/scripts。
- RMPFlow 的 low-pass / alternate config；正式配置当前使用 DiffIK。

## 7. 回归验证

最终相关单元测试命令：

```bash
cd /home/ubuntu/projects/isaaclab_arena_proj/IsaacLab-Arena
ACCEPT_EULA=Y \
OMNI_KIT_ACCEPT_EULA=YES \
MPLCONFIGDIR=/tmp/matplotlib \
PYTHONPATH=/home/ubuntu/projects/isaaclab_arena_proj/IsaacLab-Arena/submodules/Isaac-GR00T \
.venv/bin/python -m pytest -q \
  isaaclab_arena_gr00t/tests/test_gr00t_remote_closedloop_policy.py \
  isaaclab_arena_gr00t/tests/test_gr00t_n17_droid_contract.py \
  isaaclab_arena_gr00t/tests/test_gr00t_n17_agibot_contract.py \
  isaaclab_arena_gr00t/tests/test_state_history.py
```

结果：

```text
45 passed, 100 warnings
```

warnings 为 Isaac Lab/Kit API deprecation，未发现本次逻辑失败。`git diff --check` 通过。

测试中旧 `_FakePolicyClient` / `_FakeClient` 最初不接受真实 N1.7 client 的 `options` 参数，并返回 `info=None`。这导致过期 test doubles 报错，而非生产代码回归；已更新 fake signature 为 `get_action(observation, options=...)`、`reset(options=...)`，并回传实际 denoise metadata。

本轮新增的三个分析脚本和 dataset-camera diagnostic policy 均通过 `py_compile` 与 `git diff --check`。Typed experiment loader 和实际 Isaac Sim runtime 已通过两次成功的 110-step dataset-camera rollout 验证。由于新增文件不改变 production policy 的默认路径，没有重复运行上面的 45-test suite；下次修改 production camera/runtime 后必须重跑。

## 8. 下一阶段计划：恢复生成时 runtime provenance

### 8.1 P0：Right-wrist generation/runtime 对照

字段、processor 和当前源码配置已经核对。现在最关键的未知量不是“代码现在看起来是否相同”，而是训练 MP4 **当时实际由什么 commit、container image 和 renderer settings 生成**。

按以下顺序进行：

1. 从 dataset metadata、生成日志或 shell history 恢复生成 episode 78 的 git SHA、Docker image、Isaac Sim/Isaac Lab version 和完整启动命令。
2. 在生成环境中重放 row 26，保存 H.264 编码前的 raw RGB、编码后的 decoded RGB，以及 camera world transform/intrinsics。
3. 在当前 eval 环境保存相同 state 的对应值，逐项比较：parent prim、world pose、focal length、horizontal aperture、clipping、render interval、DLSS/AA、曝光/色彩空间、灯光、Fabric/geometry streaming。
4. 对 wrist pose update 顺序做最小 A/B：

```text
scene.reset_to -> aim_wrist_cams -> render
scene.reset_to -> sim.forward -> aim_wrist_cams -> render
```

5. 先只评估 right wrist，再验证三路组合。每个候选变换都必须同时报告 processor-pixel distance、固定-seed action prefix 8/16/40，以及 110-step closed-loop state error。

验收目标：找到一个**不读取 dataset reference frame**、可由 camera/renderer 配置复现的变换，使 row-26 live 输入的：

- right-wrist-only 左 full-40 error 接近约 `54 mm`；
- 三路对齐后接近 dataset baseline `43 mm`；
- 110-step rows 42–106 左误差均值从 live 的 `121.7 mm` 接近 dataset-camera 的 `37.5 mm`。

如果只能通过 dataset-guided ECC/histogram 达到目标，则不能算 pipeline fix。

### 8.2 判定分叉：Pipeline bug 或 checkpoint robustness

如果 P0 找到 generation/eval mismatch：

- 在正式 camera/runtime 中修复该 mismatch；
- 重做 exact-state factorial 和 110-step live rollout；
- 只有不依赖 dataset oracle 的指标复现改善，才保留修复。

如果 generation runtime 完全复现后仍存在当前敏感性，则将根因定性为 checkpoint/data robustness：

- 从当前 policy rollout 收集 near-distribution recovery trajectories。
- DAgger 或人工/teacher relabel。
- fine-tune 时加强亚像素几何、颜色、压缩、renderer noise 和轻微时序扰动。
- 处理 startup 的 observation/label aliasing：裁剪冲突的长 idle prefix，或加入历史/phase 可观测量，而不是让同一单帧对应相差 17 steps 的 onset。
- 将 validation 从随机 held-out frame 改为 closed-loop perturbation suite。
- 报告 prefix-8/16、full-40、hand phase timing，而不是只看全维平均 MAE。

temporal ensemble、trust region、anchor 和 clipping 可以作为安全 mitigation，但不能替代 recovery data。

### 8.3 可并行保留的 mitigation

- Warmup 26 已有明确短闭环收益，应作为下一次 live 300-step A/B 候选，但在完整 episode 通过前不宣称修复。
- Dataset-camera substitution、ECC 和 histogram matching 只能用于诊断，禁止进入 production eval。
- H.264 round-trip 在单 query 中把 all-live full-40 从 `140.2` 改善到 `94.3 mm`，但远未恢复且可能增加延迟；只有 streaming closed-loop A/B 通过后才考虑。
- 当前 chunk 16、state delay 1、median 5、denoise 8 和 translation anchor 1 保持不变，避免再次混淆变量。

### 8.4 下一次完整 eval 的前置条件

1. 重启 5557 server，确认启动日志启用了需要的 server patch。
2. 用同一 serialized observation 做 reset-reproducibility smoke test。
3. client log 必须显示 server acknowledge `num_inference_timesteps=8`。
4. 将 warmup 10/26 或 camera runtime fix 作为单一变量，先跑 300-step live trace。
5. 300-step 中需报告 model-command/controller 分解和 bowl displacement；只有 approach 不再发散才进入完整 eval。
6. 再跑完整 1800-step episode并开启 `--record_camera_video`。
7. 将完整 output 从 `/eval/agibot_eval_output` 保存到新的、带日期的 `results/` 目录，并在本文追加结果。

### 2026-09-06 13:49--14:04 UTC — 当前修改代码的 warmup-26 完整录像 eval

**排查点**

验证当前工作树在 live 三相机输入和真实远程 checkpoint 下是否能完成无异常的 1800-step
rollout，并直接观察 warmup 26 改善能否转化为 stack-bowls 成功。

**排查方式**

- Checkpoint：`checkpoint-10000`；seed 42；1 env；1 episode；15 Hz；上限 1800 steps/120 s。
- Client 参数：chunk 16、median of 5、denoise 8、translation anchor 1、state delay 1、RTC off。
- 单一 runtime override：`ISAACLAB_ARENA_GR00T_WARMUP_STEPS=26`；正式 YAML 的默认 warmup 10 未改写。
- 输入：当前 Isaac Sim 的 head、left-wrist、right-wrist 三路 512x512 live RGB；未使用 dataset-camera substitution。
- 记录三路 camera video，并保存前 10 次 replan 的完整 inference NPZ trace。
- 首次连接原 PID 2462503 的 5557 server 时，client 在第一次请求后发现 server 未 acknowledge
  `num_inference_timesteps=8`，因此契约检查按预期终止。确认该进程从 00:39 UTC 起一直运行，尚未加载
  当前 `gr00t_policy.py` 补丁。
- 停止旧进程后，以完全相同 checkpoint/embodiment/device/host/port 和当前工作树重启 server；新 PID
  4180030 显示 `Server ready`。随后重新运行同一 eval，denoise 契约通过。

实际成功运行的命令为：

```bash
docker exec --workdir /workspaces/isaaclab_arena \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ARENA_LOCAL_ASSET_DIR=/tmp/isaaclab_arena_agibot_assets/arena_local \
  -e HF_HOME=/tmp/isaaclab_arena_agibot_assets/hf \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e ISAACLAB_ARENA_GR00T_WARMUP_STEPS=26 \
  -e ISAACLAB_ARENA_GR00T_TRACE_DIR=/workspaces/isaaclab_arena/results/agibot_gr00t_n17_current_warmup26_patched_server_eval_20260906_1354/traces \
  -e ISAACLAB_ARENA_GR00T_TRACE_LIMIT=10 \
  isaaclab_arena-latest bash -lc \
  './submodules/IsaacLab/_isaac_sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
    --experiment_config diagnostics/eval_configs/agibot_stack_bowls_gr00t_n17_current_warmup26_one_episode.yaml \
    --headless --record_camera_video \
    --output_base_dir /workspaces/isaaclab_arena/results/agibot_gr00t_n17_current_warmup26_patched_server_eval_20260906_1354'
```

**证据与产物**

- Run status：`completed`；episode length 1800；termination 为 timeout；success false；success rate 0/1。
- 三路 MP4 均经 `ffprobe` 验证为 H.264、512x512、15 fps、1799 frames、119.934 s。
- Video：`results/agibot_gr00t_n17_current_warmup26_patched_server_eval_20260906_1354/2026-09-06_13-54-35/agibot_stack_bowls_gr00t_n17_current_warmup26/`。
- Trace：`results/agibot_gr00t_n17_current_warmup26_patched_server_eval_20260906_1354/traces/inference_0000.npz` 至 `inference_0009.npz`。
- Metrics/report：同一 timestamp 目录内的 `arena_experiment_result.json`、`episode_results_rebuild0.jsonl` 和 `index.html`。
- Head-camera 每 15 s contact sheet：`diagnostics/agibot_gr00t_n17_current_warmup26_eval_20260906_1354/head_every_15s_contact_sheet.jpg`。
- 粗略视觉检查：双臂有持续大范围移动，左臂多次接近碗区，但 15 s 间隔关键帧中三个碗始终分离，未观察到可靠抓取或堆叠。

**结论**

当前 client/server 代码和资产 pipeline 可以无异常跑完并产出有效录像；此前 denoise 报错是未重启 server
造成的进程版本不一致，不是新 client 本身失效。但 warmup 26 的短闭环误差改善没有在本次固定 seed
完整 episode 中转化为任务成功，故不能把 startup 修正视为充分修复。完整 live-camera 结果仍支持继续优先
排查 generation/eval camera-runtime domain gap，尤其 right wrist，而非继续微调执行平滑参数。

**影响/下一步**

- 保留 warmup 26 作为已验证的实验候选，但正式 YAML 暂不从 10 改为 26。
- 以本次 10 个 trace 和三路录像作为新的 production baseline；下一步对首次 approach/grasp 窗口做
  model-command、controller-following、camera pixel 和 bowl displacement 对齐分析。
- 5557 当前已运行补丁后的 server；后续实验应先确认 PID/启动时间和 denoise acknowledgement，避免再次混用旧进程。

### 2026-09-06 14:45--14:54 UTC — denoise-4 对照完整录像 eval

**排查点**

在相同 live-camera、相同 seed 和当前补丁 server 下，将 request-level denoising steps 从 8 改回
checkpoint 的默认 4，判断完整闭环行为是否改善。

**排查方式**

- 新建 `agibot_n17_gr00t_closedloop_denoise4_config.yaml`；其全部运行时数值与 denoise-8 配置一致，
  唯一变量为 `denoising_steps: 4`。
- 相同 checkpoint、seed 42、ready pose、三路 512x512 live camera、warmup 26、state delay 1、
  chunk 16、median 5、translation anchor 1、RTC off、1 episode、1800 steps。
- client 启动日志显示 `denoise=4`；补丁 server 正确 acknowledge，因此不是旧 server 静默忽略配置。

**证据与产物**

- Run status：`completed`；episode length 1800；timeout；success false；success rate 0/1；wall-clock 8:22。
- 三路 MP4 均经 `ffprobe` 验证为 H.264、512x512、15 fps、1799 frames、119.934 s。
- Video：`results/agibot_gr00t_n17_denoise4_warmup26_eval_20260906_1445/2026-09-06_14-45-43/agibot_stack_bowls_gr00t_n17_denoise4_warmup26/`。
- Trace：`results/agibot_gr00t_n17_denoise4_warmup26_eval_20260906_1445/traces/inference_0000.npz` 至 `inference_0009.npz`。
- Head-camera contact sheet：`diagnostics/agibot_gr00t_n17_denoise4_warmup26_eval_20260906_1445/head_every_15s_contact_sheet.jpg`。
- 关键帧与完整视频视觉对照：denoise-4 很快产生明显怪异的大幅双臂摆动、交叉/抬高姿态，且频繁遮挡
  head camera。行为在 **approach 阶段已经退化**：机械臂不能稳定地向碗区接近，随后远离或在非目标区域
  运动；三个碗始终保持分离，未进入可靠抓取或堆叠阶段。
- 相比之下 denoise-8 同 seed 也未成功，但仍可多次接近碗区，动作幅度较小且较少遮挡碗区。因此两者不是
  单纯“都失败”的等价结果：4-step 的 task-phase progress 与视觉可解释性显著更差。

**结论：denoise-4 不优于 denoise-8**

同一 seed 下两组均未完成任务，不能由单 episode 宣称成功率的统计显著性；但对于本条固定-seed 轨迹，
denoise-4 在 approach 前即出现明显且很大的行为退化，不能稳定接近碗，远差于 denoise-8。加上已有
teacher-forcing 证据（prefix-8 左/右位置误差由 8-step 的 6.20/5.38 mm 恶化至 4-step 的 9.12/10.98 mm），
保留 denoise 8 为正式基线；该 A/B 排除“8-step 配置本身造成当前完整 episode 失败”这一简单解释。

### 2026-09-06 15:04--15:16 UTC — denoise-16 对照完整录像 eval

**排查点**

在相同 live-camera、相同 seed 和当前补丁 server 下，把 request-level denoising steps 从 8 增加到 16，
判断更多 flow-matching integration steps 是否能继续改善闭环 approach/grasp 行为。

**排查方式**

- 新建 `agibot_n17_gr00t_closedloop_denoise16_config.yaml`；与 denoise-4/8 对照保持相同 checkpoint、
  seed 42、ready pose、三路 512x512 live camera、warmup 26、state delay 1、chunk 16、median 5、
  translation anchor 1、RTC off、1 episode、1800 steps。
- 唯一行为变量为 `denoising_steps: 16`；client 启动日志显示 `denoise=16`，且补丁 server 对每次请求
  正确 acknowledge。
- 保存三路完整视频和前 10 次 inference trace；从 head video 每 15 秒抽帧，并纵向合成 4/8/16
  三组对照图。

**证据与产物**

- Run status：`completed`；episode length 1800；timeout；success false；success rate 0/1；wall-clock 10:56。
- 三路 MP4 均经 `ffprobe` 验证为 H.264、512x512、15 fps、1799 frames、119.934 s。
- Video：`results/agibot_gr00t_n17_denoise16_warmup26_eval_20260906_1504/2026-09-06_15-04-50/agibot_stack_bowls_gr00t_n17_denoise16_warmup26/`。
- Trace：`results/agibot_gr00t_n17_denoise16_warmup26_eval_20260906_1504/traces/inference_0000.npz` 至 `inference_0009.npz`。
- Head-camera contact sheet：`diagnostics/agibot_gr00t_n17_denoise16_warmup26_eval_20260906_1504/head_every_15s_contact_sheet.jpg`。
- 4/8/16 对照图：`diagnostics/agibot_gr00t_n17_denoise_4_8_16_warmup26_comparison_20260906.jpg`（自上而下为 4、8、16）。
- 关键帧视觉检查：16-step 没有改善 approach；双臂仍出现大幅、非目标导向姿态，并在多个阶段严重
  遮挡 head camera，三个碗始终分离，未进入可靠抓取阶段。8-step 在同 seed 下的动作幅度相对较小，
  且能多次接近碗区。
- 额外计算完整 head-video framewise PSNR：4-vs-16 为 14.79 dB，4-vs-8 为 15.01 dB。两者都很低且
  相近，因此虽然稀疏关键帧中的 4/16 异常姿态观感类似，不能认定二者是相同轨迹。

**结论：denoise-16 也不优于 denoise-8**

固定 seed 的 4、8、16 三组均失败，但 16-step 没有显示更多 integration steps 带来的闭环收益，反而和
4-step 一样在 approach 阶段出现明显异常动作。当前三者中 denoise 8 的定性行为最好，但仍不足以完成任务。
保留 denoise 8 为正式基线；后续应继续排查 live camera/runtime domain gap，而不是继续单独增加 denoise steps。

### 2026-09-09 — denoise 4/8/16/32 × 20 episodes sweep 配置

**排查点**

将此前每组单 episode 的定性比较扩展为 20 个配对 scene resets，统计 denoise steps 对 stack-bowls
成功率和行为稳定性的影响。

**排查方式**

- 新增一个四 Run typed Experiment，每个 Run 1 env、20 episodes、seed 42、1 rebuild。
- 四组固定 warmup 26、server seed 10、chunk 16、median 5、state delay 1、translation anchor 1、RTC off；
  policy YAML 经回归测试确认唯一数值差异为 `denoising_steps = 4/8/16/32`。
- 四组均开启三路 `--record_camera_video`；同一补丁 server 通过 request-level denoise 接口顺序服务所有组。
- 新增 server 和容器内 eval launcher，并对缺失 server/`robodojo_table.usda` 做启动前检查。

**当前状态：启动链路已验证，正式 80-episode sweep 待运行**

- 当前 `isaaclab_arena-latest` container 和 patched 5557 server 均可用。
- 不需要四个 server，也不需要每个 denoise 重启 server；client 会逐请求传递并校验有效 denoise steps。
- `/tmp` 中此前准备的 local asset 曾被清理；已重新从 RoboDojo 官方 Hugging Face dataset 下载原始
  bowl 与 room，并按 2026-09-01 已验证版本重建 table 与 NVIDIA Mahogany material。容器内 OpenUSD
  已确认 table、bowl、room 三个 stage 均可打开。
- 用正式 launcher 启动实际 sweep：首个 denoise-4 run 完成环境、action/observation/camera manager 构建，
  进入 episode，执行完 26-step warmup，并在 warmup 后持续生成三路 MP4；此时 GPU 活动 28%，确认
  server inference 与 rollout 正常。按“跑通即停”要求终止 smoke；eval/ffmpeg 均已停止，5557 server
  仍可连接。首次运行因 RTX `RtPso` cache 编译额外耗时约 240 s，后续启动会复用 cache。另创建本次
  sweep 的专用结果目录并赋予容器 `ubuntu` 用户写权限，避免仓库 `results/` 的 root ownership 阻塞运行。

**配置与脚本**

- `AGIBOT_GR00T_N17_DENOISE_SWEEP_EVAL.md`（完整启动与复现指南）
- `isaaclab_arena_environments/experiment_configs/agibot_stack_bowls_gr00t_n17_denoise_sweep_20ep.yaml`
- `isaaclab_arena_gr00t/policy/config/agibot_n17_gr00t_closedloop_denoise32_config.yaml`
- `isaaclab_arena_gr00t/scripts/serve_agibot_stack_bowls_n17.sh`
- `isaaclab_arena_gr00t/scripts/run_agibot_stack_bowls_denoise_sweep.sh`

## 9. 关键文件索引

- `isaaclab_arena_environments/experiment_configs/agibot_stack_bowls_gr00t_n17_experiment.yaml`
- `isaaclab_arena_gr00t/policy/config/agibot_n17_gr00t_closedloop_config.yaml`
- `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py`
- `isaaclab_arena_gr00t/policy/agibot_dataset_video_closedloop_probe_policy.py`
- `isaaclab_arena_gr00t/policy/gr00t_core.py`
- `isaaclab_arena_gr00t/policy/state_history.py`
- `isaaclab_arena_gr00t/utils/agibot_eef.py`
- `isaaclab_arena/embodiments/agibot/agibot.py`
- `isaaclab_arena/embodiments/common/control_rate_diffik_actions.py`
- `isaaclab_arena_environments/agibot_stack_bowls_environment.py`
- `isaaclab_arena_gr00t/tests/test_gr00t_n17_agibot_contract.py`
- `isaaclab_arena_gr00t/scripts/probe_agibot_visual_contract.py`
- `isaaclab_arena_gr00t/scripts/probe_agibot_stationary_phase.py`
- `isaaclab_arena_gr00t/scripts/analyze_agibot_warmup_ab.py`
- `diagnostics/agibot_visual_contract_ep78_row26_joint_exact_20260906/`
- `diagnostics/agibot_visual_preprocessing_ep78_row26_20260906/`
- `diagnostics/agibot_stationary_phase_ep78_20260906/`
- `diagnostics/agibot_warmup26_ab_20260906/`
- `submodules/Isaac-GR00T/gr00t/policy/gr00t_policy.py`
- `submodules/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- Checkpoint contract：`../checkpoints/gr00t_n17_sfted_stack_bowls_with_ee_pose/checkpoint-10000/README.md`

## 10. 变更日志

### 2026-09-09

- 新增 denoise 4/8/16/32、每组 20 episodes 的配对 stack-bowls sweep 和 camera-video launcher。
- 新增 denoise-32 policy config，并增加回归测试保证四组除 denoise steps 外保持一致。
- 确认 5557 当前没有 server；记录同一 patched server 可服务全部四组，以及 local table asset 的前置条件。
- 修复 `docker/run_docker.sh` 对 Cursor Remote SSH socket 符号链接的挂载：先解析真实 Unix socket 再传给
  Docker；用短生命周期容器验证启动成功。另确认 `docker exec` 的目标容器名为
  `isaaclab_arena-latest`，输出目录环境变量和容器名不能被换行拆开。
- 恢复被清理的 RoboDojo local assets，并实际验证 sweep 已越过 warmup、进入 server-backed rollout 且
  三路 camera MP4 正常写入；随后停止 smoke，保留 5557 server 供正式运行。
- 新增 `AGIBOT_GR00T_N17_DENOISE_SWEEP_EVAL.md`，集中记录 server、Docker、容器内 sweep、报告路径、
  seed 语义、故障处理和 local asset 恢复步骤。

### 2026-09-06

- 首次将 2026-09-02 至 2026-09-06 的排查过程集中整理成本文档。
- 记录 asset path、checkpoint contract、teacher forcing、坐标修复、oracle replay、controller、gripper、server deterministic inference、RTC、clipping、chunk selection、translation anchor、state delay、modality ablation 和 phase 检查。
- 完成 joint-exact camera factorial：确认 state 精确一致时 live render 仍将左 full-40 error 从 43.3 mm 放大至 140.2 mm，right wrist 主效应最大。
- 逐项验证真实 processor：camera order、RGB、shape、crop/resize、token/state wiring 均通过；H.264/ECC 只部分改善。
- 发现 startup observation/label phase aliasing；warmup 26 相对 warmup 10 将共同阶段左臂闭环误差降低 71.3%。
- 完成 dataset-camera 闭环因果实验：right-wrist-only 和 all-camera substitution 分别将误差降低 51.0% 和 69.2%。
- 当前未解决项进一步收敛为 generation-time renderer/camera runtime provenance 与 checkpoint 对合法小像素差异的鲁棒性二选一。
- 用当前代码和 warmup 26 完成 1 个 live-camera 1800-step 录像 eval；pipeline 无异常但任务 timeout，成功率 0/1。
- 定位并修复本次运行时版本问题：旧 5557 server 未加载 denoise metadata 补丁；用同 checkpoint 重启后契约通过。
- 完成 denoise 4 与 denoise 8 的同 seed 完整录像 A/B；两者均未成功，denoise 4 的可见行为更不稳定，保留 8。
- 完成 denoise 16 完整录像对照；同样在 approach 阶段异常并 timeout，未优于 8-step，形成 4/8/16 固定-seed baseline。

### 后续记录模板

```markdown
### YYYY-MM-DD HH:MM UTC — <实验名>

**排查点**

<明确假设>

**排查方式**

- Commit / config / server PID or startup time：
- Dataset / episode / frame：
- Seed：
- Control group：
- Experiment group：
- Metrics：

**证据与产物**

- 数值：
- Trace：
- Video：
- Log：

**结论**

<已证实 / 已排除 / 部分成立 / 待验证 / 无效实验>

**影响/下一步**

<保留、回滚或下一实验>
```
