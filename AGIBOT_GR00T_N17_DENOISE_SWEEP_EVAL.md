# AgiBot GR00T N1.7 Stack-Bowls Denoise Sweep 复现指南

最后验证：2026-09-09 UTC

## 1. 评测内容

- Task：`agibot_stack_bowls`
- Checkpoint：`/home/ubuntu/projects/isaaclab_arena_proj/checkpoints/gr00t_n17_sfted_stack_bowls_with_ee_pose/checkpoint-10000`
- Denoising steps：`4 / 8 / 16 / 32`
- Episodes：每组 20，共 80
- Environment base seed：`42`
- Camera warmup：26 environment steps
- Action execution chunk：16
- Action sampling：5 次 server response 取逐元素 median
- Camera recording：head、left wrist、right wrist，512×512、15 FPS
- GR00T server：宿主机 `127.0.0.1:5557`

四个 denoise run 使用相同的 base seed 42，各自重建环境后生成相同顺序的随机场景，用于配对比较。
同一 run 的 20 个 episode 不使用 42–61 这 20 个独立 seed；它们使用 seed 42 初始化的一条可复现
随机数流，每次 reset 继续抽取新的 bowl jitter。因此 JSONL 中的 `seed` 字段均为 42，但 episode
场景并不相同。

## 2. 运行前检查

在宿主机执行：

```bash
cd /home/ubuntu/projects/isaaclab_arena_proj/IsaacLab-Arena

test -d /home/ubuntu/projects/isaaclab_arena_proj/checkpoints/gr00t_n17_sfted_stack_bowls_with_ee_pose/checkpoint-10000
test -f /tmp/isaaclab_arena_agibot_assets/arena_local/robodojo_table/robodojo_table.usda
test -f /tmp/isaaclab_arena_agibot_assets/arena_local/bowl.usdz
test -f /tmp/isaaclab_arena_agibot_assets/arena_local/simple_room_nolight/simple_room_nolight.usd
```

四条命令均应返回 exit code 0。`/tmp` 可能在重启或系统清理后丢失；资产缺失时按附录 A 恢复。

## 3. Terminal 1：启动 GR00T server

在宿主机运行：

```bash
cd /home/ubuntu/projects/isaaclab_arena_proj/IsaacLab-Arena
./isaaclab_arena_gr00t/scripts/serve_agibot_stack_bowls_n17.sh
```

该脚本默认使用上面的 checkpoint、`cuda:0` 和端口 5557。一个 patched server 可以顺序服务
全部 4/8/16/32 run，不需要每组重启。

另开终端检查端口：

```bash
timeout 3 bash -c '</dev/tcp/127.0.0.1/5557'
echo $?
```

应输出 `0`。

## 4. Terminal 2：启动 IsaacLab-Arena 容器

在宿主机运行：

```bash
cd /home/ubuntu/projects/isaaclab_arena_proj/IsaacLab-Arena
./docker/run_docker.sh
```

保持此终端和容器 shell 开启。当前标准容器名为 `isaaclab_arena-latest`，使用 host networking，
所以容器可以直接访问宿主机的 `127.0.0.1:5557`。宿主机 `/tmp` 和仓库目录也映射到容器。

## 5. Terminal 2 / 容器内：启动正式 sweep

在容器 shell 中运行：

```bash
cd /workspaces/isaaclab_arena

ARENA_LOCAL_ASSET_DIR=/tmp/isaaclab_arena_agibot_assets/arena_local \
AGIBOT_GR00T_SERVER_PORT=5557 \
AGIBOT_GR00T_OUTPUT_BASE_DIR=/workspaces/isaaclab_arena/results/agibot_gr00t_n17_denoise_sweep_20ep \
./isaaclab_arena_gr00t/scripts/run_agibot_stack_bowls_denoise_sweep.sh
```

已对这条命令做过真实启动验证：环境、三路 camera、action/observation manager 和 denoise-4 policy
contract 均成功初始化；完成 26-step warmup 后进入 server-backed rollout，并持续写入三路 MP4。
验证后按要求停止了 smoke，5557 server 未被停止。

首次运行可能长时间停在以下日志：

```text
Waiting for RtPso async group async compilation
```

2026-09-09 的首次启动为 RTX cache 编译额外等待约 240 秒，不是 eval 卡死。等待出现以下日志即可确认
进入正式 rollout：

```text
Completed setting up the environment...
GR00T remote contract: ... denoise=4 ...
Recording per-episode per-camera videos to: ...
Episodes: 0%|...| 0/20
```

## 6. 输出、视频和成功率报告

宿主机和容器内共享同一输出目录：

```text
/home/ubuntu/projects/isaaclab_arena_proj/IsaacLab-Arena/results/agibot_gr00t_n17_denoise_sweep_20ep/<timestamp>/
/workspaces/isaaclab_arena/results/agibot_gr00t_n17_denoise_sweep_20ep/<timestamp>/
```

完整运行结束后主要文件为：

```text
<timestamp>/
├── arena_experiment_result.json
├── index.html
├── report/
├── agibot_stack_bowls_gr00t_n17_denoise4_warmup26_20ep/
│   ├── episode_results_rebuild0.jsonl
│   └── robot-cam-rebuild0-env0-*-episode-*.mp4
├── agibot_stack_bowls_gr00t_n17_denoise8_warmup26_20ep/
├── agibot_stack_bowls_gr00t_n17_denoise16_warmup26_20ep/
└── agibot_stack_bowls_gr00t_n17_denoise32_warmup26_20ep/
```

- `arena_experiment_result.json`：四组的最终 success rate 汇总。
- `index.html`：可视化总报告。
- `episode_results_rebuild0.jsonl`：逐 episode 的 success、length、seed 等原始结果。
- `robot-cam-*.mp4`：三路 camera video。

查找最新结果：

```bash
find /workspaces/isaaclab_arena/results/agibot_gr00t_n17_denoise_sweep_20ep \
  -name arena_experiment_result.json -o -name episode_results_rebuild0.jsonl
```

如果中途停止，已经完成的 episode 仍会保留在对应 JSONL 中，但最终 aggregate JSON/HTML 可能尚未生成。
当前 launcher 没有断点续跑功能；再次执行会新建 timestamp 目录并从 denoise 4 开始。

## 7. 常见故障

### `Missing ... robodojo_table.usda`

`/tmp` 中的 local assets 被清理。按附录 A 恢复，不要删除或绕过 launcher 的检查。

### `No GR00T server is listening at 127.0.0.1:5557`

Terminal 1 的 server 未启动、已经退出或端口不同。重新启动 server，或同时设置：

```bash
AGIBOT_GR00T_SERVER_PORT=<实际端口>
```

### 输出目录 `Permission denied`

只修复本次专用目录，不递归修改已有结果：

```bash
mkdir -p /workspaces/isaaclab_arena/results/agibot_gr00t_n17_denoise_sweep_20ep
chown ubuntu:ubuntu /workspaces/isaaclab_arena/results/agibot_gr00t_n17_denoise_sweep_20ep
```

如果当前容器 shell 不是 root，则用宿主机上的 `docker exec` 以 root 创建一次。

### Docker 报 Cursor SSH socket `file exists`

当前 `docker/run_docker.sh` 已修复：会解析 Cursor `SSH_AUTH_SOCK` 符号链接后再挂载。若使用未包含
该补丁的旧 checkout，可临时运行：

```bash
env -u SSH_AUTH_SOCK ./docker/run_docker.sh
```

## 附录 A：恢复 `/tmp` 中的 local assets

以下命令在宿主机运行。先下载 RoboDojo 原始 bowl 和 room：

```bash
hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include 'Assets/Object/RoboDojo/Rigid/bowl/00001/**' \
  --include 'Assets/Room/Simple_Room_nolight/**' \
  --local-dir /tmp/isaaclab_arena_agibot_assets/hf \
  --max-workers 8

mkdir -p /tmp/isaaclab_arena_agibot_assets/arena_local/robodojo_table

ln -sfn \
  /tmp/isaaclab_arena_agibot_assets/hf/Assets/Object/RoboDojo/Rigid/bowl/00001/object.usdz \
  /tmp/isaaclab_arena_agibot_assets/arena_local/bowl.usdz

ln -sfn \
  /tmp/isaaclab_arena_agibot_assets/hf/Assets/Room/Simple_Room_nolight \
  /tmp/isaaclab_arena_agibot_assets/arena_local/simple_room_nolight
```

创建 table USD：

```bash
cat > /tmp/isaaclab_arena_agibot_assets/arena_local/robodojo_table/robodojo_table.usda <<'USDA'
#usda 1.0
(
    defaultPrim = "RoboDojoTable"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RoboDojoTable"
{
    def Cube "Table" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {
        rel material:binding = </RoboDojoTable/Looks/Mahogany>
        double size = 1
        uniform token physics:approximation = "none"
        color3f[] primvars:displayColor = [(0.08, 0.025, 0.01)]
        uniform token primvars:displayColor:interpolation = "constant"
        float3 xformOp:scale = (1.1, 1.4, 0.05)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }

    def Scope "Looks"
    {
        def Material "Mahogany"
        {
            token outputs:mdl:surface.connect = </RoboDojoTable/Looks/Mahogany/Mahogany_Planks.outputs:out>
            token outputs:surface.connect = </RoboDojoTable/Looks/Mahogany/Mahogany_Planks.outputs:out>

            def Shader "Mahogany_Planks"
            {
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @./Mahogany_Planks.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "Mahogany_Planks"
                bool inputs:project_uvw = 1
                bool inputs:world_or_object = 0
                float2 inputs:texture_scale = (1, 1)
                token outputs:out
            }
        }
    }
}
USDA
```

下载与已验证运行一致的 NVIDIA Mahogany material：

```bash
TABLE_DIR=/tmp/isaaclab_arena_agibot_assets/arena_local/robodojo_table
MATERIAL_URL=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/NVIDIA/Materials/Base/Wood

mkdir -p "$TABLE_DIR/Mahogany_Planks"

curl -fL "$MATERIAL_URL/Mahogany_Planks.mdl" \
  -o "$TABLE_DIR/Mahogany_Planks.mdl"
curl -fL "$MATERIAL_URL/Mahogany_Planks/Mahogany_Planks_BaseColor.png" \
  -o "$TABLE_DIR/Mahogany_Planks/Mahogany_Planks_BaseColor.png"
curl -fL "$MATERIAL_URL/Mahogany_Planks/Mahogany_Planks_ORM.png" \
  -o "$TABLE_DIR/Mahogany_Planks/Mahogany_Planks_ORM.png"
curl -fL "$MATERIAL_URL/Mahogany_Planks/Mahogany_Planks_N.png" \
  -o "$TABLE_DIR/Mahogany_Planks/Mahogany_Planks_N.png"
```

最后在容器内验证三个 stage：

```bash
cd /workspaces/isaaclab_arena

./submodules/IsaacLab/_isaac_sim/python.sh -c '
from pxr import Usd
paths = [
    "/tmp/isaaclab_arena_agibot_assets/arena_local/robodojo_table/robodojo_table.usda",
    "/tmp/isaaclab_arena_agibot_assets/arena_local/bowl.usdz",
    "/tmp/isaaclab_arena_agibot_assets/arena_local/simple_room_nolight/simple_room_nolight.usd",
]
print([(path, bool(Usd.Stage.Open(path))) for path in paths])
'
```

三个结果均应为 `True`。
