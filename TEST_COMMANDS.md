# Musculoco 常用测试命令

这份文档整理了你当前项目最常用的一组命令，按训练 -> 测试 -> 分析流程组织。

## 1. 启动 AMP 训练

在仓库根目录执行：

env PYTHONPATH=$(pwd) uv run python experiments/14_AMP_latent/launcher_walk.py


## 2. 打开 TensorBoard 查看训练曲线

示例（替换为你本次训练日志目录）：

uv run tensorboard --logdir logs/14_AMP_latent_walk_2026-03-16_22-56-25


## 3. 用 .msh 模型跑一次评测并导出 CSV/NPZ/MP4

说明：
- 这个命令会生成 recording.mp4、torques_and_ctrl.csv、torques_and_ctrl.npz
- 这里使用了 --use-box-feet，与你当前 AMPmesh.msh 的观测维度匹配

PYTHONPATH=$(pwd) .venv/bin/python experiments/eval/record_joint_torques.py \
  --checkpoint experiments/15_AMP_attention/logs/15_AMP_attention_walk_2026-03-19/0/agent_epoch_666_J_987.263211.msh \
  --episodes 1 \
  --max-steps 1000 \
  --no-absorbing \
  --deterministic \
  --record-pos \
  --record-vel \
  --record-ctrl \
  --record-actions \
  --record-video \
  --headless \
  --use-box-feet \
  --outdir experiments/15_AMP_attention/runs/walk1 \
  --csv torques_and_ctrl.csv


## 4. 从评测 CSV 生成 gait band 数据

.venv/bin/python experiments/eval/analyze_gait_bands.py \
  --csv experiments/15_AMP_attention/runs/walk1/torques_and_ctrl.csv \
  --out experiments/15_AMP_attention/runs/walk1/bands \
  --out-data experiments/15_AMP_attention/runs/walk1/bands/bands_summary.npz

排错（如果出现 Invalid lag search range）：

1. 先检查 CSV 行数是否太少（例如只有几十行）

.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv('experiments/14_AMP_latent/runs/AMP_walk2/torques_and_ctrl.csv')
print('rows =', len(df), 'duration =', float(df['time'].iloc[-1] - df['time'].iloc[0]))
PY

2. 如果行数太少，重新执行第 3 步（关键是带上 --no-absorbing）再执行第 4 步


## 5. 画 ASCII 参考误差带（左侧）

.venv/bin/python experiments/eval/plot_ascii_reference_bands.py \
  --layout sagittal \
  --ascii-root ASCII-files \
  --side L \
  --overlay-band-npz experiments/15_AMP_attention/runs/walk1/bands/bands_summary.npz \
  --model-mass-kg 80 \
  --auto-align-phase \
  --ref-plot trials \
  --out experiments/15_AMP_attention/runs/walk1/ascii_ref/amp_walk1_ascii_overlay_L.png \
  --metrics-out experiments/15_AMP_attention/runs/walk1/ascii_ref/amp_walk1_ascii_metrics_L.csv


## 6. 画 ASCII 参考误差带（右侧）

.venv/bin/python experiments/eval/plot_ascii_reference_bands.py \
  --layout sagittal \
  --ascii-root ASCII-files \
  --side R \
  --overlay-band-npz experiments/14_AMP_latent/runs/AMP_walk1/bands/bands_summary.npz \
  --model-mass-kg 130 \
  --auto-align-phase \
  --ref-plot trials \
  --out experiments/14_AMP_latent/runs/AMP_walk1/ascii_ref/amp_walk1_ascii_overlay_R.png \
  --metrics-out experiments/14_AMP_latent/runs/AMP_walk1/ascii_ref/amp_walk1_ascii_metrics_R.csv


## 7. 快速查看结果文件

ls -lh experiments/14_AMP_latent/runs/AMP_walk1
ls -lh experiments/14_AMP_latent/runs/AMP_walk1/ascii_ref


## 8. 推荐执行顺序

1. 先训练得到新的 .msh
2. 跑第 3 步导出评测结果
3. 跑第 4 步生成 band
4. 跑第 5 和第 6 步生成误差带图与 metrics
5. 对比 amp_walk1_ascii_metrics_L.csv 与 amp_walk1_ascii_metrics_R.csv 的 rmse_weighted
