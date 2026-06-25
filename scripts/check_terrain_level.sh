#!/usr/bin/env bash
# 检查最新的 training run 中 terrain_level 是否在上涨
# 用法: ./scripts/check_terrain_level.sh
# 可选: ./scripts/check_terrain_level.sh logs/rsl_rl/unitree_go2_velocity/2026-06-24_16-01-51

LOG_DIR="${1:-$(ls -dt logs/rsl_rl/unitree_go2_velocity/2*/ | head -1)}"
EVENT_FILE=$(ls "$LOG_DIR"/events.out.* 2>/dev/null | head -1)

if [ -z "$EVENT_FILE" ]; then
    echo "❌ 未找到 tensorboard event 文件"
    echo "   检查目录: $LOG_DIR"
    exit 1
fi

python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('$EVENT_FILE')
ea.Reload()

tags = ea.Tags().get('scalars', [])
if 'Curriculum/terrain_levels' not in tags:
    print('⏳ 训练尚未写入 terrain_level 数据')
    exit(0)

ev = ea.Scalars('Curriculum/terrain_levels')
first, last = ev[0], ev[-1]
delta = last.value - first.value

print(f'📁 目录: $LOG_DIR')
print(f'📊 训练轮次: {first.step} → {last.step}')
arrow = '▲' if delta > 0 else '▼'
print(f'   terrain_level: {first.value:.2f} → {last.value:.2f} ({arrow} {abs(delta):.2f})')

if delta > 0.5:
    print('✅ 课程学习正常推进中！')
elif delta > 0.1:
    print('📈 课程学习缓慢上涨')
elif delta < -0.5:
    print('⚠️  terrain_level 在下降，检查课程条件')
else:
    print('⏳ terrain_level 基本持平，继续训练观察')

# 同时显示其他关键指标
for tag, label in [('Curriculum/lin_vel_cmd_levels', '速度指令等级'),
                     ('Train/mean_reward', '平均奖励'),
                     ('Train/mean_episode_length', '平均回合长度')]:
    if tag in tags:
        e = ea.Scalars(tag)
        print(f'   {label}: {e[0].value:.2f} → {e[-1].value:.2f}')
" 2>&1
