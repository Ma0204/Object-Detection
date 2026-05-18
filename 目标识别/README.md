# 场景类别表（目标识别）

这里把任务写成 **「一张场景图里有哪些类别、每类几个」**，对外只谈 **真值类别表** 与 **预测类别表** 是否一致；**不把画框当作接口或主叙事**（底层仍用同一套前向，在内部把高分候选聚成计数表）。

## 代码结构（刻意与「检测脚本一排 whitebox_*.py」区分开）

| 文件 | 作用 |
|------|------|
| `run_recognition.py` | **唯一推荐入口**：`attack`（白盒/黑盒多种策略）、`noise`（高斯示例） |
| `recognition_core.py` | 词汇表、真值表、预测表、`tables_match` |
| `recognition_perturb_whitebox.py` | 在 L∞ 下扰动场景图（梯度多步 / Adam 变体），不出现检测命名 |
| `recognition_benchmark.py` | 跑 VOC 子集，统计 **干净表准确率**、**受限表破坏率** |
| `recognition_backend.py` | 仅负责把上级目录加入 `sys.path` 并安全加载共享实现（文件名历史遗留，不必写进论文叙述） |

黑盒在说明里称为 **「查询式 A/B/C」**，对应上级工程里三种无梯度优化核；**指标与叙事**已换成「表」而不是 GT-ASR。

## 运行示例

在 **`目标识别`** 目录下：

```bash
cd 目标识别
python run_recognition.py attack --strategy fgsm --weights ..\yolov8n.pt --num_eval 80
python run_recognition.py attack --strategy pgd --weights ..\yolov8n.pt
python run_recognition.py attack --strategy adam --weights ..\yolov8n.pt
python run_recognition.py attack --strategy ttba --weights ..\yolov8n.pt --num_eval 15
python run_recognition.py attack --strategy seri --weights ..\yolov8n.pt --num_eval 15
python run_recognition.py attack --strategy adba --weights ..\yolov8n.pt --num_eval 15
python run_recognition.py noise --weights ..\yolov8n.pt --num_eval 200
```

白盒/黑盒子命令共用一套超参参数名；与「检测版」数值可对齐的项（如 `eps`、`conf`、`infer_imgsz`）仍保留，便于对照实验。

## 指标

- **干净表准确率**：有前景真值的场景中，过滤后的预测计数表与真值表完全一致的比例。  
- **受限表破坏率**：仅在干净表已全对的前提下，扰动后预测表是否被「破坏」（不再与真值表一致）。

更复杂的物理噪声（雨丝、椒盐块等）请继续使用上级目录的 **`object_detection_voc_noise_experiments.py`**；本目录的 `noise` 子命令为高斯 L∞ 下的表匹配示例。
