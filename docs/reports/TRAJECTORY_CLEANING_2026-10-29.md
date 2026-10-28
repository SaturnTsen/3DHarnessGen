# GPT-6-Astra 轨迹清洗结果（2026-10-29）

脚本：`scripts/clean_astra_trajectories.py`。只读取四种场景的 `gpt-6-astra/*/post_train_trajectory/trajectory.jsonl`，原始数据只读。输出为 `data/trajectory_cleaned/<setting>/<object>/trajectory.jsonl`，图片为同目录 `imgs/<sha256>.png`。

## 清洗规则

- 每个原有训练目标独立输出一行 v3 记录，当前目标另外保留；保留该目标原始可见历史中的开头用户任务、developer 指令及精简工具声明。
- 移除经全量调用检查确认为未使用的候选工具及 exec 内嵌说明，包括 web、图片生成、目标管理、协作 agent、询问用户和 MCP 资源发现/读取；实际出现过的工具保留。
- 按 `ERROR` 步骤及实际返回中的明确失败标记移除失败目标、调用和返回；保持独立文本块，能够识别视角标签夹杂的错误结果。UNKNOWN 不直接视为失败。
- 对每个目标保留此前最近五轮未失败的调用及返回，相关 agent 说明文字随轮保留；一次带调用的 assistant 消息计一轮，wait 同样计一轮。当前目标的后续返回、图片或产物不进入模型输入。
- 重排 messages.index、context_indices、target_index；保留原 step.id，session_id 加上清洗目标后缀保证每行独立。来源路径重新定位，旧索引映射保存在 cleaning_manifest.jsonl。
- 只复制保留消息引用的原图；图片字节不变，重复引用及顺序不变。每个对象的 imgs 目录是该对象所有保留样本所需图片的并集。

## 输出

| setting | 输出目标数 | 剔除失败目标 | 图片文件 | 原输入平均 token | 清洗后平均 token | 超过 49,152 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-view | 400 | 1 | 400 | 7,667 | 7,667 | 0 |
| Multi-view | 400 | 4 | 1,600 | 9,053 | 9,053 | 0 |
| ActiveVisual | 3,395 | 110 | 4,441 | 49,308 | 21,747 | 0 |
| Full3DInteraction | 3,284 | 152 | 3,050 | 61,669 | 24,716 | 90 |

输入输出均值比较的是同一批最终保留目标，统计全部输出而非仅训练集。共 400 个对象 JSONL、7,479 条目标、9,491 张图片；剔除 267 条失败目标，277 张源图片未被任何保留样本引用。

Full3DInteraction 仍有 90 条超过当前 49,152 token 预算，最长 79,758；脚本未执行额外长度筛选或截断。训练前仍需按实际模型配置编码检查。窗口截取也不重写调用代码或补造跨窗口变量状态。

## 验证

- 78 项脚本/轨迹约定测试、71 项配置/数据/模板测试通过；脚本及新增测试的静态类型检查通过。
- 对全部 7,479 行重新检查 v3 格式、目标及初始消息不变、最后五轮选择、调用与返回关联、无未来信息、图片内容及顺序，并检查图片目录没有未引用文件；400 个输入文件哈希保持一致。
- 全部输出用 27B 实际 tokenizer 统计长度；9 条完整 Processor 编码核对、8 条标签核对通过。没有加载模型权重或启动训练。
- 实际 data.files 读取检查覆盖四种场景，结果另存 reader_verification.json。

## 使用

```bash
pixi run python -m scripts.clean_astra_trajectories
```

默认输出已生成；脚本拒绝覆盖已有目录。重跑可用 `--output-root` 指定新的目录，`--settings` 选择场景，`--keep-tool-rounds` 调整历史轮数。清洗结果被训练读取器和独立轨迹校验器接受；训练时通过 data.files 显式选择对象 JSONL，image_root 保持 null。输出属于派生数据，仍须先按对象划分训练/验证/测试。

数据目录内提供 cleaning_report.json、cleaning_manifest.jsonl、verification_report.json、token_report.json 和 reader_verification.json。
