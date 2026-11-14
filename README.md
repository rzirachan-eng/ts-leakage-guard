# ts-leakage-guard

**Stdlib-only checks that catch look-ahead leakage in time-series forecasting pipelines.** You hand it your own feature or fit/predict function; it reruns the function on perturbed copies of the same data and fails if the answer for the past changes when only the future (or a not-yet-known label) was touched.

给时间序列预测流水线用的防"偷看未来"检验：只用 Python 标准库。你把自己的特征函数或训练预测函数交给它，它在被扰动的数据副本上重跑；如果只改了未来（或尚未成熟的标签），过去的结果却变了，就报错并指出是哪一行。

> 仓库里所有数据都是合成的，没有任何真实数据或模型。`examples/` 里为每种检验各写了一个**正确**和一个**故意泄漏**的流水线，用来证明检验真的抓得到。

## 检验一览

| 检验 | 做法 | 能抓到的典型错误 |
|---|---|---|
| `prefix_invariance` | 删掉某个时点之后的所有行，旧行的特征必须一字不变 | 居中窗口、用全序列均值/方差做标准化、任何越过自身时点的统计量 |
| `future_mutation_invariance` | 把某时点之后的行改成极端值，旧行的特征必须不变 | 同上，且对"只在某些值上才泄漏"更敏感 |
| `immature_label_invariance` | 把截止时点时还未知的标签换成毒值，结果必须不变 | 训练时用了尚未成熟（还没发生）的标签 |
| `deterministic_rerun` | 同一流程重跑，输出的 SHA-256 必须一致 | 未固定随机种子、依赖运行顺序或时间 |
| `failure_produces_no_output` | 强制让流程失败，不得留下（或偷偷复用）输出文件 | 失败后静默读取旧预测、写了一半的输出 |
| `permuted` | 同分布、可复现的标签置换，作为负对照的素材 | 用来检验"打乱标签后分数应当崩掉" |

## 快速试用

需要 Python 3.9+，无第三方依赖。

```bash
python examples/demo.py                      # 对 7 对正确/泄漏流水线逐个给出判定
python -m unittest discover -s tests         # 29 个测试
```

演示输出（节选）：

```
PASS  prefix_invariance / trailing_mean
LEAK  prefix_invariance / centered_mean
      -> row 0 (t=0) changes when data after cutoff 0 is removed: [100.0] != [102.13597348398076]
LEAK  immature_label_invariance / all_labels
      -> result at cutoff 20 depends on 11 labels that are not yet known: ...
```

## 在自己的流水线上用

```python
from leakage_guard import prefix_invariance, future_mutation_invariance

# rows: 按时间排序的 dict 列表；my_features(rows) 返回"每行一个特征"
prefix_invariance(my_features, rows)

def spike(row):
    row["value"] = 1e9
    return row

future_mutation_invariance(my_features, rows, cutoff=20, mutate=spike)
```

发现泄漏时抛出 `CheckFailure`（`AssertionError` 的子类），因此可以直接写进 `unittest` 或 `pytest`，并在 CI 里自动运行。

## 限制

- 通过检验**不等于**没有泄漏。这些检验只能发现"改变未来/未成熟标签会影响过去结果"这一类泄漏；它们发现不了由样本选择、特征筛选、超参数搜索、或评估集本身被反复使用带来的泄漏。
- 检验依赖你提供的扰动方式；扰动选得太温和可能漏报。
- `deterministic_rerun` 只比较输出的哈希，不解释为什么不一致。

## 开发说明

检验的设计来自作者对预测流水线做防泄漏审计的实践；代码实现借助了 AI 编程工具，并通过 `tests/` 里的测试验证（每个检验都有"通过"和"抓到泄漏"两类用例）。GitHub Actions 在 Ubuntu/Windows × Python 3.9/3.12 上自动运行测试。
