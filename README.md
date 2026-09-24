# MAS Communication Fault Experiments

LLM 多智能体系统通信故障实验代码、实验配置与分析脚本。此仓库是 2026-09-24 整理的本地代码快照，不包含论文、模型权重、原始实验日志或真实凭据。

## 内容

- `src/mas_faults/`：故障注入、工作流、证据验证与缓解实现。
- `run_*.py`、`scripts/`：实验入口、审计、并行运行与推理探针。
- `tests/`：恢复工作副本的离线测试。
- `mas_rq4_20260914/` 与 `mas_rq4_*.py`：后续 RQ4 实验及分析代码，作为独立版本保留。
- `analysis/rq123_20260915/`：RQ1–3 历史分析代码快照；其中的 `mas_faults` 只服务于该分析快照，不覆盖主源码。
- `reports/domain_pattern_audit.py`、`reports/*/analysis_config.json`：分域分析及矩阵配置。
- `deploy/`、`*-env.sh`：历史部署模板，运行前需调整机器目录、端口及设备分配。
- `SOURCE_MANIFEST.json`：逐文件来源、原始和打包后 SHA-256。

## 环境与离线测试

历史部署使用 Python 3.11。`requirements.lock.txt` 和 `conda-explicit-linux-64.txt` 是 Linux 部署快照，不适合直接当作 macOS 通用环境安装。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest -q
python -m pytest -q test_mas_rq4_analysis_20260914.py
# RQ4 使用自己的 legacy 源码，在独立 Python 进程验证
(cd mas_rq4_20260914 && PYTHONPATH="$PWD/legacy/src:$PWD/legacy:$PWD" python -m pytest -q -o pythonpath='' test_contract.py test_design.py test_exposure.py test_runner.py test_runtime.py)
```

默认测试禁用网络 socket（允许 Unix socket）。离线测试不能代替真实模型和站点验证。不要直接运行历史部署或批量实验脚本；先配置自己的服务与数据目录。模型认证通过环境变量设置，仓库中的 `local-no-secret` 和测试用 token 均为占位值。

## 分析输入

分析配置的输入路径已改成相对仓库根目录的路径；从仓库根目录执行。JSONL 原始数据未上传，需自行放入配置指定的位置后运行。`domain_pattern_audit.py` 使用 `reports/rq123_six_condition_inputs_20260916_v2/analysis_config.json`。缺少原始数据时不能重算论文统计。

## 复现边界

主代码取自 `mas_reproduction_20260910` 本地恢复工作副本；后续 RQ4 和分析快照单独保留，没有未经验证地混合覆盖不同版本。本次未核验与服务器最新代码是否一致。

历史恢复记录指出 `mas_faults.causal_trace_report` 缺失，且 SWE 多轮入口与恢复的 SWE 模块存在接口版本差异；相关路径不能声称已完整复现。历史实验完成数和旧测试通过数不作为本次验证。当前验证结果见 `VALIDATION.md`。

论文稿件、图稿、完整结果、模型和第三方 benchmark 仓库均未纳入。未指定开源许可证；根据用户 2026-09-24 的最新决定，仓库保持公开。

## 在线历史回放

[MAS Fault Observatory](https://SonGoKu21.github.io/mas_communication/) 展示历史实验，不执行实时模型调用。界面已按最新论文 revision 7 对齐四个 RQ 与 Finding 1–10；网页、脱敏数据及重建工具位于 `web/`。具体证据范围及案例缺口见 [web/README.md](web/README.md)。
