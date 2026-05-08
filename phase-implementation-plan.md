# Triton-agent / Apeinx OpEvolver 阶段化落地计划

> **核心定位**: 低成本开发高质量 GPU 算子
>
> 面向低成本高质量算子开发的轻量自优化智能体

---

## 总体架构概览

```text
Triton-agent 完整架构 =
  算子注册表
  + 算子契约
  + PyTorch 正确性基线
  + Triton 候选变体生成
  + 编译器反馈
  + 正确性验证
  + 性能 Profile
  + MicroRL/Bandit 调优
  + 晋升/回滚
  + Evidence Replay
  + 轻量 PyTorch/Runtime 适配
```

### 核心运行链路

```text
注册算子 → 读取算子契约 → 生成 Triton 候选变体 → 编译
→ 正确性验证 → 性能 Profile → MicroRL/Bandit 更新选择策略
→ 选择最快且正确的 variant → 晋升到 leaderboard
→ 保存 episode → 可回放/可对比/可回滚
```

### 10 层架构模块

| 层 | 模块 | 职责 |
|----|------|------|
| 0 | Operator Registry | 算子注册与管理 |
| 1 | Op Contract | 算子契约定义搜索空间 |
| 2 | Reference & TestCase | PyTorch 正确性基线 |
| 3 | Candidate Generator | 候选 Triton 变体生成 |
| 4 | Compile | Triton JIT 编译 |
| 5 | Verifier | 正确性验证门控 |
| 6 | Profiler | 性能测量与分析 |
| 7 | MicroRL/Bandit | 轻量策略优化 |
| 8 | Selector/Promotion | 最优变体选择与晋升 |
| 9 | Evidence/Replay | 记录回放与排行榜 |
| 10 | Integration Adapter | 轻量运行时适配 |

---

## 目录结构

```text
triton-agent/
├── agent/                          # 智能体核心
│   ├── planner.py                  # 根据 op/shape/device 决定搜索策略
│   ├── generator.py                # 生成候选 Triton variant
│   ├── repairer.py                 # 根据 compile/verify/profile 反馈修复
│   ├── selector.py                 # 选择最佳 variant
│   └── promoter.py                 # 晋升 / 回滚
│
├── core/                           # 核心基础设施
│   ├── registry.py                 # 算子注册表
│   ├── contract.py                 # contract.yaml 解析
│   ├── spec.py                     # shape/dtype/device profile
│   ├── compiler.py                 # Triton compile
│   ├── verifier.py                 # 正确性验证
│   ├── profiler.py                 # 性能测试
│   ├── reward.py                   # reward / score
│   ├── replay.py                   # 回放
│   └── storage.py                  # episode / leaderboard 存储
│
├── microrl/                        # 轻量强化学习
│   ├── state.py                    # op + shape + dtype + device
│   ├── action.py                   # template/config 动作空间
│   ├── bandit.py                   # UCB / Thompson Sampling
│   ├── reinforce_lite.py           # REINFORCE 轻量版 (后置)
│   ├── grpo_lite.py                # GRPO 轻量版 (后置)
│   └── trainer.py                  # 策略更新
│
├── ops/                            # 算子实现
│   ├── rmsnorm/
│   │   ├── contract.yaml
│   │   ├── reference.py
│   │   ├── templates/
│   │   │   ├── triton_v1.py
│   │   │   └── triton_v2.py
│   │   ├── verify.py
│   │   └── benchmark.py
│   ├── rope/
│   │   ├── contract.yaml
│   │   ├── reference.py
│   │   ├── templates/
│   │   ├── verify.py
│   │   └── benchmark.py
│   └── fused_bias_gelu/
│       ├── contract.yaml
│       ├── reference.py
│       ├── templates/
│       ├── verify.py
│       └── benchmark.py
│
├── integrations/                   # 集成适配
│   ├── pytorch_wrapper.py
│   ├── torch_compile_compare.py
│   ├── vllm_mock.py
│   └── sglang_mock.py
│
├── episodes/                       # 优化记录
│   └── .gitkeep
│
├── leaderboard/                    # 排行榜
│   ├── leaderboard.sqlite
│   └── best_configs.json
│
├── benchmarks/                     # 基准测试
│   ├── bench_rmsnorm.py
│   ├── bench_rope.py
│   └── bench_fused_bias_gelu.py
│
├── tests/                          # 单元测试
│   ├── test_contract.py
│   ├── test_verifier.py
│   ├── test_profiler.py
│   ├── test_replay.py
│   └── test_ops.py
│
├── cli.py
├── pyproject.toml
└── README.md
```

---

## 阶段化落地计划

---

### Phase 0: 基础框架搭建 (Week 1-2)

**目标**: 建立项目骨架、核心基础设施和可运行的空管线。

#### 0.1 项目初始化
- [ ] 创建 `pyproject.toml`，配置项目元信息与依赖
- [ ] 配置 Python 虚拟环境和 `triton`, `torch`, `pyyaml` 等核心依赖
- [ ] 编写 `cli.py` 骨架（click/typer 框架）

#### 0.2 核心基础设施
- [ ] `core/contract.py` — contract.yaml 解析器，支持 OpContract dataclass
- [ ] `core/spec.py` — OpState dataclass（op_name, B, T, D, dtype, device, gpu_name, baseline_latency_us, historical_best_config）
- [ ] `core/registry.py` — 算子注册表，支持 register/get/list 算子

#### 0.3 数据结构定义
- [ ] OpAction dataclass（template_id, block_size, num_warps, num_stages, vectorize, fusion）
- [ ] CandidateResult dataclass（compile_pass, verify_pass, latency, speedup, variance, memory_peak, reward, promoted）
- [ ] Episode 存储格式定义

#### 0.4 测试框架
- [ ] `tests/test_contract.py` — 测试契约解析
- [ ] 集成 pytest 和 CI 骨架

**交付物**:
- 可运行的 `pip install -e .`
- `triton-agent --help` 可输出帮助信息
- contract.yaml 解析通过单元测试

---

### Phase 1: MVP 核心闭环 (Week 3-6)

**目标**: 在 3 个 P0 算子上跑通完整的"生成→编译→验证→Profile→选择→记录"闭环。

#### 1.1 P0 算子实现
- [ ] **a. RMSNorm** — contract.yaml + reference.py + templates (v1/v2) + verify.py + benchmark.py
- [ ] **b. RoPE** — 同上五件套
- [ ] **c. fused_bias_gelu** — 同上五件套

#### 1.2 生成层 (Generator)
- [ ] `agent/generator.py` — 模板枚举 + grid search 生成候选变体
- [ ] 支持参数空间: BLOCK_SIZE, num_warps, num_stages, vectorize

#### 1.3 编译层 (Compiler)
- [ ] `core/compiler.py` — Triton JIT compile 封装
- [ ] 捕获编译错误，记录 compile log 和 compile time
- [ ] 失败变体写入错误日志供 repairer 使用

#### 1.4 验证层 (Verifier)
- [ ] `core/verifier.py` — correctness gate
- [ ] max_abs_error / mean_abs_error 检查
- [ ] NaN / Inf 检查
- [ ] shape / dtype 检查
- [ ] verify pass 才允许进入 profile

#### 1.5 性能层 (Profiler)
- [ ] `core/profiler.py` — 延迟测量 (p50/p90/p99)
- [ ] throughput 计算
- [ ] memory peak 和 bandwidth estimate
- [ ] warmup + repeat 轮次控制
- [ ] 输出 profile_result.json

#### 1.6 选择层 (Selector & Promotion)
- [ ] `agent/selector.py` — Best-of-N 策略选择最优 variant
- [ ] `agent/promoter.py` — 晋升满足条件的 variant，记录回滚目标
- [ ] 晋升条件: compile_pass + verify_pass + speedup >= min_speedup + variance <= threshold

#### 1.7 证据层 (Evidence & Replay)
- [ ] `core/storage.py` — SQLite leaderboard + JSONL episode 存储
- [ ] `core/replay.py` — episode 回放、对比、回滚功能
- [ ] Episode 目录结构: `episodes/{op}/{episode_id}/`

#### 1.8 Reward 系统
- [ ] `core/reward.py` — score 计算（compile +0.2, verify +0.6, speedup bonus, variance penalty）

#### 1.9 CLI 核心命令
- [ ] `triton-agent init <op>` — 初始化算子
- [ ] `triton-agent optimize <op> --shape ...` — 优化指定 shape
- [ ] `triton-agent leaderboard <op>` — 查看排行榜
- [ ] `triton-agent replay <episode_path>` — 回放
- [ ] `triton-agent compare <op> --baseline torch --variant best` — 对比

#### 1.10 单元测试
- [ ] `tests/test_verifier.py`
- [ ] `tests/test_profiler.py`
- [ ] `tests/test_replay.py`
- [ ] `tests/test_ops.py`

**交付物**:
- 3 个 P0 算子的完整"优化→验证→记录"闭环
- CLI 工具可完成端到端算子优化流程
- SQLite leaderboard 记录历史最优配置
- 单元测试覆盖率 > 60%

---

### Phase 2: Bandit 优化与算子扩展 (Week 7-10)

**目标**: 引入 Bandit 调优策略替代 Best-of-N，扩展 P1/P2 算子。

#### 2.1 MicroRL / Bandit 层
- [ ] `microrl/state.py` — OpState 向量化
- [ ] `microrl/action.py` — 动作空间编码（template + config）
- [ ] `microrl/bandit.py` — UCB (Upper Confidence Bound) 算法
- [ ] `microrl/bandit.py` — Thompson Sampling 算法
- [ ] `microrl/trainer.py` — 策略更新，记录 shape→best_config 映射
- [ ] `agent/planner.py` — 根据 state 决策搜索方向

#### 2.2 Repairer
- [ ] `agent/repairer.py` — 根据 compile error/verify failure 自动调整参数
- [ ] 编译失败 → 调整 BLOCK_SIZE/num_stages
- [ ] 验证失败 → 调整 tolerance/numerical 参数

#### 2.3 P1 算子实现
- [ ] **SwiGLU** — 五件套
- [ ] **quant_dequant** — 五件套
- [ ] **LayerNorm** — 五件套

#### 2.4 集成适配层
- [ ] `integrations/pytorch_wrapper.py` — PyTorch 算子包装器
- [ ] `integrations/torch_compile_compare.py` — torch.compile 对比基准

#### 2.5 Benchmark 套件
- [ ] `benchmarks/bench_rmsnorm.py` — 多 shape 综合基准
- [ ] `benchmarks/bench_rope.py`
- [ ] `benchmarks/bench_fused_bias_gelu.py`

**交付物**:
- Bandit (UCB / Thompson) 驱动的搜索策略
- 6 个算子族完成优化闭环
- 修复器自动处理编译/验证失败
- PyTorch wrapper 可用

---

### Phase 3: 策略优化与深度集成 (Week 11-14)

**目标**: 引入 REINFORCE-lite / GRPO-lite，扩展 P2/P3 高难度算子。

#### 3.1 高级策略优化
- [ ] `microrl/reinforce_lite.py` — REINFORCE 轻量版实现
- [ ] `microrl/grpo_lite.py` — GRPO 轻量版实现
- [ ] trainer 支持多策略切换 (Bandit / REINFORCE / GRPO)

#### 3.2 P2 算子实现
- [ ] **KV Append** — 五件套 + 融合优化
- [ ] **RoPE + KV Append** (融合) — contract + templates
- [ ] **simple matmul epilogue** — contract + templates

#### 3.3 P3 算子实现（部分）
- [ ] **QuantMatMul** — 量化矩阵乘
- [ ] **Paged KV** — 分页 KV 缓存
- [ ] **PagedAttention** — 分页注意力 (简化版)
- [ ] **FlashAttention-like kernel** — 类 FlashAttention 实现

#### 3.4 深度集成
- [ ] `integrations/vllm_mock.py` — vLLM adapter mock
- [ ] `integrations/sglang_mock.py` — SGLang adapter mock

#### 3.5 Leaderboard 增强
- [ ] shape-specific best config 查询
- [ ] leaderboard 可视化 (CLI table)
- [ ] cross-operator 性能对比

**交付物**:
- 3 种搜索策略 (Bandit / REINFORCE / GRPO) 可选切换
- 10+ 算子族覆盖
- vLLM/SGLang mock adapter

---

### Phase 4: 生产硬化与生态 (Week 15-16+)

**目标**: 稳定化、性能优化、文档完善。

#### 4.1 生产硬化
- [ ] CUDA Extension 后置兜底编译
- [ ] 多 GPU 批量优化支持
- [ ] 性能回归检测自动化
- [ ] 错误恢复与断点续跑

#### 4.2 性能优化
- [ ] 编译缓存机制（相同 config 不重复编译）
- [ ] 并行 candidate evaluation
- [ ] profile 采样优化

#### 4.3 文档与示例
- [ ] README.md 完整文档
- [ ] 算子开发指南
- [ ] 添加新算子教程
- [ ] 最佳实践与调优技巧

#### 4.4 质量保障
- [ ] 完整单元测试套件 (覆盖率 > 80%)
- [ ] 集成测试覆盖端到端流程
- [ ] nightly benchmark 自动运行
- [ ] leaderboard 历史趋势追踪

**交付物**:
- 生产可用的算子自优化工具
- 完整文档与教程
- CI/CD 自动化测试与基准

---

## 算子优先级时间线

| 阶段 | 算子 | 优先级 | 预期状态 |
|------|------|--------|----------|
| Phase 1 | RMSNorm | P0 | 完整闭环 |
| Phase 1 | RoPE | P0 | 完整闭环 |
| Phase 1 | fused_bias_gelu | P0 | 完整闭环 |
| Phase 2 | SwiGLU | P1 | 完整闭环 |
| Phase 2 | quant_dequant | P1 | 完整闭环 |
| Phase 2 | LayerNorm | P1 | 完整闭环 |
| Phase 3 | KV Append | P2 | LLM decode hot path |
| Phase 3 | RoPE + KV Append | P2 | 融合优化 |
| Phase 3 | simple matmul epilogue | P2 | 矩阵乘后置 |
| Phase 3 | QuantMatMul | P3 | 量化 |
| Phase 3 | Paged KV | P3 | 分页缓存 |
| Phase 4 | PagedAttention | P3 | 高难度推理 |
| Phase 4 | FlashAttention-like | P3 | 高难度推理 |

---

## 搜索策略演进路线

```text
v0: Best-of-N / Grid Search        ← Phase 1 实现
    └── 穷举或随机采样参数空间

v1: Contextual Bandit              ← Phase 2 实现
    ├── UCB (Upper Confidence Bound)
    └── Thompson Sampling

v2: REINFORCE-lite                 ← Phase 3 实现
    └── 轻量策略梯度

v3: GRPO-lite                      ← Phase 3 实现
    └── 轻量组相对策略优化
```

---

## 明确不做 (Non-Goals)

- ❌ 完整 AIOS 操作系统
- ❌ 端到端推理框架
- ❌ 完整 Triton Inference Server
- ❌ 自研硬件/驱动
- ❌ 自研大模型训练
- ❌ 复杂 Web UI
- ❌ 多机多卡分布式训练
- ❌ 大规模 PPO 强化学习
- ❌ 任意算子自由生成（必须在 contract 内）

---

## 核心数据结构速查

### OpState
```python
@dataclass
class OpState:
    op_name: str
    B: int; T: int; D: int
    dtype: str; device: str
    gpu_name: str
    baseline_latency_us: float
    historical_best_config: dict | None
```

### OpAction
```python
@dataclass
class OpAction:
    template_id: str
    block_d: int; num_warps: int; num_stages: int
    vectorize: bool; fusion: bool
```

### CandidateResult
```python
@dataclass
class CandidateResult:
    compile_pass: bool; verify_pass: bool
    latency_us_p50: float; latency_us_p90: float; latency_us_p99: float
    speedup: float; variance: float; memory_peak_mb: float
    reward: float; promoted: bool
```

---

## CLI 命令速查

```bash
# 初始化算子
triton-agent init rmsnorm

# 优化某个 shape
triton-agent optimize rmsnorm --shape B=8,T=2048,D=4096 --dtype fp16 --device cuda

# 查看排行榜
triton-agent leaderboard rmsnorm

# 回放某次优化
triton-agent replay episodes/rmsnorm/000042

# 对比 baseline
triton-agent compare rmsnorm --baseline torch --variant best

# 导出最佳算子
triton-agent export rmsnorm --shape B=8,T=2048,D=4096 --out dist/rmsnorm_best/
```
