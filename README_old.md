

### 1. train.py
- ✅ **修复 scheduler 调用时机**: 仅在优化器步骤后调用，而非每个累积步骤
- ✅ **预缓存参数列表**: 避免每次梯度裁剪时重新迭代
- ✅ **支持 persistent_workers**: 减少 DataLoader worker 启动开销
- ✅ **增强监控**: 添加 GPU 内存监控指标
- ✅ **non_blocking 数据传输**: 异步数据传输到 GPU

### 2. dataloader.py
- ✅ **异步预取机制**: 后台线程预取数据，隐藏IO延迟
- ✅ **批量奖励计算**: BatchedRewardCalculator 支持批量 GRP 推理
- ✅ **优化缓冲区管理**: 减少内存分配/释放开销
- ✅ **生产者-消费者模式**: ProducerConsumerDataLoader (可选)

### 3. engine.py
- ✅ **张量缓冲区预分配**: 减少推理时的内存分配
- ✅ **批量推理优化**: 高效填充预分配缓冲区
- ✅ **EfficientBatchInference**: 支持请求聚合的高级推理器

### 4. reward_calculator.py
- ✅ **BatchRewardCalculator**: 批量计算多游戏奖励
- ✅ **CachedRewardCalculator**: 带 LRU 缓存的计算器
- ✅ **工厂函数**: 方便创建不同模式的计算器

## 🔧 使用方法

### 基本使用
直接替换原有文件即可：
```bash
cp train.py walking/
cp dataloader.py walking/
cp engine.py walking/
cp reward_calculator.py walking/
```

### 分布式训练
```bash
# 单节点 4 GPU
torchrun --nproc_per_node=4 train.py

# 多节点
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=MASTER_IP train.py
```

### 模型管理
```bash
# 列出模型版本
python model_registry.py list

# A/B 对比测试
python model_registry.py compare v0001 v0002 --games 2000

# 提升到生产
python model_registry.py promote v0002
```

## ⚠️ 注意事项

1. **配置兼容性**: 确保 config.toml 中添加必要的新配置项
2. **依赖检查**: 分布式训练需要正确配置 NCCL
3. **渐进式采用**: 建议先单独测试每个优化模块
4. **性能验证**: 使用相同种子对比优化前后的训练曲线

---

*Jeff Dean 分布式系统设计原则*
*生成日期: 2025-12-01*


(walking) jiacheng@ags1:/root/dylan/icml2026/WALKING$ cargo build --release。。。

error: linking with cc failed: exit status: 1
  |
  = note:  "cc" "-m64" "<1 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/tmp/rustcddbgCM/liblibmimalloc_sys-875d0865e196ae20.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/libcompiler_builtins-*.rlib" "-Wl,-Bdynamic" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustcddbgCM/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/root/dylan/icml2026/WALKING/target/release/build/libmimalloc-sys-7a10940477920827/out" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/root/dylan/icml2026/WALKING/target/release/deps/stat-ffe5267afd0de45f" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-Wl,-O1" "-nodefaultlibs"
  = note: some arguments are omitted. use --verbose to show all linker arguments
  = note: rust-lld: error: undefined symbol: PyUnicode_FromStringAndSize

这个错误是因为链接时找不到 Python 库。这个项目使用了 PyO3（Rust 的 Python 绑定），需要正确配置 Python 环境。

conda activate walking

# 设置 PyO3 使用当前 Python
export PYO3_PYTHON=$(which python)

# 查看 Python 配置（用于调试）
python3-config --ldflags

# 重新编译（先清理）
cargo clean
cargo build --release

如果还是报同样的错误，试试这个：
bash# 设置链接标志
export RUSTFLAGS="-C link-args=-Wl,-undefined,dynamic_lookup"

这个项目有 pymod feature 默认启用，编译时需要链接 Python。试试这个：
unset RUSTFLAGS

# 设置 Python 链接
export PYO3_PYTHON=$(which python)

# 使用 maturin 构建（推荐用于 PyO3 项目）
pip install maturin

cd /root/dylan/icml2026/WALKING/libriichi
maturin build --release

python -c "import riichi; print('成功导入 riichi 模块')"

(walking) jiacheng@ags1:/root/dylan/icml2026/WALKING/libriichi$ pip install /root/dylan/icml2026/WALKING/target/wheels/libriichi-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
Processing /root/dylan/icml2026/WALKING/target/wheels/libriichi-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
Installing collected packages: libriichi
Successfully installed libriichi-0.1.0
(walking) jiacheng@ags1:/root/dylan/icml2026/WALKING/libriichi$ python -c "import riichi; print('成功导入 riichi 模块')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/jiacheng/anaconda3/envs/walking/lib/python3.12/site-packages/riichi/__init__.py", line 1, in <module>
    from .riichi import *
ImportError: dynamic module does not define module export function (PyInit_riichi)

这是因为构建时警告的问题——模块名不匹配,cat /root/dylan/icml2026/WALKING/libriichi/src/lib.rs
cd /root/dylan/icml2026/WALKING/libriichi

# 修改 Cargo.toml，把库名从 riichi 改成 libriichi
sed -i 's/name = "riichi"/name = "libriichi"/' Cargo.toml

# 确认修改
grep 'name = ' Cargo.toml

# 先卸载旧的
pip uninstall libriichi -y

# 重新构建
maturin build --release
pip install /root/dylan/icml2026/WALKING/target/wheels/libriichi-*.whl

# 测试导入
python -c "import libriichi; print('成功导入 libriichi 模块')"