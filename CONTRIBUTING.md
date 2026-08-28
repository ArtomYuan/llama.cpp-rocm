# 贡献指南

感谢你考虑为 **llama.cpp-rocm** 做出贡献！🎉

本项目是面向 **AMD Strix Halo**（Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151）的融合引擎：官方 llama.cpp master + **ROCmFPX 量化格式** + **多模态视觉塔** 三合一。

欢迎各种形式的贡献：

- 🐛 报告 bug（构建失败、量化错误、推理异常）
- 💡 提出功能建议
- 📝 修正或补充文档
- 🔧 提交代码 PR

---

## 开发环境

本项目只支持 ROCm HIP 后端（无 Vulkan/CUDA），构建目标为 gfx1151。前置要求：

- ROCm 10.0.0（安装于 `/opt/rocm`）
- gcc / g++、cmake（≥ 3.14）、git

构建命令（与 README 一致）：

```bash
export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build -j $(nproc) --target llama-server llama-quantize llama-bench
```

产物在 `build/bin/`（llama-server / llama-quantize / llama-bench + 共享库）。

## 提交规范

提交信息使用 **Conventional Commits** 风格，格式：`<type>: <描述>`（描述用中文，简洁说明改了什么、为什么）。

| 类型 | 用途 | 示例 |
|:--|:--|:--|
| `feat:` | 新功能 | `feat: 支持 Q4_0_ROCMFP4_STRIX 量化类型` |
| `fix:` | 修复 bug | `fix: 修复 HIP 后端在 gfx1151 上的对齐崩溃` |
| `docs:` | 文档 | `docs: 更新量化格式列表` |
| `chore:` | 构建/工具/杂项 | `chore: 清理 CMake 缓存` |
| `refactor:` | 重构（不改变行为） | `refactor: 简化视觉塔投影器代码` |

要求：

- 一次提交只做一件事
- 提交信息描述行为变化，而不是复述 diff
- 保持与官方 llama.cpp 同步时使用上游原始提交信息，不要重写

## 分支与 PR 流程

1. 从最新的 `master` 切出功能分支：`feat/xxx`、`fix/xxx`、`docs/xxx`
2. 修改代码，本地构建验证（见「测试」）
3. 按提交规范提交
4. push 到你的 fork
5. 在 GitHub 上向本仓库 `master` 发起 Pull Request，描述改动内容并关联相关 issue（如有）

PR 提交前请自查：

- [ ] 本地构建通过（Release + gfx1151）
- [ ] 改动聚焦单一目的，没有夹带无关修改
- [ ] 新代码与周边风格一致
- [ ] 涉及量化类型/视觉模型时，已附实测或验证结果

## 代码风格

- C++ 遵循官方 llama.cpp 的代码风格：与周边代码保持一致，不引入新的模式
- 注释简洁：只解释「为什么」，不解释「做了什么」
- 不硬性换行、不用花哨的 Unicode 字符
- 优先复用现有基础设施，避免引入新的子系统

## 测试

本库**没有 CI**（官方 CI 矩阵已移除），本地构建验证是主要手段。改动提交前至少完成：

1. **构建通过**：上面的 cmake 构建命令无错误
2. **量化验证**（涉及量化代码时）：

   ```bash
   build/bin/llama-quantize --allow-requantize <源模型.gguf> <输出.gguf> Q4_0_ROCMFP4_FAST 32
   ```

3. **推理冒烟测试**：

   ```bash
   build/bin/llama-server -m <量化后模型.gguf> --port 8080
   # 发一条基本请求，确认能正常生成
   ```

## 沟通

- 提 issue 请使用对应模板（Bug 报告 / 功能请求）
- 本仓库默认语言为**中文**（与 README 一致）
- 建议先在 issue / 讨论中沟通想法，再投入大规模代码工作

再次感谢你的贡献！
