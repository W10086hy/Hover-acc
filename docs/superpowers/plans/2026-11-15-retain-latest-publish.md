# 最新训练／仿真整理与发布

**Goal:** 按用户 2026-11-15 新授权清理中间候选，保留最新安装误差 P3、冻结 P2、完整配对证据和历史必要依赖，上传 W10086hy/Hover-acc。

**Scope:** 不训练、不重新仿真、不改变模型字节。逐文件保留／删除清单见 `cleanup/retention_p3_installation_20261115.json`。该新授权替代先前对未选调试结果的临时全量保留；受保护的历史最终模型继续保留。

- [x] 核对当前结果、模型索引及远端；远端 Hover-acc/main 与本地原 HEAD 文件树相同，提交历史不同。
- [x] 列出 327 个过期候选／调试文件，合并原始统计，核对 139 个必要文件受保护。
- [x] 按绝对路径／哈希核对清单执行本目录清理，拒绝链接和越界；327 文件、70,356,393 字节。
- [x] 更新整理说明、模型／状态；最新及前一最终配对全率重审、模型／Adam、媒体核验，全工程 234 tests / OK（1 skipped）。

提交前门槛为 `cleanup/verification.json` 的严格核验通过。提交后由本次聊天执行向 Hover-acc/main 的普通快进推送，并核对远端提交和文件树；远端提交 SHA 在聊天交付结果记录，不向旧 Hover-BPTT 上传。

**Files:** 新增清理清单与合并统计，修改 AGENTS／README／P3 进展与状态、cleanup/README、工程清单。代码及最新实验原始证据保持字节一致。删除范围只来自明确清单，空目录单独删除且不递归。

**Verification:** `P3.report_installation` 全率重算；`P3.report_body_calibration` 核对保留历史最终配对；模型、原始数据、媒体及冻结源码哈希；`cleanup.package manifest`；`verify_project.py --strict-sources --write-report`；Git diff/索引完整性；远端 main SHA／tree 一致。GitHub 插件连接只读，批量二进制采用本地 Git。
