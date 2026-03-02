# Vulcan 中文介绍

Vulcan 是一个面向漏洞检测的深度学习框架与服务化工具集，覆盖数据收集、模型训练、验证评估与后端 API 能力。

## 项目能力

- 支持多种漏洞检测模型与数据集配置
- 提供训练、验证、基准测试与导出命令行入口
- 内置数据收集与数据集优化相关服务
- 可通过配置文件驱动实验流程

## 快速开始

建议先阅读英文主文档：

- 项目总览与安装：`README.md`
- 使用说明：`docs/usage.md`
- 架构说明：`docs/architecture.md`

## 常用命令

```bash
pip install -e .
vulcan-train --cfg configs/regvd_reveal.yaml
vulcan-val --cfg configs/regvd_reveal.yaml
```

## 说明

本文件仅提供中文简介。英文内容与完整技术细节以 `README.md` 和 `docs/` 文档为准。
