---
aliases: 
cssclasses: 
ReviewedDate: "[[Daily_Notes/15-10-24]]"
tags:
  - my/summary
  - state/process
  - python
  - env
child:
---

python 是一个高级语言，其生态十分发达。为了让我们能用 python 进行数据分析，我们需要用许多 package （别人写的代码）作为我们与数据交互的工具。

通常我们用于管理 python package 的工具为 conda 。以下我们将学习如何用 conda 管理一个适合中国宝宝体质的 python 环境

## Conda 安装
> ref:: [miniconda 官网命令行安装](https://docs.anaconda.com/miniconda/#quick-command-line-install)


1. [[美美的数据 PlayGround#^822521|打开命令行]]
2. 按照 [miniconda 官网命令行安装](https://docs.anaconda.com/miniconda/#quick-command-line-install) 的指令进行安装

- MacOS 下的操作
	```bash
	mkdir -p ~/miniconda3 # 创建安装目录
	curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh # 下载安装器
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 # 执行安装
	rm ~/miniconda3/miniconda.sh # 删掉安装器
	```
	- 激活环境
	
		```bash
		# 一次性
		source activate ~/miniconda3/bin/activate
		# 一劳永逸，看个人喜好
		conda init --all
		```
## 镜像配置
> ref:: [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

conda 进行包安装时，需要从外网下载。为此，我们需要把安装源替换成适合中国宝宝的镜像源。


> [!WARNING] 
> 这些镜像站会偶尔挂掉，因而在下载不 work 时，可关注以下链接 https://mirrors.cernet.edu.cn/list/anaconda




