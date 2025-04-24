---
title: "使用Anaconda搭建虚拟环境"
description: "Building a virtual environment using Anaconda"
pubDate: "Apr 20 2025"
image: /image/head.jpg
categories:
  - tech
tags:
  - Machine Learning

---

# 搭建虚拟环境和使用anaconda

搭建虚拟环境有许多优点

- 避免依赖冲突
- 方便项目迁移与共享
- ~~便于删库跑路~~（笑死）

下面是在windows环境中,使用`Anaconda`搭建虚拟环境的步骤

## 1.1创建虚拟环境

输入此条指令创建conda虚拟环境,可以自己更改名称和python版本

`conda create -n 自己输入名称 python=版本号`

如下所示,我示例创建名为 NAME 的环境

`conda create -n NAME python=3.8 `

## 1.2 激活虚拟环境

输入这一行代码激活刚才创建的环境,名称要与之前创建时设置的名称相应

`conda activate 名称`

运行后,左边括号里变成你虚拟变量的名称,即为进入此环境了,之后你输入的所有命令都是在此环境中执行的

## 1.3 使用虚拟环境打开python文件

### 1.3.1 打开对应文件位置

方法一：使用cd指令

即`cd 目录位置 `

方法二：在文件夹的目录中输入cmd，然后回车

> ==注意:==
> 跨盘符直接输入`D:`

### 1.3.2 打开虚拟环境，打开文件

- 打开虚拟环境
  `conda activate 虚拟环境名称`

- 打开文件
  输入`python 文件名称`，运行python文件了,注意python文件是.py结尾的后缀.

## 2 在虚拟环境中安装库以及配置镜像源

### 2.1 在虚拟环境中安装库

在虚拟环境中输入`pip install 库名`即可在此环境中安装库

或者有requirement文件，输入`pip install -r requirement.txt`

### 2.2 配置镜像源

如果用pip安装库的过程中，下载速度特别慢停滞不动，或者出现红色报错，可以配置镜像源，能使下载速度快很多.

选择一种直接复制整条代码ctrl+V到cmd中即可，然后再运行pip安装指令速度就很快了

#### 清华镜像源

```
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
```

#### 中科大镜像源

```
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/
```

### 2.3 镜像源的其他指令

1. 查看已经添加的镜像
   `conda config --show channels`

2. 删除所有镜像源
   输入此条指令即可删除所有镜像源，在有时候镜像源出现问题报错时可删除掉，或者换成其他源.
   ` conda config --remove-key channels`

## 3 conda的其他使用指令

### 3.1 创建环境

`conda create --name your_env_name`

### 3.2 创建包含某些包的环境

`conda create --name your_env_name numpy scipy`这是包含numpy的环境

### 3.3 创建指定python版本下包含某些包的环境

`conda create --name your_env_name python=3.5 numpy scipy`

### 3.4 列举当前所有环境

`conda env list`

### 3.5 删除环境

`conda remove -n xxxxx(名字) --all`

## 4 使用jupyter notebook 打开虚拟环境

- 错误示范：在annaconda的虚拟环境中输入`jupyter notebook`,并没有进入设置好的虚拟环境中

- 正确方法：在虚拟环境中加入外部库，使虚拟环境导入notebook中
  1. 在虚拟环境中，输入` conda install ipykernel`
  2. 输入`python -m ipykernel install --user --name your_env_name --display-name your_env_displayname`
     其中gh1为虚拟环境的名称，display-name代表显示的是什么

## 5 在虚拟环境中的其他安装问题

### 5.1 numpy库和TensorFlow库版本不一定兼容

TensorFlow是图像识别需要的库,numpy是机器学习需要的库

检查现有库的版本 `pip show tensorflow`

卸载原有的TensorFlow库 `pip uninstall tensorflow`

查表寻找tensorflow和numpy版本对应表

安装对应版本的TensorFlow库 `pip install tensorflow==<version>`

### 5.2 anaconda无法正确cd

原因：conda Prompt中cd只能在同一盘符下切换路径，不能进行切换盘符

解决办法1:先切换盘符,再cd

例如想要进入D盘

```
D:
cd D:\code
```

解决办法2:直接跨盘符

```
cd /d D:\xxx\xxx
```
