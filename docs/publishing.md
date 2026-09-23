# GitHub 同步与发布

实际仓库：[Fanzx42/MOP-Problems](https://github.com/Fanzx42/MOP-Problems)。
代码已托管到 GitHub；PyPI 发布和 GitHub Release 标签是另外的步骤。

## 获取代码

```sh
git clone https://github.com/Fanzx42/MOP-Problems.git
cd MOP-Problems
python -m pip install -e ".[dev,reference,plot]"
```

## 修改后的检查

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
python scripts/build_catalog.py --check
python examples/representative_suite.py --samples 2000 --seed 42
python -m build
python -m twine check dist/mop_problems-0.2.0-py3-none-any.whl dist/mop_problems-0.2.0.tar.gz
python scripts/check_wheel.py dist/mop_problems-0.2.0-py3-none-any.whl
```

## 同步到已有仓库

先检查当前分支和文件；下列命令适用于本仓库的 main 分支：

```sh
git status
git remote -v
git add .
git commit -m "Expand representative benchmarks and reproducible examples"
git push origin main
```

当前 origin 应为真实地址。如果仍显示旧占位符，可修正：

```sh
git remote set-url origin https://github.com/Fanzx42/MOP-Problems.git
```

如远程新增提交，先获取并合并或 rebase，处理冲突后再推送，不使用强制推送。
Git/GitHub 管理登录凭据，token 不写入源代码或远程 URL。

## Windows 目录所有权提示

如果在这个可信的本地项目中遇到 dubious ownership，只为该目录添加例外：

```sh
git config --global --add safe.directory C:/Users/89526/PycharmProjects/MOP-Problems
```

不需要将所有目录加入 safe.directory。

## 发布范围

0.2.0 有 43 个目录条目、47 个可运行问题，剩余 28 个目录条目待实现。
版本信息、CITATION.cff 和项目元数据已经包含实际 GitHub 地址。
正式创建 Release 时应说明具体问题和变体，并检查
[GitHub Actions](https://github.com/Fanzx42/MOP-Problems/actions) 的结果。

本仓库没有自动上传 PyPI 的流程。原始代码采用 MIT 许可证；
问题原论文和外部参考实现保持各自许可证。
