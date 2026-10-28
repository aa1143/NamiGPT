"""让 chapters/ 下的脚本可以直接 `python ep12_xxx.py` 运行，并 import 仓库根目录的 gpt2 包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
