import subprocess
import sys
import os

# 必要なパッケージを自動インストール
def install_requirements():
    try:
        import streamlit
        import torch
        import torchvision
        import PIL
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Streamlit アプリを自動起動
def launch_streamlit():
    script_path = os.path.abspath("main.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])

if __name__ == "__main__":
    install_requirements()
    launch_streamlit()
