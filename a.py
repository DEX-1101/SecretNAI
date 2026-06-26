import os
import sys
import subprocess
import urllib.request
import zipfile
import json

# === CONFIGURATION ===
MAIN_DIR = "/kaggle/working/x1101"
SHORTCUT_DIR = "/kaggle/working"
SD_REPO_URL = "https://github.com/Haoming02/sd-webui-forge-classic.git"
SD_BRANCH = "neo"
TARGET_PYTHON = "3.13" # Note: Change back to 3.10 if PyTorch fails to build wheels

# === TERMINAL COLORS ===
class C:
    GRN = '\033[92m'
    YLW = '\033[93m'
    RED = '\033[91m'
    CYN = '\033[96m'
    RST = '\033[0m'
    BOLD = '\033[1m'

def log(msg, level="info"):
    """Simplified, color-coded logging."""
    if level == "err":
        print(f"{C.RED}{C.BOLD}✗ [ERROR]{C.RST} {msg}")
    elif level == "warn":
        print(f"{C.YLW}⚠ [WARN]{C.RST} {msg}")
    elif level == "ok":
        print(f"{C.GRN}✓ [SUCCESS]{C.RST} {msg}")
    else:
        print(f"{C.CYN}• [INFO]{C.RST} {msg}")

def run_cmd(cmd, env=None, cwd=None):
    """Executes commands silently unless an error occurs to keep logs clean."""
    try:
        result = subprocess.run(cmd, env=env, cwd=cwd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {' '.join(cmd)}", "err")
        print(f"{C.RED}{e.stderr.strip()}{C.RST}")
        raise

def get_latest_python_url():
    """Dynamically fetches the latest targeted headless Linux build."""
    log(f"Resolving latest portable-python {TARGET_PYTHON} release...")
    api_url = "https://api.github.com/repos/bjia56/portable-python/releases?per_page=100"
    
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Kaggle-Setup-Script'})
        with urllib.request.urlopen(req) as response:
            releases = json.loads(response.read())
            
        for release in releases:
            if f"cpython-v{TARGET_PYTHON}" in release.get("tag_name", ""):
                for asset in release.get("assets", []):
                    name = asset.get("name", "")
                    if "headless" in name and "linux-x86_64" in name and name.endswith(".zip"):
                        return asset.get("browser_download_url")
        raise Exception(f"Could not find a valid {TARGET_PYTHON} headless release.")
    except Exception as e:
        raise Exception(f"Failed to fetch Python link: {e}")

def setup_environment():
    try:
        print(f"\n{C.BOLD}=== Initializing Stable Diffusion Setup ==={C.RST}\n")
        os.makedirs(MAIN_DIR, exist_ok=True)

        # 1. Dynamically Get and Extract Portable Python
        python_zip_path = f"{MAIN_DIR}/python-portable.zip"
        python_extract_dir = f"{MAIN_DIR}/python-portable-extracted"
        
        if not os.path.exists(python_extract_dir):
            python_url = get_latest_python_url()
            log("Downloading portable Python...")
            urllib.request.urlretrieve(python_url, python_zip_path)
            
            log("Extracting Python archive...")
            with zipfile.ZipFile(python_zip_path, 'r') as zip_ref:
                zip_ref.extractall(python_extract_dir)
            os.remove(python_zip_path)
            log("Python extracted successfully.", "ok")
        else:
            log("Portable Python already exists, skipping download.")
        
        # Resolve directories dynamically
        extracted_subdirs = [d for d in os.listdir(python_extract_dir) if os.path.isdir(f"{python_extract_dir}/{d}")]
        inner_python_dir = f"{python_extract_dir}/{extracted_subdirs[0]}" if extracted_subdirs else python_extract_dir
        
        python_bin_dir = f"{inner_python_dir}/bin"
        python_exe = f"{python_bin_dir}/python"
        
        log("Configuring execution permissions...")
        run_cmd(["chmod", "-R", "+x", python_bin_dir])
        
        # 2. Environment Isolation
        isolated_env = os.environ.copy()
        isolated_env["PATH"] = f"{python_bin_dir}:{isolated_env.get('PATH', '')}"
        isolated_env["PYTHONPATH"] = "" 
        
        # 3. Bootstrap Pip & Tools
        log("Bootstrapping core tools (pip, uv, aria2)...")
        run_cmd([python_exe, "-m", "ensurepip", "--upgrade"], env=isolated_env)
        run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip", "-q"], env=isolated_env)
        run_cmd([python_exe, "-m", "pip", "install", "uv", "aria2", "-q"], env=isolated_env)
        log("Core tools installed.", "ok")
        
        # 4. Clone Repo
        sd_dir = f"{MAIN_DIR}/sd-webui-forge-classic"
        if not os.path.exists(sd_dir):
            log(f"Cloning WebUI (Branch: {SD_BRANCH})...")
            run_cmd(["git", "clone", "-q", "-b", SD_BRANCH, SD_REPO_URL, sd_dir])
            log("Repository cloned successfully.", "ok")
        else:
            log("WebUI repository already exists, skipping clone.", "warn")

        # 5. Create Execution Shortcut
        log("Generating shortened execution script...")
        shortcut_path = f"{SHORTCUT_DIR}/run.sh"
        with open(shortcut_path, "w") as f:
            f.write(f'#!/bin/bash\n')
            f.write(f'export PATH="{python_bin_dir}:$PATH"\n')
            f.write(f'export PYTHONPATH=""\n')
            f.write(f'{python_exe} {sd_dir}/launch.py "$@"\n')
        os.chmod(shortcut_path, 0o755)
        log(f"Shortcut created at: {shortcut_path}", "ok")
            
        # 6. Verification
        print(f"\n{C.BOLD}=== System Verification ==={C.RST}")
        log(f"Python: {run_cmd([python_exe, '--version'], env=isolated_env).strip()}")
        log(f"UV: {run_cmd([f'{python_bin_dir}/uv', '--version'], env=isolated_env).strip()}")
        log(f"Aria2c: {run_cmd([f'{python_bin_dir}/aria2c', '--version'], env=isolated_env).splitlines()[0]}")
        
        print(f"\n{C.GRN}{C.BOLD}Setup Complete!{C.RST}")
        print(f"{C.CYN}To start the WebUI with arguments, run this in a new cell:{C.RST}")
        print(f"{C.BOLD}!./run.sh --share --api{C.RST}\n")
        
    except Exception as e:
        log(f"Script interrupted: {e}", "err")
        sys.exit(1)

if __name__ == "__main__":
    setup_environment()
