import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import tarfile
import json

# === CONFIGURATION ===
MAIN_DIR = "/kaggle/working/x1101"
SD_REPO_URL = "https://github.com/Haoming02/sd-webui-forge-classic.git"
SD_BRANCH = "neo"
ARIA2_STATIC_URL = "https://github.com/q3aql/aria2-static-builds/releases/download/v1.36.0/aria2-1.36.0-linux-gnu-64bit-build1.tar.bz2"

def log(msg, is_error=False):
    """Custom logger for clear debugging output."""
    prefix = "[ERROR] " if is_error else "[INFO] "
    print(f"{prefix}{msg}")

def run_cmd(cmd, env=None, cwd=None):
    """Executes system commands and captures robust debugging information."""
    try:
        log(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, cwd=cwd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {' '.join(cmd)}", is_error=True)
        log(f"Error Output: {e.stderr.strip()}", is_error=True)
        raise

def get_latest_python_url():
    """Dynamically fetches the latest 3.10 headless Linux build to prevent dead links."""
    log("Fetching the latest portable-python 3.10 release from GitHub API...")
    api_url = "https://api.github.com/repos/bjia56/portable-python/releases?per_page=100"
    
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Kaggle-Setup-Script'})
        with urllib.request.urlopen(req) as response:
            releases = json.loads(response.read())
            
        for release in releases:
            if "cpython-v3.10" in release.get("tag_name", ""):
                for asset in release.get("assets", []):
                    name = asset.get("name", "")
                    if "headless" in name and "linux-x86_64" in name and name.endswith(".zip"):
                        download_url = asset.get("browser_download_url")
                        log(f"Successfully resolved dynamic link: {download_url}")
                        return download_url
        raise Exception("Could not find a valid 3.10 headless release in the API response.")
    except Exception as e:
        raise Exception(f"Failed to fetch dynamic Python link: {e}")

def setup_environment():
    try:
        log("Starting Isolated SD Setup...")
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
        else:
            log("Portable Python already extracted.")
        
        # Resolve the inner bin directory path dynamically
        extracted_subdirs = [d for d in os.listdir(python_extract_dir) if os.path.isdir(f"{python_extract_dir}/{d}")]
        inner_python_dir = f"{python_extract_dir}/{extracted_subdirs[0]}" if extracted_subdirs else python_extract_dir
        
        python_bin_dir = f"{inner_python_dir}/bin"
        python_exe = f"{python_bin_dir}/python"
        
        # FIX: Grant execute permissions to all binaries in the portable bin folder
        log("Restoring execute permissions for portable binaries...")
        run_cmd(["chmod", "-R", "+x", python_bin_dir])
        
        # 2. Strict Environment Isolation
        isolated_env = os.environ.copy()
        isolated_env["PATH"] = f"{python_bin_dir}:{isolated_env.get('PATH', '')}"
        isolated_env["PYTHONPATH"] = "" 
        
        # 3. Bootstrap Pip and Install UV
        log("Bootstrapping pip and installing uv for fast dependency resolution...")
        run_cmd([python_exe, "-m", "ensurepip", "--upgrade"], env=isolated_env)
        run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip"], env=isolated_env)
        run_cmd([python_exe, "-m", "pip", "install", "uv"], env=isolated_env)
        
        # 4. Download and configure aria2c natively in the portable bin
        aria2_exe = f"{python_bin_dir}/aria2c"
        if not os.path.exists(aria2_exe):
            log("Downloading static aria2c binary for isolated model downloads...")
            aria2_tar_path = f"{MAIN_DIR}/aria2.tar.bz2"
            urllib.request.urlretrieve(ARIA2_STATIC_URL, aria2_tar_path)
            
            log("Extracting aria2c...")
            with tarfile.open(aria2_tar_path, "r:bz2") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("aria2c"):
                        member.name = os.path.basename(member.name) 
                        tar.extract(member, path=python_bin_dir)
                        break
            os.remove(aria2_tar_path)
            run_cmd(["chmod", "+x", aria2_exe])
            log("aria2c secured in portable bin directory.")
        
        # 5. Clone Stable Diffusion Repository
        sd_dir = f"{MAIN_DIR}/sd-webui-forge-classic"
        if not os.path.exists(sd_dir):
            log(f"Cloning SD repository (Branch: {SD_BRANCH})...")
            run_cmd(["git", "clone", "-b", SD_BRANCH, SD_REPO_URL, sd_dir])
        else:
            log("SD repository already exists. Skipping clone.")
            
        # 6. Final Diagnostics and Verification
        log("=== Verification Check ===")
        log(f"Python: {run_cmd([python_exe, '--version'], env=isolated_env).strip()}")
        uv_exe = f"{python_bin_dir}/uv"
        log(f"UV: {run_cmd([uv_exe, '--version'], env=isolated_env).strip()}")
        log(f"Aria2c: {run_cmd([aria2_exe, '--version'], env=isolated_env).splitlines()[0]}")
        
        log("========================================")
        log("Setup complete! Your isolated environment is ready.")
        log(f"Working Directory: {MAIN_DIR}")
        log(f"To run webui, execute: {python_exe} {sd_dir}/launch.py")
        
    except Exception as e:
        log(f"CRITICAL FAILURE: An unhandled exception interrupted the script: {e}", is_error=True)
        sys.exit(1)

if __name__ == "__main__":
    setup_environment()
