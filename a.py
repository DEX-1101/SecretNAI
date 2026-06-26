import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import tarfile

# === CONFIGURATION ===
MAIN_DIR = "/kaggle/working/x1101"
SD_REPO_URL = "https://github.com/Haoming02/sd-webui-forge-classic.git"
SD_BRANCH = "neo"

# Updated to Python 3.13.12 based on bjia56's naming convention
# Note: If this exact build tag does not exist yet on the repo, this will throw a 404 HTTP Error.
PYTHON_ZIP_URL = "https://github.com/bjia56/portable-python/releases/download/cpython-v3.13.12-build.0/python-headless-3.13.12-linux-x86_64.zip"

# Static build of aria2 for isolation
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
        
        # Specific debugging message for 3.13 dependency failures
        if "pip" in cmd or "uv" in cmd:
            log("DEBUG TIP: This failure is likely because Python 3.13 lacks pre-compiled wheels for PyTorch or xformers. Consider rolling back to 3.10 if this persists.", is_error=True)
            
        raise

def setup_environment():
    try:
        log("Starting Isolated SD Setup (Python 3.13.12 Experimental)...")
        os.makedirs(MAIN_DIR, exist_ok=True)

        # 1. Download and Extract Portable Python
        python_zip_path = os.path.join(MAIN_DIR, "python-portable.zip")
        python_extract_dir = os.path.join(MAIN_DIR, "python-portable-extracted")
        
        if not os.path.exists(python_extract_dir):
            log(f"Downloading portable Python from {PYTHON_ZIP_URL}...")
            urllib.request.urlretrieve(PYTHON_ZIP_URL, python_zip_path)
            
            log("Extracting Python archive...")
            with zipfile.ZipFile(python_zip_path, 'r') as zip_ref:
                zip_ref.extractall(python_extract_dir)
            os.remove(python_zip_path)
        else:
            log("Portable Python already extracted.")
        
        # Resolve the inner bin directory path dynamically
        extracted_subdirs = os.listdir(python_extract_dir)
        inner_python_dir = os.path.join(python_extract_dir, extracted_subdirs[0])
        python_bin_dir = os.path.join(inner_python_dir, "bin")
        python_exe = os.path.join(python_bin_dir, "python")
        
        # 2. Strict Environment Isolation
        # Overwrite PATH to prioritize the portable bin and clear PYTHONPATH
        isolated_env = os.environ.copy()
        isolated_env["PATH"] = f"{python_bin_dir}:{isolated_env.get('PATH', '')}"
        isolated_env["PYTHONPATH"] = "" 
        
        # 3. Bootstrap Pip and Install UV
        log("Bootstrapping pip and installing uv for fast dependency resolution...")
        run_cmd([python_exe, "-m", "ensurepip", "--upgrade"], env=isolated_env)
        run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip"], env=isolated_env)
        run_cmd([python_exe, "-m", "pip", "install", "uv"], env=isolated_env)
        
        # 4. Download and configure aria2c natively in the portable bin
        aria2_exe = os.path.join(python_bin_dir, "aria2c")
        if not os.path.exists(aria2_exe):
            log("Downloading static aria2c binary for isolated model downloads...")
            aria2_tar_path = os.path.join(MAIN_DIR, "aria2.tar.bz2")
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
        sd_dir = os.path.join(MAIN_DIR, "sd-webui-forge-classic")
        if not os.path.exists(sd_dir):
            log(f"Cloning SD repository (Branch: {SD_BRANCH})...")
            run_cmd(["git", "clone", "-b", SD_BRANCH, SD_REPO_URL, sd_dir])
        else:
            log("SD repository already exists. Skipping clone.")
            
        # 6. Final Diagnostics and Verification
        log("=== Verification Check ===")
        log(f"Python: {run_cmd([python_exe, '--version'], env=isolated_env).strip()}")
        uv_exe = os.path.join(python_bin_dir, "uv")
        log(f"UV: {run_cmd([uv_exe, '--version'], env=isolated_env).strip()}")
        log(f"Aria2c: {run_cmd([aria2_exe, '--version'], env=isolated_env).splitlines()[0]}")
        
        log("========================================")
        log("Setup complete! Your isolated Python 3.13 environment is ready.")
        log(f"Working Directory: {MAIN_DIR}")
        log(f"To run webui, you must execute it using: {python_exe}")
        
    except urllib.error.HTTPError as e:
        log(f"CRITICAL FAILURE: Could not download from URL. Python 3.13.12 might not exist on the portable-python repo yet. Error: {e}", is_error=True)
        sys.exit(1)
    except Exception as e:
        log(f"CRITICAL FAILURE: An unhandled exception interrupted the script: {e}", is_error=True)
        sys.exit(1)

if __name__ == "__main__":
    setup_environment()