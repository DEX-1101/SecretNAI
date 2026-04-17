import os, subprocess, requests, re, argparse, shutil, zipfile
from collections import defaultdict

COLOR_FN = '\033[96m'
COLOR_OK = '\033[92m'
COLOR_DIR = '\033[93m'
COLOR_ERR = '\033[91m'
COLOR_RESET = '\033[0m'

parser = argparse.ArgumentParser()
parser.add_argument("--hf", default="", help="HuggingFace API token")
parser.add_argument("--civitai", default="", help="Civitai API token")
parser.add_argument("--req", action="store_true", help="Install requirements.txt in cloned repos")
parser.add_argument("--zip", default="", help="Password for extracting ZIP files")
parser.add_argument("--upload_to", default="", help="Upload folder to HF: username/repo::[remote_folder]::local_folder or username/repo::local_folder")
args, _ = parser.parse_known_args()

try:
    from IPython import get_ipython
    user_ns = get_ipython().user_ns
except:
    user_ns = globals()

HF_TOKEN = args.hf
CIVITAI_TOKEN = args.civitai

VAR_REGEX = re.compile(r'\{([^}]+)\}')

def resolve_vars(text):
    return VAR_REGEX.sub(lambda m: str(user_ns.get(m.group(1), m.group(0))), text)

DOWNLOAD_BATCHES = defaultdict(list)
current_dir = "downloads"
raw_list = user_ns.get('DOWNLOAD_LIST', '')

if raw_list:
    for line in raw_list.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        line = resolve_vars(line)

        if line.startswith('http'):
            DOWNLOAD_BATCHES[current_dir].append(line)
        else:
            current_dir = line
else:
    raw_batches = user_ns.get('DOWNLOAD_BATCHES', {})
    for k, v in raw_batches.items():
        k_res = resolve_vars(k)
        DOWNLOAD_BATCHES[k_res] = [resolve_vars(url) for url in v]

def get_info(url, headers):
    try:
        with requests.get(url, headers=headers, stream=True, timeout=15) as r:
            r.raise_for_status()
            
            if "/login" in r.url:
                print(f"❌ Authentication failed: Redirected to login page. Please check your Civitai API token.")
                return None, None
                
            m = re.search('filename="?([^";]+)"?', r.headers.get("Content-Disposition", ""))
            fn = m.group(1) if m else r.url.split("/")[-1].split("?")[0]
            if "civitai" in url and "." not in fn: fn += ".safetensors"
            return fn, r.url
    except Exception as e:
        print(f"❌ Failed to access link: {e}")
        return None, None

def extract_zip(file_path, folder, pwd):
    if not file_path.lower().endswith('.zip'): return
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            if pwd:
                z.setpassword(pwd.encode('utf-8'))
            infos = z.infolist()
            total = len(infos)
            if total == 0:
                return
            
            ext_counts = defaultdict(int)
            skipped = 0
            
            for i, info in enumerate(infos, 1):
                fname = info.filename if len(info.filename) < 60 else info.filename[:57] + '...'
                print(f"\rExtracting [{i}/{total}]: {COLOR_FN}{fname}{COLOR_RESET}\033[K", end="", flush=True)
                
                target_path = os.path.join(folder, info.filename)
                
                if not info.is_dir():
                    ext = os.path.splitext(info.filename)[1].lower() or 'no-ext'
                    ext_counts[ext] += 1
                    
                    if os.path.exists(target_path):
                        skipped += 1
                        continue
                
                z.extract(info, folder)
                
            count_strs = [f"{COLOR_FN}{ext.lstrip('.')}{COLOR_RESET} ({COLOR_OK}{count}{COLOR_RESET})" for ext, count in ext_counts.items()]
            ext_summary = ", ".join(count_strs)
            if skipped > 0:
                ext_summary += f" | {COLOR_ERR}{skipped} skipped{COLOR_RESET}"
                
            print(f"\r🗡️ Extracted [{total}/{total}]: {COLOR_OK}Done{COLOR_RESET} [ {ext_summary} ]\033[K")
    except Exception as e:
        print(f"\n❌ Error extracting {os.path.basename(file_path)}: {e}")

def run_upload():
    parts = args.upload_to.split("::")
    if len(parts) == 2:
        repo_id, local_folder = parts
        remote_folder = ""
    elif len(parts) == 3:
        repo_id, remote_folder, local_folder = parts
    else:
        print(f"❌ Invalid format. Use: {COLOR_FN}username/repo::[remote_folder]::local_folder{COLOR_RESET} or {COLOR_FN}username/repo::local_folder{COLOR_RESET}")
        return
        
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(f"⚙️ Installing {COLOR_OK}huggingface_hub{COLOR_RESET}... ", end="", flush=True)
        subprocess.run("pip install -q huggingface_hub", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from huggingface_hub import HfApi
        print("\r\033[K", end="")
        
    if not HF_TOKEN:
        print(f"❌ HF Token is required for uploading! Pass it via {COLOR_FN}--hf{COLOR_RESET}")
        return
    if not os.path.exists(local_folder):
        print(f"❌ Local folder {COLOR_DIR}{local_folder}{COLOR_RESET} does not exist!")
        return

    api = HfApi(token=HF_TOKEN)
    
    try:
        api.model_info(repo_id)
    except:
        print(f"⚙️ Creating private repo {COLOR_FN}{repo_id}{COLOR_RESET}... ", end="", flush=True)
        try:
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            print(f"[{COLOR_OK}OK{COLOR_RESET}]")
        except Exception as e:
            print(f"\n❌ Failed to create repo: {e}")
            return

    print(f"💦 Uploading to {COLOR_FN}{repo_id}/{remote_folder if remote_folder else 'root'}{COLOR_RESET}...")
    
    try:
        api.upload_folder(
            folder_path=local_folder,
            path_in_repo=remote_folder,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ Upload {COLOR_OK}Done{COLOR_RESET}!")
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")

if args.upload_to:
    run_upload()
elif not DOWNLOAD_BATCHES:
    print("❌ DOWNLOAD_LIST not found. Declare a text (string) variable in the Colab cell before running %run.")
else:
    if not shutil.which("aria2c"):
        print(f"⚙️ Installing {COLOR_OK}aria2c{COLOR_RESET}... ", end="", flush=True)
        try:
            subprocess.run("apt-get install -y -qq aria2", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if shutil.which("aria2c"): 
                print("\r\033[K", end="", flush=True)
        except: pass

    for folder, links in DOWNLOAD_BATCHES.items():
        if not links: continue
        
        os.makedirs(folder, exist_ok=True)
        
        for url in links:
            if "github.com" in url and not any(x in url for x in ["/releases/download/", "/raw/", "/blob/"]):
                repo_name = [p for p in url.split("/") if p][-1].replace(".git", "")
                repo_path = os.path.join(folder, repo_name)
                clone_success = False
                
                if os.path.exists(repo_path):
                    print(f"💀 {COLOR_FN}{repo_name}{COLOR_RESET} already exists in {COLOR_DIR}{folder}{COLOR_RESET}")
                    clone_success = True
                else:
                    print(f"🍆 Cloning: {COLOR_FN}{repo_name}{COLOR_RESET}", end="", flush=True)
                    try:
                        p = subprocess.run(["git", "clone", url], cwd=folder, capture_output=True, text=True)
                        if p.returncode != 0:
                            err = p.stderr.strip().split('\n')[-1] if p.stderr else "Unknown error"
                            print(f"\n❌ Clone failed: {err}")
                        else:
                            print(f" [{COLOR_OK}OK{COLOR_RESET}] | Saved to : {COLOR_DIR}{folder}{COLOR_RESET}")
                            clone_success = True
                    except Exception as e:
                        print(f"\n❌ System error occurred: {e}")
                
                if args.req and clone_success:
                    print(f"Installing requirements for {COLOR_FN}{repo_name}{COLOR_RESET}... ", end="", flush=True)
                    req_file = os.path.join(repo_path, "requirements.txt")
                    
                    if not os.path.exists(req_file):
                        print(f"[{COLOR_ERR}No requirements.txt found{COLOR_RESET}]")
                    elif os.path.getsize(req_file) == 0:
                        print(f"[{COLOR_ERR}requirements.txt is empty{COLOR_RESET}]")
                    else:
                        try:
                            req_p = subprocess.run(["uv", "pip", "install", "--system", "-r", "requirements.txt"], cwd=repo_path, capture_output=True, text=True)
                            if req_p.returncode == 0:
                                print(f"[{COLOR_OK}OK{COLOR_RESET}]")
                            else:
                                err_lines = [line.strip() for line in req_p.stderr.split('\n') if line.strip()]
                                err_msg = err_lines[-1] if err_lines else "Unknown install error"
                                print(f"[{COLOR_ERR}Failed: {err_msg}{COLOR_RESET}]")
                        except Exception as e:
                            print(f"[{COLOR_ERR}System error: {e}{COLOR_RESET}]")
                
                print()
                continue

            if "civitai" in url and CIVITAI_TOKEN and "token=" not in url:
                url += f"{'&' if '?' in url else '?'}token={CIVITAI_TOKEN}"

            auth = f"Bearer {HF_TOKEN}" if "huggingface" in url and HF_TOKEN else ""
            h = {"User-Agent": "Mozilla/5.0"}
            if auth: h["Authorization"] = auth
            
            fn, furl = get_info(url, h)
            if not fn: continue

            file_path = os.path.join(folder, fn)
            # Skip if the file exists and there's no active .aria2 active download file
            if os.path.exists(file_path) and not os.path.exists(file_path + ".aria2"):
                print(f"💀 {COLOR_FN}{fn}{COLOR_RESET} already exists in {COLOR_DIR}{folder}{COLOR_RESET}")
                if fn.lower().endswith('.zip'):
                    extract_zip(file_path, folder, args.zip)
                print()
                continue

            print(f"⬇️ Downloading: {COLOR_FN}{fn}{COLOR_RESET}")
            cmd = ["aria2c", "--console-log-level=error", "--summary-interval=1", "-c", "-x", "16", "-s", "16", "-k", "1M", "--header=User-Agent: Mozilla/5.0", "-d", folder, "-o", fn]
            
            if furl == url:
                if "huggingface.co" in furl and HF_TOKEN:
                    cmd.append(f"--header=Authorization: Bearer {HF_TOKEN}")
                
            cmd.append(furl)
            
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    if line.startswith("[#"): print(f"\r{line.strip():<20}", end="", flush=True)
                p.wait()
                
                if p.returncode == 0:
                    print(f" [{COLOR_OK}OK{COLOR_RESET}] | Saved to : {COLOR_DIR}{folder}{COLOR_RESET}")
                    if fn.lower().endswith('.zip'):
                        extract_zip(file_path, folder, args.zip)
                else:
                    print(f"❌ Download failed (Aria2 Error Code: {p.returncode})")
            except Exception as e:
                print(f"❌ System error occurred: {e}")
                
            print()
