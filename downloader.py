import os, subprocess, requests, re, argparse
from collections import defaultdict

COLOR_FN = '\033[96m'
COLOR_OK = '\033[92m'
COLOR_DIR = '\033[93m'
COLOR_ERR = '\033[91m'
COLOR_RESET = '\033[0m'

parser = argparse.ArgumentParser()
parser.add_argument("--hf", default="")
parser.add_argument("--civitai", default="")
parser.add_argument("--req", action="store_true", help="Automatically run uv pip install on requirements.txt in cloned repos")
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
            m = re.search('filename="?([^"]+)"?', r.headers.get("Content-Disposition", ""))
            fn = m.group(1) if m else url.split("/")[-1]
            if "civitai" in url and "." not in fn: fn += ".safetensors"
            return fn, r.url
    except Exception as e:
        print(f"❌ Failed to access link: {e}")
        return None, None

if not DOWNLOAD_BATCHES:
    print("❌ DOWNLOAD_LIST not found. Declare a text (string) variable in the Colab cell before running %run.")
else:
    for folder, links in DOWNLOAD_BATCHES.items():
        if not links: continue
        
        os.makedirs(folder, exist_ok=True)
        
        for url in links:
            if "github.com" in url and not any(x in url for x in ["/releases/download/", "/raw/", "/blob/"]):
                repo_name = [p for p in url.split("/") if p][-1].replace(".git", "")
                repo_path = os.path.join(folder, repo_name)
                clone_success = False
                
                if os.path.exists(repo_path):
                    print(f"🍆 Cloning: {COLOR_FN}{repo_name}{COLOR_RESET} > {COLOR_DIR}{folder}{COLOR_RESET} [{COLOR_OK}SKIP{COLOR_RESET}]")
                    clone_success = True
                else:
                    print(f"🍆 Cloning: {COLOR_FN}{repo_name}{COLOR_RESET} > {COLOR_DIR}{folder}{COLOR_RESET}")
                    try:
                        p = subprocess.run(["git", "clone", url], cwd=folder, capture_output=True, text=True)
                        if p.returncode != 0:
                            err = p.stderr.strip().split('\n')[-1] if p.stderr else "Unknown error"
                            print(f"❌ Clone failed: {err}")
                        else:
                            clone_success = True
                    except Exception as e:
                        print(f"❌ System error occurred: {e}")
                
                # Check for requirements.txt if --req is passed and repo exists/cloned
                if args.req and clone_success:
                    print(f"📦 Installing requirements for {COLOR_FN}{repo_name}{COLOR_RESET}... ", end="", flush=True)
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

            auth = f"Bearer {HF_TOKEN}" if "huggingface" in url and HF_TOKEN else f"Bearer {CIVITAI_TOKEN}" if "civitai" in url and CIVITAI_TOKEN else ""
            h = {"User-Agent": "Mozilla/5.0"}
            if auth: h["Authorization"] = auth
            
            fn, furl = get_info(url, h)
            if not fn: continue

            print(f"⬇️ Downloading: {COLOR_FN}{fn}{COLOR_RESET} > {COLOR_DIR}{folder}{COLOR_RESET}")
            cmd = ["aria2c", "--console-log-level=error", "--summary-interval=1", "-c", "-x", "16", "-s", "16", "-k", "1M", "--header=User-Agent: Mozilla/5.0", "-d", folder, "-o", fn]
            
            if "huggingface.co" in furl and HF_TOKEN:
                cmd.append(f"--header=Authorization: Bearer {HF_TOKEN}")
            elif "civitai.com" in furl and CIVITAI_TOKEN:
                cmd.append(f"--header=Authorization: Bearer {CIVITAI_TOKEN}")
                
            cmd.append(furl)
            
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    if line.startswith("[#"): print(f"\r{line.strip():<20}", end="", flush=True)
                p.wait()
                
                if p.returncode == 0:
                    print(f" [{COLOR_OK}OK{COLOR_RESET}]")
                else:
                    print(f"❌ Download failed (Aria2 Error Code: {p.returncode})")
            except Exception as e:
                print(f"❌ System error occurred: {e}")
                
            print()
