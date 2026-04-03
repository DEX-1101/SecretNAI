import os, subprocess, requests, re, argparse

# ANSI color codes for terminal output
COLOR_FN = '\033[96m'  # Cyan for filename
COLOR_OK = '\033[92m'  # Green for OK
COLOR_DIR = '\033[93m' # Yellow for directory
COLOR_RESET = '\033[0m' # Reset color

parser = argparse.ArgumentParser()
parser.add_argument("--hf", default="")
parser.add_argument("--civitai", default="")
args, _ = parser.parse_known_args()

try:
    from IPython import get_ipython
    user_ns = get_ipython().user_ns
except:
    user_ns = globals()

HF_TOKEN = args.hf
CIVITAI_TOKEN = args.civitai

DOWNLOAD_BATCHES = {}
current_dir = "downloads"

for line in user_ns.get('DOWNLOAD_LIST', '').splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue

    if line.startswith('http'):
        os.makedirs(current_dir, exist_ok=True)
        DOWNLOAD_BATCHES.setdefault(current_dir, []).append(line)
    else:
        current_dir = line

if not DOWNLOAD_BATCHES:
    DOWNLOAD_BATCHES = user_ns.get('DOWNLOAD_BATCHES', {})

def get_info(url, headers):
    try:
        with requests.get(url, headers=headers, stream=True, timeout=15) as r:
            r.raise_for_status()
            m = re.search('filename="?([^"]+)"?', r.headers.get("Content-Disposition", ""))
            fn = m.group(1) if m else url.split("/")[-1]
            if "civitai" in url and "." not in fn: fn += ".safetensors"
            return fn, r.url
    except Exception as e:
        print(f"\n❌ Failed to access link: {e}")
        return None, None

if not DOWNLOAD_BATCHES:
    print("❌ DOWNLOAD_LIST not found. Declare a text (string) variable in the Colab cell before running %run.")
else:
    for folder, links in DOWNLOAD_BATCHES.items():
        if not links: continue
        os.makedirs(folder, exist_ok=True)
        
        for url in links:
            auth = f"Bearer {HF_TOKEN}" if "huggingface" in url and HF_TOKEN else f"Bearer {CIVITAI_TOKEN}" if "civitai" in url and CIVITAI_TOKEN else ""
            h = {"User-Agent": "Mozilla/5.0"}
            if auth: h["Authorization"] = auth
            
            fn, furl = get_info(url, h)
            if not fn: continue

            print(f"\n⬇️ Downloading: {COLOR_FN}{fn}{COLOR_RESET} > {COLOR_DIR}{folder}{COLOR_RESET}")
            cmd = ["aria2c", "--console-log-level=error", "--summary-interval=1", "-c", "-x", "16", "-s", "16", "-k", "1M", "--header=User-Agent: Mozilla/5.0", "-d", folder, "-o", fn]
            if auth: cmd.append(f"--header=Authorization: {auth}")
            cmd.append(furl)
            
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    if line.startswith("[#"): print(f"\r{line.strip():<20}", end="", flush=True)
                p.wait()
                
                if p.returncode == 0:
                    print(f" [{COLOR_OK}OK{COLOR_RESET}]")
                else:
                    print(f"\n❌ Download failed (Aria2 Error Code: {p.returncode})\n")
            except Exception as e:
                print(f"\n❌ System error occurred: {e}\n")
