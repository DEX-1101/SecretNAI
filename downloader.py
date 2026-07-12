import os, subprocess, requests, re, argparse, shutil, zipfile
from collections import defaultdict

COLOR_FN = '\033[38;5;231m' # 256-color Pure White (Bypasses Kaggle grey override)
COLOR_OK = '\033[95m' # Magenta
COLOR_DIR = '\033[93m'
COLOR_ERR = '\033[91m'
COLOR_RESET = '\033[0m'
COLOR_WARN = '\033[93m' # Yellow
COLOR_UNIT = '\033[95m' # Added color for units
COLOR_SUCCESS = '\033[92m' # Green

# Robust line wipe to prevent leftover characters in Kaggle
CLEAR = '\r' + ' ' * 150 + '\r'

parser = argparse.ArgumentParser()
parser.add_argument("--hf", default="", help="HuggingFace API token(s), separated by ::")
parser.add_argument("--civitai", default="", help="Civitai API token(s), separated by ::")
parser.add_argument("--req", action="store_true", help="Install requirements.txt in cloned repos")
parser.add_argument("--zip", default="", help="Password for extracting ZIP files")
parser.add_argument("--upload_to", default="", help="Upload folder to HF: username/repo::[remote_folder]::local_folder or username/repo::local_folder")
args, _ = parser.parse_known_args()

try:
    from IPython import get_ipython
    user_ns = get_ipython().user_ns
except:
    user_ns = globals()

# Token parsing into lists for the adaptive fallback
hf_tokens = [t.strip() for t in args.hf.split("::") if t.strip()]
civitai_tokens = [t.strip() for t in args.civitai.split("::") if t.strip()]

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

def get_info(url, headers, suppress_err=False):
    try:
        with requests.get(url, headers=headers, stream=True, timeout=15) as r:
            r.raise_for_status()
            
            if "/login" in r.url:
                if not suppress_err: print(f"❌ Authentication failed: Redirected to login page.")
                return None, None
                
            m = re.search('filename="?([^";]+)"?', r.headers.get("Content-Disposition", ""))
            fn = m.group(1) if m else r.url.split("/")[-1].split("?")[0]
            if "civitai" in url and "." not in fn: fn += ".safetensors"
            return fn, r.url
    except Exception as e:
        if not suppress_err: print(f"❌ Failed to access link: {e}")
        return None, None

def extract_zip(file_path, folder, pwd, prefix=""):
    if not file_path.lower().endswith('.zip'): 
        if prefix: print(f"{CLEAR}{prefix}\033[K")
        return
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            if pwd:
                z.setpassword(pwd.encode('utf-8'))
            infos = z.infolist()
            total = len(infos)
            if total == 0:
                if prefix: print(f"{CLEAR}{prefix}\033[K")
                return
            
            ext_counts = defaultdict(int)
            skipped = 0
            
            for i, info in enumerate(infos, 1):
                print(f"{CLEAR}{prefix} {COLOR_OK}[{COLOR_RESET}Extracting {i}/{total}{COLOR_OK}]{COLOR_RESET}\033[K", end="", flush=True)
                
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
                
            print(f"{CLEAR}{prefix} {COLOR_OK}[{COLOR_RESET}Extracted {ext_summary}{COLOR_OK}]{COLOR_RESET}\033[K")
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
        print(f"{CLEAR}\033[K", end="")
        
    upload_token = hf_tokens[0] if hf_tokens else ""
    if not upload_token:
        print(f"❌ HF Token is required for uploading! Pass it via {COLOR_FN}--hf{COLOR_RESET}")
        return
    if not os.path.exists(local_folder):
        print(f"❌ Local folder {COLOR_DIR}{local_folder}{COLOR_RESET} does not exist!")
        return

    api = HfApi(token=upload_token)
    
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
                print(f"{CLEAR}\033[K", end="", flush=True)
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
                    prefix = f"{COLOR_SUCCESS}◩{COLOR_RESET} {COLOR_FN}{repo_path}{COLOR_RESET}"
                    print(f"{CLEAR}{prefix} {COLOR_OK}[{COLOR_RESET}Already exists{COLOR_OK}]{COLOR_RESET}\033[K", end="", flush=True)
                    clone_success = True
                else:
                    prefix = f"{COLOR_OK}◩{COLOR_RESET} {COLOR_FN}{repo_name}{COLOR_RESET}"
                    print(f"{CLEAR}{prefix} {COLOR_OK}[{COLOR_RESET}Cloning...{COLOR_OK}]{COLOR_RESET}\033[K", end="", flush=True)
                    try:
                        p = subprocess.run(["git", "clone", url], cwd=folder, capture_output=True, text=True)
                        if p.returncode != 0:
                            err = p.stderr.strip().split('\n')[-1] if p.stderr else "Unknown error"
                            print(f"{CLEAR}❌ Clone failed: {err}\033[K")
                        else:
                            prefix = f"{COLOR_SUCCESS}◩{COLOR_RESET} {COLOR_FN}{repo_path}{COLOR_RESET}"
                            print(f"{CLEAR}{prefix}\033[K", end="", flush=True)
                            clone_success = True
                    except Exception as e:
                        print(f"{CLEAR}❌ System error occurred: {e}\033[K")
                
                if clone_success:
                    if args.req:
                        req_file = os.path.join(repo_path, "requirements.txt")
                        if os.path.exists(req_file) and os.path.getsize(req_file) > 0:
                            req_prefix = f"{COLOR_OK}◩{COLOR_RESET} {COLOR_FN}{repo_path}{COLOR_RESET}"
                            print(f"{CLEAR}{req_prefix} {COLOR_OK}[{COLOR_RESET}Installing reqs...{COLOR_OK}]{COLOR_RESET}\033[K", end="", flush=True)
                            try:
                                req_p = subprocess.run(["uv", "pip", "install", "--system", "-r", "requirements.txt"], cwd=repo_path, capture_output=True, text=True)
                                if req_p.returncode == 0: 
                                    installed_pkgs = []
                                    for line in (req_p.stdout + '\n' + req_p.stderr).splitlines():
                                        line = line.strip()
                                        if line.startswith('+ ') or line.startswith('~ '):
                                            parts = line.split()
                                            if len(parts) >= 2:
                                                raw_pkg = " ".join(parts[1:])
                                                m = re.match(r'^([a-zA-Z0-9_\-\.]+)(.*)$', raw_pkg)
                                                if m:
                                                    name, version = m.groups()
                                                    installed_pkgs.append(f"{COLOR_FN}{name}{COLOR_RESET}{COLOR_WARN}{version}{COLOR_RESET}")
                                                else:
                                                    installed_pkgs.append(f"{COLOR_FN}{raw_pkg}{COLOR_RESET}")
                                    
                                    pkg_info = ""
                                    if installed_pkgs:
                                        pkg_str = ", ".join(installed_pkgs)
                                        pkg_info = f" {COLOR_OK}[{COLOR_RESET}{pkg_str}{COLOR_OK}]{COLOR_RESET}"

                                    success_prefix = f"{COLOR_SUCCESS}◩{COLOR_RESET} {COLOR_FN}{repo_path}{COLOR_RESET}"
                                    print(f"{CLEAR}{success_prefix} {COLOR_OK}[{COLOR_RESET}Reqs installed{COLOR_OK}]{COLOR_RESET}{pkg_info}\033[K", end="", flush=True)
                                else:
                                    err_lines = [line.strip() for line in req_p.stderr.split('\n') if line.strip()]
                                    print(f"\n❌ Reqs failed: {err_lines[-1] if err_lines else 'Unknown'}")
                            except Exception as e: 
                                print(f"\n❌ System error: {e}")
                    
                    print() # Append newline since we used end=""
                continue

            # Determine platform and which tokens to use for the retry loop
            is_civitai = "civitai" in url.lower()
            is_hf = "huggingface" in url.lower()
            
            if is_civitai and civitai_tokens:
                tokens_to_try = civitai_tokens
            elif is_hf and hf_tokens:
                tokens_to_try = hf_tokens
            else:
                tokens_to_try = [""] # Run once without tokens if none are provided
                
            download_success = False

            for attempt, current_token in enumerate(tokens_to_try, 1):
                test_url = url
                if is_civitai and current_token and "token=" not in test_url:
                    test_url += f"{'&' if '?' in test_url else '?'}token={current_token}"

                auth = f"Bearer {current_token}" if is_hf and current_token else ""
                h = {"User-Agent": "Mozilla/5.0"}
                if auth: h["Authorization"] = auth
                
                is_last_attempt = (attempt == len(tokens_to_try))
                fn, furl = get_info(test_url, h, suppress_err=not is_last_attempt)
                
                if not fn: 
                    continue

                file_path = os.path.join(folder, fn)
                
                if os.path.exists(file_path) and not os.path.exists(file_path + ".aria2"):
                    prefix = f"{COLOR_SUCCESS}◉{COLOR_RESET} {COLOR_FN}{file_path}{COLOR_RESET}"
                    if fn.lower().endswith('.zip'):
                        extract_zip(file_path, folder, args.zip, prefix)
                    else:
                        print(f"{CLEAR}{prefix} {COLOR_OK}[{COLOR_RESET}Already exists{COLOR_OK}]{COLOR_RESET}\033[K")
                    download_success = True
                    break

                attempt_str = f" [{COLOR_OK}⇋{COLOR_RESET} {attempt}/{COLOR_OK}{len(tokens_to_try)}{COLOR_RESET}]" if len(tokens_to_try) > 1 else ""
                
                cmd = ["aria2c", "--console-log-level=error", "--summary-interval=1", "-c", "-x", "16", "-s", "16", "-k", "1M", "--header=User-Agent: Mozilla/5.0", "-d", folder, "-o", fn]
                
                if furl == test_url:
                    if "huggingface.co" in furl and is_hf and current_token:
                        cmd.append(f"--header=Authorization: Bearer {current_token}")
                    
                cmd.append(furl)
                
                try:
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in p.stdout:
                        if line.startswith("[#"):
                            clean_line = line.strip().strip('[]')
                            parts = clean_line.split()
                            if len(parts) >= 2:
                                size_pct = parts[1]
                                size, pct = size_pct, ""
                                # Extract size and percentage if present
                                if '(' in size_pct and size_pct.endswith(')'):
                                    size, pct = size_pct.split('(')
                                    pct = pct.rstrip(')')
                                
                                # Extract Download Speed (strip 'DL:')
                                speed = next((p.replace('DL:', '') for p in parts if p.startswith('DL:')), '')
                                
                                # Apply colors to units
                                if size: size = re.sub(r'([a-zA-Z]+)', f'{COLOR_UNIT}\\1{COLOR_RESET}', size)
                                if speed: speed = re.sub(r'([a-zA-Z]+)', f'{COLOR_UNIT}\\1{COLOR_RESET}', speed)
                                if pct: pct = pct.replace('%', f'{COLOR_UNIT}%{COLOR_RESET}')
                                
                                out_parts = []
                                if size: out_parts.append(size) # Removed 'Size: ' prefix
                                if pct: out_parts.append(f"[{pct}]")
                                if speed: out_parts.append(f"DL: {speed}")
                                
                                aria_out = " ".join(out_parts)
                                if aria_out:
                                    # Wrap entire aria output in colored brackets
                                    aria_out = f"{COLOR_OK}[{COLOR_RESET}{aria_out}{COLOR_OK}]{COLOR_RESET}"
                                
                                prefix = f"{COLOR_OK}◉{COLOR_RESET} {COLOR_FN}{fn}{COLOR_RESET}{attempt_str}"
                                
                                # Overwrite line cleanly using \r, massive wipe padding, and \033[K
                                print(f"{CLEAR}{prefix} {aria_out}\033[K", end="", flush=True)
                    p.wait()
                    
                    if p.returncode == 0:
                        prefix = f"{COLOR_SUCCESS}◉{COLOR_RESET} {COLOR_FN}{file_path}{COLOR_RESET}"
                        if fn.lower().endswith('.zip'):
                            extract_zip(file_path, folder, args.zip, prefix)
                        else:
                            print(f"{CLEAR}{prefix}\033[K")
                        download_success = True
                        break # Success! Break out of the token retry loop
                    else:
                        if is_last_attempt:
                            print(f"\n❌ Download failed (Aria2 Error Code: {p.returncode})")
                        else:
                            print() # Clears output line to restart neatly
                except Exception as e:
                    if is_last_attempt:
                        print(f"\n❌ System error occurred: {e}")
                    else:
                        print()
            
            if not download_success and len(tokens_to_try) > 1:
                print(f"❌ {COLOR_ERR}All provided tokens failed for this file.{COLOR_RESET}")
