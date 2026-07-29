import os, subprocess, requests, re, argparse, shutil, zipfile, html, time, uuid
from collections import defaultdict
from IPython.display import display, HTML

try:
    from google.colab import output
    from IPython import get_ipython
    ipy = get_ipython()
    user_ns = ipy.user_ns if ipy else globals()
    IN_COLAB = True
except ImportError:
    user_ns = globals()
    IN_COLAB = False
    print("⚠️ Warning: This UI architecture is specifically designed for Google Colab.")

VAR_REGEX = re.compile(r'\{([^}]+)\}')
def resolve_vars(text):
    return VAR_REGEX.sub(lambda m: str(user_ns.get(m.group(1), m.group(0))), text)

# Argparse setup (Used to populate default values in the UI)
parser = argparse.ArgumentParser()
parser.add_argument("--hf", default="", help="HuggingFace API token(s)")
parser.add_argument("--civitai", default="", help="Civitai API token(s)")
parser.add_argument("--req", action="store_true", help="Install requirements.txt")
parser.add_argument("--zip", default="", help="Password for ZIP files")
parser.add_argument("--upload_to", default="", help="Upload folder to HF")
args, _ = parser.parse_known_args()

# Populate initial download list from Colab notebook variables
init_dl_list = user_ns.get('DOWNLOAD_LIST', '')
if not init_dl_list:
    raw_batches = user_ns.get('DOWNLOAD_BATCHES', {})
    lines = []
    for k, v in raw_batches.items():
        lines.append(k)
        for url in v:
            lines.append(url)
    init_dl_list = "\n".join(lines)

def get_info(url, headers, suppress_err=False):
    try:
        with requests.get(url, headers=headers, stream=True, timeout=15) as r:
            r.raise_for_status()
            if "/login" in r.url: return None, None
            m = re.search('filename="?([^";]+)"?', r.headers.get("Content-Disposition", ""))
            fn = m.group(1) if m else r.url.split("/")[-1].split("?")[0]
            if "civitai" in url and "." not in fn: fn += ".safetensors"
            return fn, r.url
    except Exception:
        return None, None

class DownloaderUI:
    """
    Unified Dashboard UI for Google Colab based on the requested template.
    Displays a history of completed files (checks/crosses) and a single active progress row.
    """
    def __init__(self, total_files):
        self.display_id = "ui_" + uuid.uuid4().hex
        self.is_notebook = get_ipython() is not None
        self.status = "Initializing..."
        self.current_file = ""
        self.detail_text = ""
        self.file_size = ""
        self.current_size = ""
        self.file_idx = 0
        self.total_files = total_files
        self.pct = 0.0
        self.speed = ""
        self.eta = ""
        self.history = []
        self.is_finished = False
        self.last_update = 0
        self.update_interval = 0.2

        if self.is_notebook:
            display(HTML(""), display_id=self.display_id)
            self._render()

    def update_status(self, text):
        self.status = text
        self._render()

    def start_file(self, name, increment=True):
        self.current_file = name
        self.detail_text = ""
        self.file_size = ""
        self.current_size = ""
        if increment:
            self.file_idx += 1
        self.pct = 0.0
        self.speed = "..."
        self.eta = "..."
        self.status = "Downloading"
    def update_progress(self, pct, speed, eta, detail_text=None, file_size=None, current_size=None):
        self.pct = pct
        self.speed = speed
        self.eta = eta
        if detail_text is not None:
            self.detail_text = detail_text
        if file_size is not None:
            self.file_size = file_size
        if current_size is not None:
            self.current_size = current_size
        now = time.time()
        if (now - self.last_update) > self.update_interval or pct >= 100:
            self._render()
            self.last_update = now

    def add_history(self, name, status_type):
        self.history.append((name, status_type))
        self.current_file = "" # Clear current so progress bar hides until next file
        self.detail_text = ""
        self._render()

    def finish(self):
        self.status = ""
        self.is_finished = True
        self.current_file = ""
        self._render()

    def _render(self):
        if not self.is_notebook: return

        status_color = "#4ade80" # Default Green
        if "Error" in self.status or "❌" in self.status or "Failed" in self.status:
            status_color = "#f87171"
        elif "Skipping" in self.status or "⚠️" in self.status or "Already" in self.status or "Extracted" in self.status:
            status_color = "#fbbf24"
        elif "✅" in self.status or "Complete" in self.status or "Success" in self.status:
            status_color = "#4ade80"

        hist_html = ""
        if self.history:
            tags = []
            for name, st in self.history:
                if st in ['success', 'skipped']:
                    tags.append(f'<span style="background:#1e1e1e; border:1px solid #2a2a2a; padding:3px 8px; border-radius:4px; color:#e2e2e2;"><span style="color:#4ade80; font-weight:900;">✓</span> {html.escape(name)}</span>')
                else:
                    tags.append(f'<span style="background:#1e1e1e; border:1px solid #2a2a2a; padding:3px 8px; border-radius:4px; color:#e2e2e2;"><span style="color:#f87171; font-weight:900;">✗</span> {html.escape(name)}</span>')
            
            hist_margin = "margin-top: 10px;" if (self.current_file and not self.is_finished) else ""
            hist_html = f"""
            <div style="font-size: 11.5px; display: flex; flex-wrap: wrap; gap: 8px; {hist_margin}">
                {''.join(tags)}
            </div>
            """

        pb_html = ""
        if not self.is_finished and self.current_file:
            display_speed = self.speed.replace("MiB", "MB").replace("KiB", "KB").replace("GiB", "GB")
            display_speed = re.sub(r'([0-9.]+)([a-zA-Z]+)', r'\1 \2', display_speed)
            
            display_size = self.file_size.replace("MiB", "MB").replace("KiB", "KB").replace("GiB", "GB") if self.file_size else ""
            if display_size:
                display_size = re.sub(r'([0-9.]+)([a-zA-Z]+)', r'\1 \2', display_size)
                
            display_current = self.current_size.replace("MiB", "MB").replace("KiB", "KB").replace("GiB", "GB") if hasattr(self, 'current_size') and self.current_size else ""
            if display_current:
                display_current = re.sub(r'([0-9.]+)([a-zA-Z]+)', r'\1 \2', display_current)
                
            if display_current and display_size:
                progress_text = f"{display_current} / {display_size}"
            else:
                progress_text = f"{self.pct:.1f}%"
            
            queue_badge = f'<span style="background: #2a2a2a; color: #e2e2e2; padding: 2px 6px;">{self.file_idx} / {self.total_files}</span>' if self.total_files > 0 else ""
            status_pill = f'<span style="background: {status_color}; color: #111; padding: 2px 6px; font-weight: 600;">{html.escape(self.status)}</span>'
            size_badge = f'<span style="background: #2a2a2a; color: #e2e2e2; padding: 2px 6px; margin-left: 8px; font-size: 11px; font-weight: bold;">{html.escape(display_size)}</span>' if display_size else ""
            detail_badge = f'<span style="background: #2a2a2a; color: #e2e2e2; padding: 2px 6px; margin-left: 8px; font-size: 11px; font-weight: bold;">{html.escape(self.detail_text)}</span>' if self.detail_text else ""
            
            pb_html = f"""
            <div style="font-size: 12px; margin-bottom: 4px; color: #e2e2e2; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                {html.escape(self.current_file)}{size_badge}{detail_badge}
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #2d2d2d; height: 4px; width: 100%; border-radius: 0; overflow:hidden;">
                    <div style="background: #4ade80; height: 100%; width: {self.pct}%; transition: width 0.3s linear;"></div>
                </div>
                <div style="font-size: 11px; font-weight: 600; color: #4ade80; min-width: 110px; text-align: right; white-space: nowrap; flex-shrink: 0;">
                    {progress_text}
                </div>
            </div>
            <div style="display: flex; gap: 6px; font-size: 11px; margin-top: 6px;">
                {status_pill}
                <span style="background: #2a2a2a; color: #e2e2e2; padding: 2px 6px;">{html.escape(display_speed)}</span>
                <span style="background: #2a2a2a; color: #e2e2e2; padding: 2px 6px;">{html.escape(self.eta)}</span>
                {queue_badge}
            </div>
            """

        header_html = ""
        # Hide standalone header text if a file is currently active or if status is completely empty
        if self.status and not (self.current_file and not self.is_finished):
            header_html = f'<div style="font-size: 12px; font-weight: 500; color: {status_color}; margin-bottom: {6 if hist_html else 0}px;">{self.status}</div>'

        html_content = f"""
        <div style="font-family: 'Segoe UI Variable', 'Segoe UI', -apple-system, sans-serif;
                    background-color: #181818; border: 1px solid #2e2e2e; padding: 10px 14px; 
                    max-width: 800px; border-left: 3px solid {status_color}; border-radius: 0;">
            {header_html}
            {pb_html}
            {hist_html}
        </div>
        """

        if not hasattr(self, '_displayed'):
            display(HTML(html_content), display_id=self.display_id)
            self._displayed = True
        else:
            display(HTML(html_content), display_id=self.display_id, update=True)


def start_colab_dl(dl_text, hf_token, civitai_token, req, zip_pwd, upload_to):
    DOWNLOAD_BATCHES = defaultdict(list)
    current_dir = "downloads"
    for line in dl_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        line = resolve_vars(line)
        if line.startswith('http'): DOWNLOAD_BATCHES[current_dir].append(line)
        else: current_dir = line

    total_files = sum(len(links) for links in DOWNLOAD_BATCHES.values())
    
    # Initialize the new minimalist UI
    ui = DownloaderUI(total_files)

    if not DOWNLOAD_BATCHES and not upload_to:
        ui.update_status("No download links provided.")
        return

    # Check for aria2c engine
    if DOWNLOAD_BATCHES and not shutil.which("aria2c"):
        ui.update_status("Installing aria2c...")
        subprocess.run("apt-get install -y -qq aria2", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ui.update_status("Aria2 engine ready")

    hf_tokens = [t.strip() for t in hf_token.split("::") if t.strip()]
    civitai_tokens = [t.strip() for t in civitai_token.split("::") if t.strip()]

    for folder, links in DOWNLOAD_BATCHES.items():
        if not links: continue
        os.makedirs(folder, exist_ok=True)
        
        for url in links:
            # Handle Git Repositories
            if "github.com" in url and not any(x in url for x in ["/releases/download/", "/raw/", "/blob/"]):
                repo_name = [p for p in url.split("/") if p][-1].replace(".git", "")
                repo_path = os.path.join(folder, repo_name)
                
                if os.path.exists(repo_path):
                    ui.file_idx += 1
                    ui.add_history(repo_name, "skipped")
                    clone_success = True
                else:
                    ui.start_file(repo_name)
                    ui.update_status("Cloning")
                    
                    clone_success = False
                    try:
                        # Use Popen to stream git progress in real-time
                        cmd = ["git", "clone", "--progress", url]
                        p = subprocess.Popen(cmd, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                        for line in p.stdout:
                            clean_line = line.strip()
                            if clean_line:
                                # Git uses \r to rewrite terminal lines, split it to get the latest status chunk
                                parts = clean_line.split('\r')
                                last_part = parts[-1].strip()
                                if last_part:
                                    ui.update_progress(0.0, "-", "-", detail_text=last_part[:60])
                        p.wait()
                        clone_success = (p.returncode == 0)
                    except Exception:
                        pass
                    
                    ui.detail_text = ""
                    if clone_success:
                        ui.add_history(repo_name, "success")
                    else:
                        ui.add_history(repo_name, "error")
                
                if clone_success and req:
                    req_file = os.path.join(repo_path, "requirements.txt")
                    if os.path.exists(req_file) and os.path.getsize(req_file) > 0:
                        ui.start_file(repo_name, increment=False)
                        ui.update_status("Installing Reqs")
                        req_p = subprocess.run(["uv", "pip", "install", "--system", "-r", "requirements.txt"], cwd=repo_path, capture_output=True, text=True)
                        if req_p.returncode != 0:
                            req_p = subprocess.run(["pip", "install", "-r", "requirements.txt"], cwd=repo_path, capture_output=True, text=True)
                continue

            # Standard File Downloads
            is_civitai = "civitai" in url.lower()
            is_hf = "huggingface" in url.lower()
            tokens_to_try = civitai_tokens if is_civitai and civitai_tokens else (hf_tokens if is_hf and hf_tokens else [""])
            
            download_success = False
            for attempt, current_token in enumerate(tokens_to_try, 1):
                test_url = url
                if is_civitai and current_token and "token=" not in test_url:
                    test_url += f"{'&' if '?' in test_url else '?'}token={current_token}"

                auth = f"Bearer {current_token}" if is_hf and current_token else ""
                h = {"User-Agent": "Mozilla/5.0"}
                if auth: h["Authorization"] = auth
                
                fn, furl = get_info(test_url, h, suppress_err=(attempt < len(tokens_to_try)))
                if not fn: continue

                file_path = os.path.join(folder, fn)
                if os.path.exists(file_path) and not os.path.exists(file_path + ".aria2"):
                    if fn.lower().endswith('.zip'):
                        ui.start_file(fn)
                    else:
                        ui.file_idx += 1
                        ui.add_history(fn, "skipped")
                    download_success = True
                    break

                ui.start_file(fn)

                cmd = ["aria2c", "--console-log-level=error", "--summary-interval=1", "-c", "-x", "16", "-s", "16", "-k", "1M", "--header=User-Agent: Mozilla/5.0", "-d", folder, "-o", fn]
                if furl == test_url and is_hf and current_token: cmd.append(f"--header=Authorization: Bearer {current_token}")
                cmd.append(furl)
                
                try:
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in p.stdout:
                        if line.startswith("[#"):
                            pct_m = re.search(r'\((\d+)%\)', line)
                            speed_m = re.search(r'DL:([^\s]+)', line)
                            eta_m = re.search(r'ETA:([^\s\]]+)', line)
                            size_m = re.search(r'([^\s]+)/([^\s\(]+)\(', line)
                            
                            pct = float(pct_m.group(1)) if pct_m else 0.0
                            speed = speed_m.group(1) if speed_m else "..."
                            eta = eta_m.group(1) if eta_m else "..."
                            curr_size = size_m.group(1) if size_m else ""
                            tot_size = size_m.group(2) if size_m else ""
                            
                            ui.update_progress(pct, speed, eta, file_size=tot_size, current_size=curr_size)
                            
                    p.wait()
                    if p.returncode == 0:
                        download_success = True
                        break
                except Exception:
                    pass
            
            # ZIP Extraction Logic
            if download_success and fn and fn.lower().endswith('.zip'):
                ui.current_file = fn
                ui.pct = 0.0
                ui.speed = "-"
                ui.eta = "-"
                ui.update_status("Extracting")
                try:
                    with zipfile.ZipFile(file_path, 'r') as z:
                        if zip_pwd: z.setpassword(zip_pwd.encode('utf-8'))
                        infos = z.infolist()
                        total_items = len(infos)
                        for i, info in enumerate(infos, 1):
                            ui.update_progress((i/total_items) * 100, "-", "-", detail_text=info.filename, file_size=str(total_items), current_size=str(i))
                            z.extract(info, folder)
                    ui.detail_text = ""
                    ui.add_history(f"{fn} ({total_items} files)", "success")
                except Exception as e:
                    ui.detail_text = ""
                    ui.add_history(f"{fn} (Zip Error)", "error")
            elif download_success:
                ui.add_history(fn, "success")
            else:
                try_name = url.split('/')[-1][:20]
                ui.add_history(try_name, "error")

    if upload_to:
        ui.start_file("HuggingFace Upload", increment=False)
        ui.update_status("Preparing Upload")
        parts = upload_to.split("::")
        if len(parts) >= 2:
            repo_id = parts[0]
            local_folder = parts[-1]
            remote_folder = parts[1] if len(parts) == 3 else ""
            
            if not hf_tokens: 
                ui.update_status("HF Token required for upload.")
            elif not os.path.exists(local_folder): 
                ui.update_status(f"Folder {local_folder} not found.")
            else:
                try:
                    from huggingface_hub import HfApi
                except ImportError:
                    subprocess.run(["pip", "install", "-q", "huggingface_hub"])
                    from huggingface_hub import HfApi
                
                api = HfApi(token=hf_tokens[0])
                try:
                    api.model_info(repo_id)
                except:
                    ui.update_status("Creating Repo")
                    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
                
                ui.update_status("Uploading")
                try:
                    api.upload_folder(folder_path=local_folder, path_in_repo=remote_folder, repo_id=repo_id, repo_type="model")
                    ui.add_history("HF Upload", "success")
                except Exception as e:
                    ui.add_history("HF Upload", "error")
        else:
            ui.update_status("❌ Invalid upload format.")

    ui.finish()


if IN_COLAB:
    start_colab_dl(init_dl_list, args.hf, args.civitai, args.req, args.zip, args.upload_to)
