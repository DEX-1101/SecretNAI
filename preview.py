import os
import time
import threading
import base64
import io
import argparse
import subprocess
import shutil
import psutil
from PIL import Image, UnidentifiedImageError
from IPython.display import display, HTML, update_display, clear_output

class MonitorGambarKaggle:
    def __init__(self, folder_path, max_images=20, kolom=4, interval_detik=5):
        self.folder_path = folder_path
        self.max_images = max_images
        self.kolom = kolom
        self.interval = interval_detik
        self._sedang_auto_refresh = False
        self._thread = None
        self.display_id = f"monitor_{int(time.time())}"
        
        # Cache untuk optimasi (gambar tidak perlu di-load tiap detik)
        self._cached_img_html = ""
        self._last_img_update = 0
        self._last_files_count = 0
        
        # Proteksi resource (Mencegah script hang saat CPU/GPU 100%)
        self._last_files_state = []
        self._cached_gpu_info = "GPU: N/A | VRAM: N/A"
        self._last_gpu_update = 0

    def ambil_file_terbaru(self):
        if not os.path.exists(self.folder_path): return [], []
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif')
        file_list = []
        for f in os.listdir(self.folder_path):
            if f.lower().endswith(valid_exts):
                path = os.path.join(self.folder_path, f)
                try:
                    file_list.append((f, os.path.getmtime(path)))
                except OSError: pass # Abaikan jika file sedang ditulis ulang
        
        file_list.sort(key=lambda x: x[1], reverse=True)
        file_list = file_list[:self.max_images]
        files = [f[0] for f in file_list]
        return files, file_list

    def get_sys_stats(self):
        # CPU
        try:
            cpu = psutil.cpu_percent()
        except:
            cpu = 0.0
            
        # RAM
        try:
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            ram_percent = ram.percent
        except:
            ram_used_gb = ram_total_gb = ram_percent = 0.0
            
        # Disk
        try:
            disk = shutil.disk_usage("/")
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
        except:
            disk_used_gb = disk_total_gb = disk_percent = 0.0
        
        # GPU - Menggunakan timeout & interval agar tidak hang saat GPU 100% sibuk
        current_time = time.time()
        if current_time - self._last_gpu_update >= 2.0: # Hanya panggil setiap 2 detik
            try:
                # Timeout 0.5 detik sangat penting untuk mencegah proses stuck
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'],
                    encoding='utf-8',
                    timeout=0.5
                )
                gpus = result.strip().split('\n')
                if gpus:
                    gpu_util, vram_used, vram_total = gpus[0].split(', ')
                    self._cached_gpu_info = f"GPU: {gpu_util}% | VRAM: {vram_used}/{vram_total} MB"
            except Exception:
                pass # Jika gagal/timeout, gunakan cache GPU yang terakhir berhasil
            self._last_gpu_update = current_time
            
        stats_html = f"""
        <div style="font-size:12px; color:#a0a0a0; background:#1a1a1a; padding:10px; border-radius:4px; margin-bottom:12px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; font-family:monospace; border:1px solid #333;">
            <span><b style="color:#fff;">CPU:</b> {cpu}%</span>
            <span><b style="color:#fff;">RAM:</b> {ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({ram_percent}%)</span>
            <span><b style="color:#fff;">Disk:</b> {disk_used_gb:.1f}/{disk_total_gb:.1f} GB ({disk_percent:.1f}%)</span>
            <span style="color:#4CAF50;"><b>{self._cached_gpu_info}</b></span>
        </div>
        """
        return stats_html

    def generate_img_html(self):
        files, current_state = self.ambil_file_terbaru()
        self._last_files_count = len(files)
        
        # Optimasi Ekstra: Hindari proses ulang gambar PIL jika tidak ada file yang berubah
        if current_state == self._last_files_state and self._cached_img_html:
            return self._cached_img_html
            
        self._last_files_state = current_state
        
        if not files: 
            return '<div style="color:#666;font-size:12px;text-align:center;padding:20px;">Menunggu gambar...</div>'
        
        html = f'<div style="display:grid;grid-template-columns:repeat({self.kolom}, 1fr);gap:6px;">'
        for f in files:
            try:
                with Image.open(os.path.join(self.folder_path, f)) as img:
                    if img.mode != 'RGB': img = img.convert('RGB')
                    img.thumbnail((300, 300))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                html += f'<div style="background:#1a1a1a;padding:4px;border-radius:4px;"><img src="data:image/jpeg;base64,{b64}" style="width:100%;height:auto;border-radius:2px;display:block;"/><div style="margin-top:4px;font-size:10px;color:#777;text-align:center;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-family:monospace;">{f}</div></div>'
            except (IOError, OSError, UnidentifiedImageError): pass
        return html + "</div>"

    def generate_html(self, force_img_update=False):
        current_time = time.time()
        
        # Hanya perbarui gambar jika sudah melewati interval yang ditentukan
        if force_img_update or (current_time - self._last_img_update >= self.interval):
            self._cached_img_html = self.generate_img_html()
            self._last_img_update = current_time

        # Selalu perbarui statistik sistem (Hardware)
        sys_html = self.get_sys_stats()
        
        # Header UI
        html = f'<div style="background:#0e0e0e;color:#d4d4d4;padding:12px;border-radius:6px;font-family:sans-serif;border:1px solid #2d2d2d;">'
        html += f'<div style="font-size:11px;color:#858585;margin-bottom:12px;display:flex;justify-content:space-between;">'
        html += f'<span><span style="color:#4CAF50;">●</span> <b>Live</b> | {time.strftime("%H:%M:%S")}</span>'
        html += f'<span style="font-family:monospace;">{self._last_files_count} img | {self.folder_path}</span></div>'
        
        return html + sys_html + self._cached_img_html + '</div>'

    def tampilkan_ui(self):
        display(HTML(self.generate_html(force_img_update=True)), display_id=self.display_id)

    def refresh_manual(self, force_img_update=False):
        update_display(HTML(self.generate_html(force_img_update=force_img_update)), display_id=self.display_id)

    def mulai_auto_refresh(self):
        if not self._sedang_auto_refresh:
            self._sedang_auto_refresh = True
            self._thread = threading.Thread(target=self._loop_background, daemon=True)
            self._thread.start()

    def hentikan_auto_refresh(self):
        self._sedang_auto_refresh = False
        if self._thread: self._thread.join(timeout=1)

    def _loop_background(self):
        while self._sedang_auto_refresh:
            # Perbarui UI (Hardware stats tiap 1 detik, Image sesuai interval)
            self.refresh_manual(force_img_update=False)
            
            # Sleep 1 detik (dibagi 0.1s agar thread bisa ditutup instan)
            for _ in range(10): 
                if not self._sedang_auto_refresh: break
                time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default="/root/LoRA/output/sample")
    args, _ = parser.parse_known_args()
    
    os.makedirs(args.target_folder, exist_ok=True)
    
    # Hentikan thread lama jika cell di-rerun untuk elak penumpukan proses
    if 'monitor_aktif' in globals():
        monitor_aktif.hentikan_auto_refresh()
        
    clear_output(wait=True) # Bersihkan UI output sebelumnya
    monitor_aktif = MonitorGambarKaggle(args.target_folder, 16, 4, 10) # Update gambar setiap 10 detik
    monitor_aktif.tampilkan_ui()
    monitor_aktif.mulai_auto_refresh()
