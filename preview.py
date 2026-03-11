import os
import time
import threading
import base64
import io
import argparse
from PIL import Image, UnidentifiedImageError
from IPython.display import display, HTML, update_display

class MonitorGambarKaggle:
    def __init__(self, folder_path, max_images=20, kolom=4, interval_detik=5):
        self.folder_path = folder_path
        self.max_images = max_images
        self.kolom = kolom
        self.interval = interval_detik
        self._sedang_auto_refresh = False
        self._thread = None
        self.display_id = f"monitor_{int(time.time())}"

    def ambil_file_terbaru(self):
        if not os.path.exists(self.folder_path): return []
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif')
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_exts)]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.folder_path, x)), reverse=True)
        return files[:self.max_images]

    def generate_html(self):
        files = self.ambil_file_terbaru()
        html = f'<div style="background:#0e0e0e;color:#d4d4d4;padding:12px;border-radius:6px;font-family:sans-serif;border:1px solid #2d2d2d;"><div style="font-size:11px;color:#858585;margin-bottom:12px;display:flex;justify-content:space-between;"><span><span style="color:#4CAF50;">●</span> <b>Live</b> | {time.strftime("%H:%M:%S")}</span><span style="font-family:monospace;">{len(files)} img | {self.folder_path}</span></div>'
        
        if not files: return html + '<div style="color:#666;font-size:12px;text-align:center;padding:20px;">Menunggu gambar...</div></div>'
        
        html += f'<div style="display:grid;grid-template-columns:repeat({self.kolom}, 1fr);gap:6px;">'
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
        return html + "</div></div>"

    def tampilkan_ui(self):
        display(HTML(self.generate_html()), display_id=self.display_id)

    def refresh_manual(self):
        update_display(HTML(self.generate_html()), display_id=self.display_id)

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
            self.refresh_manual()
            for _ in range(self.interval * 10): 
                if not self._sedang_auto_refresh: break
                time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default="/root/LoRA/output/sample")
    args, _ = parser.parse_known_args()
    
    os.makedirs(args.target_folder, exist_ok=True)
    monitor = MonitorGambarKaggle(args.target_folder, 16, 4, 10)
    monitor.tampilkan_ui()
    monitor.mulai_auto_refresh()
