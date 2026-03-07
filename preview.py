import os
import math
import time
import threading
import base64
import io
import argparse
from PIL import Image
from IPython.display import display, HTML, update_display

class MonitorGambarKaggle:
    def __init__(self, folder_path, max_images=20, kolom=4, interval_detik=5):
        self.folder_path = folder_path
        self.max_images = max_images
        self.kolom = kolom
        self.interval = interval_detik
        self._sedang_auto_refresh = False
        self._thread = None
        self.display_id = "monitor_gambar_" + str(int(time.time()))

    def ambil_file_terbaru(self):
        if not os.path.exists(self.folder_path):
            return []
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif')
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_exts)]
        
        if not files:
            return []
            
        files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.folder_path, x)), 
            reverse=True
        )
        return files[:self.max_images]

    def generate_html(self):
        """Mengonversi gambar menjadi Base64 dan menyusunnya dalam Grid HTML."""
        files = self.ambil_file_terbaru()
        waktu_sekarang = time.strftime('%H:%M:%S')
        
        # Header HTML
        html_content = f"""
        <div style="background-color: #0e0e0e; color: #d4d4d4; padding: 12px; border-radius: 6px; font-family: sans-serif; border: 1px solid #2d2d2d;">
            <div style="font-size: 11px; color: #858585; margin-bottom: 12px; display: flex; justify-content: space-between;">
                <span><span style="color: #4CAF50;">●</span> <b>Live</b> | {waktu_sekarang}</span>
                <span style="font-family: monospace;">{len(files)} img | {self.folder_path}</span>
            </div>
        """

        if not files:
            html_content += f'<div style="color: #666; font-size: 12px; text-align: center; padding: 20px;">Menunggu gambar...</div></div>'
            return html_content

        html_content += f'<div style="display: grid; grid-template-columns: repeat({self.kolom}, 1fr); gap: 6px;">'
        
        for f in files:
            img_path = os.path.join(self.folder_path, f)
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.thumbnail((300, 300))
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=70)
                    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                html_content += f"""
                <div style="background: #1a1a1a; padding: 4px; border-radius: 4px;">
                    <img src="data:image/jpeg;base64,{b64_str}" style="width: 100%; height: auto; border-radius: 2px; display: block;" />
                    <div style="margin-top: 4px; font-size: 10px; color: #777; text-align: center; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-family: monospace;" title="{f}">{f}</div>
                </div>
                """
            except Exception:
                pass # Lewati jika gagal membaca gambar
                
        html_content += "</div></div>"
        return html_content

    def tampilkan_ui(self):
        """Merender HTML pertama kali ke layar Jupyter."""
        html = self.generate_html()
        display(HTML(html), display_id=self.display_id)

    def refresh_manual(self):
        """Memperbarui gambar yang sudah ada secara manual tanpa berkedip."""
        html = self.generate_html()
        update_display(HTML(html), display_id=self.display_id)

    def mulai_auto_refresh(self):
        """Memulai auto-refresh di latar belakang (Background Thread)."""
        if self._sedang_auto_refresh:
            print("▶️ Auto-refresh sudah berjalan.")
            return
            
        self._sedang_auto_refresh = True
        self._thread = threading.Thread(target=self._loop_background, daemon=True)
        self._thread.start()
        print(f"▶️ Auto-refresh dimulai. Gambar akan diperbarui tiap {self.interval} detik.")

    def hentikan_auto_refresh(self):
        """Menghentikan auto-refresh latar belakang."""
        self._sedang_auto_refresh = False
        if self._thread:
            self._thread.join(timeout=1)
        print("⏸️ Auto-refresh dihentikan.")

    def _loop_background(self):
        """Fungsi loop untuk Background Thread"""
        while self._sedang_auto_refresh:
            self.refresh_manual()
            for _ in range(self.interval * 10): 
                if not self._sedang_auto_refresh:
                    break
                time.sleep(0.1)

# === BLOK EKSEKUSI UTAMA ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor folder untuk gambar terbaru.")
    parser.add_argument("--target_folder", type=str, default="/root/LoRA/output/sample", help="Folder target yang akan dimonitor")
    args = parser.parse_args()

    folder_target = args.target_folder
    
    # Buat folder jika belum ada agar tidak error
    os.makedirs(folder_target, exist_ok=True)

    monitor = MonitorGambarKaggle(
        folder_path = folder_target,
        max_images = 16,        # Maksimal gambar ditampilkan (terbaru)
        kolom = 4,              # Jumlah kolom ke samping
        interval_detik = 10     # Jarak waktu auto-refresh
    )

    monitor.tampilkan_ui()
    monitor.mulai_auto_refresh()

    # Menahan main thread agar skrip tidak langsung selesai
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.hentikan_auto_refresh()
        print("\nMonitoring dihentikan oleh user.")
