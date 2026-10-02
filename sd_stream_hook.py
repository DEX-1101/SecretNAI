"""
SD Stream hook - tiny WebUI (Forge Classic / A1111) extension script for sd_stream.py auto-save.

Install on the server as:
    <webui folder>/extensions/sd-stream-hook/scripts/sd_stream_hook.py
then restart the WebUI (with your usual --encrypt-pass etc.).

What it does:
  * remembers the last few images the WebUI saved (as plain, decrypted PNG + generation info)
  * exposes them to the SD Stream desktop app:
        GET /sd_stream/latest[?after=ID]   -> {"latest": ID, "items": [{"id", "name", "ts"}]}
        GET /sd_stream/image/ID            -> the PNG

Images are read back through PIL.Image.open, which the sd-image-encryption extension patches to
decrypt, so what the app receives is the real picture, not the pixel-shuffled file on disk.
"""

import io
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

from fastapi import FastAPI, Response
from PIL import Image, PngImagePlugin

from modules import script_callbacks

KEEP = 12  # how many recent images stay available
SKIP_SUFFIXES = ("-before-highres-fix", "-before-color-correction", "-init-images",
                 "-mask", "-mask-composite")
DROP_KEYS = {"Encrypt", "EncryptPwdSha"}

_lock = threading.Lock()
_boot = int(time.time() * 1000)  # ids only grow, even across WebUI restarts
_counter = _boot
_items: "OrderedDict[int, dict]" = OrderedDict()


def _to_png(path, fallback_image, fallback_info):
    """Return PNG bytes (with generation info) of the saved image, decrypted."""
    img, info = None, {}
    try:
        img = Image.open(path)  # patched by sd-image-encryption -> decrypted pixels
        img.load()
        info = dict(img.info or {})
        if info.get("Encrypt"):  # still scrambled (encryption inactive), use the in-memory copy
            img = None
    except Exception:
        img = None
    if img is None:
        img, info = fallback_image, dict(fallback_info or {})
    for k, v in (fallback_info or {}).items():  # e.g. infotext missing from a jpg/webp file
        info.setdefault(k, v)
    png = PngImagePlugin.PngInfo()
    for k, v in info.items():
        if isinstance(v, str) and v and k not in DROP_KEYS:
            png.add_text(k, v)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=png)
    return buf.getvalue()


def _on_image_saved(params):
    global _counter
    try:
        name = os.path.basename(params.filename)
        stem = os.path.splitext(name)[0]
        if name.startswith("grid-") or stem.endswith(SKIP_SUFFIXES):
            return  # only final result images, no grids / intermediate saves
        data = _to_png(params.filename, params.image, getattr(params, "pnginfo", None))
        with _lock:
            _counter += 1
            _items[_counter] = {"name": name, "ts": time.time(), "data": data}
            while len(_items) > KEEP:
                _items.popitem(last=False)
    except Exception as ex:
        print(f"[sd_stream_hook] could not record {getattr(params, 'filename', '?')}: {ex}")


def _register(_demo, app: FastAPI):
    @app.get("/sd_stream/latest")
    def latest(after: Optional[int] = None):
        with _lock:
            newest = next(reversed(_items), _boot)
            items = []
            if after is not None:
                items = [{"id": i, "name": v["name"], "ts": v["ts"]} for i, v in _items.items() if i > after]
        return {"latest": newest, "items": items}

    @app.get("/sd_stream/image/{item_id}")
    def image(item_id: int):
        with _lock:
            item = _items.get(item_id)
        if item is None:
            return Response(status_code=404)
        return Response(content=item["data"], media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    print("[sd_stream_hook] ready: /sd_stream/latest")


script_callbacks.on_app_started(_register)
script_callbacks.on_image_saved(_on_image_saved)
