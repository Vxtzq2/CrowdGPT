#!/usr/bin/env python3
"""
CrowdGPT GUI Client v1, DirectML runtime fallback.
"""
import os, sys, io, json, time, struct, math, gc, threading, base64, logging, tempfile, atexit, subprocess, hashlib, shutil, ctypes
from pathlib import Path
import webview
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from torch.utils.checkpoint import checkpoint
import http.client
from requests.exceptions import ChunkedEncodingError
import gzip

logging.getLogger("pywebview").setLevel(logging.CRITICAL)
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

CLIENT_VERSION = "1"
VERSION_URL = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/version.txt"
CLIENT_URL  = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/client.py"

def _parse_version(v):
    try:
        parts = str(v).strip().split("-")[0].split(".")
        return tuple(int(x) for x in parts[:4])
    except Exception:
        return (0,)

def _fetch_remote_version():
    try:
        r = requests.get(VERSION_URL, timeout=5)
        r.raise_for_status()
        line = r.text.strip().splitlines()[0].strip()
        return line or None
    except Exception:
        return None

def _download_new_client(expected_version):
    try:
        r = requests.get(CLIENT_URL, timeout=60)
        r.raise_for_status()
        code = r.text
    except Exception:
        return None
    try:
        compile(code, "client.py", "exec")
    except SyntaxError:
        return None
    markers = [f'CLIENT_VERSION = "{expected_version}"', f"CLIENT_VERSION = '{expected_version}'"]
    if not any(m in code for m in markers):
        return None
    return code

def _apply_update(new_code):
    try:
        current = Path(__file__).resolve()
    except Exception:
        return False
    backup = current.with_name(current.name + ".bak")
    tmp = current.with_name(current.name + ".new")
    try:
        try:
            shutil.copy2(str(current), str(backup))
        except Exception:
            pass
        tmp.write_text(new_code, encoding="utf-8")
        try:
            st = current.stat()
            os.chmod(str(tmp), st.st_mode)
        except Exception:
            pass
        os.replace(str(tmp), str(current))
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False

def check_for_update():
    remote = _fetch_remote_version()
    if not remote:
        return False
    if _parse_version(remote) <= _parse_version(CLIENT_VERSION):
        return False
    log.info(f"Update available: {CLIENT_VERSION} -> {remote}")
    new_code = _download_new_client(remote)
    if not new_code:
        return False
    if not _apply_update(new_code):
        return False
    log.info(f"client.py updated to {remote}")
    return True

def _restart_self():
    python = sys.executable
    script = str(Path(__file__).resolve())
    args = [python, script] + sys.argv[1:]
    if os.name == "nt":
        subprocess.Popen(args, close_fds=True, creationflags=0x00000008)
        sys.exit(0)
    else:
        os.execv(python, args)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SERVER = "https://server.crowdgpt.net"
DATASET_REPO_ID = "Vxtzq/CrowdGPT"
TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
HF_MODEL_REPO_ID = "Vxtzq/Crowd-v1"
USE_HF_WEIGHTS = True
UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024
MAX_DIRECT_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_CHUNK_RETRY = 4

MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
}
VOCAB_SIZE = MODEL_CONFIG["vocabSize"]
DIM = MODEL_CONFIG["dim"]
N_LAYERS = MODEL_CONFIG["nLayers"]
N_HEADS = MODEL_CONFIG["nHeads"]
N_KV_HEADS = MODEL_CONFIG["nKvHeads"]
HEAD_DIM = MODEL_CONFIG["headDim"]
MAX_SEQ_LEN = MODEL_CONFIG["maxSeqLen"]
MLP_HIDDEN = MODEL_CONFIG["mlpHidden"]
ENG_NUM_BUCKETS = 227865
LOSS_CHUNK = 256
UPLOAD_BUFFER_MIN = 25
ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM * 2 + DIM * (N_HEADS * HEAD_DIM) + DIM * (N_KV_HEADS * HEAD_DIM) * 2 + DIM * DIM + DIM * 2
        size += DIM * MLP_HIDDEN * 2 + MLP_HIDDEN * DIM
    size += DIM * 2
    return size

EXPECTED_MODEL_SIZE = _calc_model_size()
EXPECTED_ENGRAM_SIZE = ENG_NUM_BUCKETS * DIM
EXPECTED_MODEL_BYTES = EXPECTED_MODEL_SIZE * 2
EXPECTED_ENGRAM_BYTES = EXPECTED_ENGRAM_SIZE * 2
EXPECTED_WEIGHT_BYTES = EXPECTED_MODEL_BYTES + EXPECTED_ENGRAM_BYTES
memory_config = {"ram_gb": 12, "is_auto_detected": False}
train_device, train_backend = None, None

def safe_float(v, default=0.0):
    try:
        if v is None or math.isnan(v) or math.isinf(v):
            return default
        return float(v)
    except Exception:
        return default

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0.0
        return obj
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return 0.0 if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)

TOKENIZER = None
_tokenizer_started = False
PREVIEW_EVERY = 10

def start_tokenizer_loader():
    global _tokenizer_started
    if _tokenizer_started:
        return
    _tokenizer_started = True
    threading.Thread(target=_load_tokenizer_worker, daemon=True).start()

def _load_tokenizer_worker():
    global TOKENIZER
    try:
        from transformers import AutoTokenizer
        candidates = [TOKENIZER_REPO, "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "Vxtzq/Crowd-v1"]
        for repo in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
                if len(tok) >= 151000:
                    TOKENIZER = tok
                    log.info(f"Tokenizer loaded: {repo} ({len(tok)} tokens)")
                    return
            except Exception:
                continue
        log.warning("No compatible tokenizer loaded.")
    except ImportError:
        log.warning("transformers not installed.")
    TOKENIZER = None

def decode_ids(ids, max_chars=220):
    try:
        ids = [int(i) for i in ids]
        if TOKENIZER is not None:
            s = TOKENIZER.decode(ids, skip_special_tokens=True)
            s = ' '.join(s.split())
            if len(s) > max_chars:
                s = s[:max_chars] + "…"
            return s or "…"
    except Exception:
        pass
    s = ' '.join(f"#{int(i)}" for i in ids)
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s

def decode_one(tok_id):
    s = decode_ids([tok_id], max_chars=32)
    return s if s.strip() else "␠"

# =========================================================================
# Logos + window icon
# =========================================================================
LOGO_URI_LIGHT = ""
LOGO_URI_DARK = ""
_lp_light = Path(__file__).parent / "docs/logo-black.svg"
_lp_app   = Path(__file__).parent / "docs/logo-app.svg"
_lp_dark  = Path(__file__).parent / "docs/logo-white.svg"

if _lp_light.exists():
    try:
        LOGO_URI_LIGHT = "data:image/svg+xml;base64," + base64.b64encode(_lp_light.read_bytes()).decode()
    except Exception:
        pass
if _lp_dark.exists():
    try:
        LOGO_URI_DARK = "data:image/svg+xml;base64," + base64.b64encode(_lp_dark.read_bytes()).decode()
    except Exception:
        pass
if not LOGO_URI_DARK:
    LOGO_URI_DARK = LOGO_URI_LIGHT

# --- icon plumbing -------------------------------------------------------
ICON_PATH = None          # raster path handed to pywebview (PNG / ICO / etc.)
WINDOWS_ICO_PATH = None   # real .ico used by the native Win32 icon swap
WINDOW_TITLE = "CrowdGPT"
_cleanup_icon = []

def _render_svg_to_png(svg_path, size: int = 256):
    """
    Render an SVG to a PNG using whichever backend is available.
    Returns path to a temp PNG, or None. Never raises.
    Backends tried, in order:
      1. cairosvg          (best on Linux/macOS; needs Cairo DLLs on Windows)
      2. svglib+reportlab  (pure Python — recommended on Windows)
      3. resvg-py          (Rust wheel, fast, works on Windows)
      4. Qt (PySide6/PyQt5, ships with pywebview on Windows)
      5. ImageMagick CLI
      6. Inkscape CLI
    """
    svg_path = Path(svg_path)
    if not svg_path.exists():
        return None

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    out = tmp.name

    def _ok():
        try:
            return os.path.getsize(out) > 100
        except Exception:
            return False

    # --- 1. cairosvg ----------------------------------------------------
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=out,
                         output_width=size, output_height=size)
        if _ok():
            log.info("[icon] Rendered SVG via cairosvg")
            return out
    except Exception as e:
        log.debug(f"[icon] cairosvg failed: {e}")

    # --- 2. svglib + reportlab (pure Python — best on Windows) ----------
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(str(svg_path))
        if drawing is not None:
            sw, sh = float(drawing.width or 0), float(drawing.height or 0)
            if sw > 0 and sh > 0:
                scale = size / max(sw, sh)
                drawing.scale(scale, scale)
                drawing.width = sw * scale
                drawing.height = sh * scale
            renderPM.drawToFile(drawing, out, fmt='PNG')
            if _ok():
                log.info("[icon] Rendered SVG via svglib+reportlab")
                return out
    except Exception as e:
        log.debug(f"[icon] svglib failed: {e}")

    # --- 3. resvg-py ----------------------------------------------------
    try:
        import resvg_py
        resvg_py.svg_to_bytes(svg_path=str(svg_path), write_to=out,
                              width=size, height=size)
        if _ok():
            log.info("[icon] Rendered SVG via resvg-py")
            return out
    except Exception as e:
        log.debug(f"[icon] resvg-py failed: {e}")

    # --- 4. Qt (PySide6 / PyQt5) ---------------------------------------
    for mod in ("PySide6", "PyQt5"):
        try:
            if mod == "PySide6":
                from PySide6.QtSvg import QSvgRenderer
                from PySide6.QtGui import QImage, QPainter
                from PySide6.QtCore import QByteArray
            else:
                from PyQt5.QtSvg import QSvgRenderer
                from PyQt5.QtGui import QImage, QPainter
                from PyQt5.QtCore import QByteArray
            renderer = QSvgRenderer(QByteArray(svg_path.read_bytes()))
            if not renderer.isValid():
                continue
            img = QImage(size, size, QImage.Format_ARGB32)
            img.fill(0)
            painter = QPainter(img)
            renderer.render(painter)
            painter.end()
            if img.save(out, "PNG") and _ok():
                log.info(f"[icon] Rendered SVG via {mod}")
                return out
        except Exception as e:
            log.debug(f"[icon] {mod} failed: {e}")

    # --- 5. ImageMagick CLI --------------------------------------------
    try:
        r = subprocess.run(
            ["magick", "-background", "none", str(svg_path),
             "-resize", f"{size}x{size}", out],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and _ok():
            log.info("[icon] Rendered SVG via ImageMagick")
            return out
    except Exception as e:
        log.debug(f"[icon] magick failed: {e}")

    # --- 6. Inkscape CLI -----------------------------------------------
    try:
        r = subprocess.run(
            ["inkscape", str(svg_path),
             "-w", str(size), "-h", str(size), "-o", out],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0 and _ok():
            log.info("[icon] Rendered SVG via Inkscape")
            return out
    except Exception as e:
        log.debug(f"[icon] inkscape failed: {e}")

    # Nothing worked → cleanup and return None
    try:
        os.unlink(out)
    except Exception:
        pass
    return None


def _png_to_ico(png_path: str, sizes=(256, 128, 64, 48, 32, 16)):
    """Convert PNG to a multi-size ICO. Returns path or None. Never raises."""
    try:
        from PIL import Image
    except ImportError:
        log.warning("[icon] Pillow not installed — cannot build .ico for Windows. "
                    "Install with: pip install pillow")
        return None
    try:
        img = Image.open(png_path).convert("RGBA")
        tmp = tempfile.NamedTemporaryFile(suffix='.ico', delete=False)
        tmp.close()
        img.save(tmp.name, format="ICO", sizes=[(s, s) for s in sizes])
        return tmp.name
    except Exception as e:
        log.warning(f"[icon] PNG→ICO failed: {e}")
        return None


def _cleanup_icon_files():
    for p in _cleanup_icon:
        try:
            os.unlink(p)
        except Exception:
            pass
atexit.register(_cleanup_icon_files)


def _find_best_icon():
    """
    Resolution order (first hit wins):
      1. icon.ico next to the script  (explicit user override, best on Windows)
      2. docs/icon.ico                (explicit user override)
      3. docs/logo-app.svg            (rendered via whatever SVG engine is
                                       available — this is the *correct* app
                                       logo, so it beats any stray .png)
      4. icon.png next to the script
      5. docs/logo-app.png
    Returns (raster_path_for_webview, ico_path_for_windows_or_None).
    Never raises. Never tries to open a directory.
    """
    base = Path(__file__).parent

    ico_candidates = [base / "icon.ico", base / "docs" / "icon.ico"]
    svg_candidates = [base / "docs" / "logo-app.svg",
                      base / "logo-app.svg"]
    png_candidates = [base / "icon.png",
                      base / "docs" / "logo-app.png"]

    # 1/2. Direct .ico — perfect on Windows, accepted by pywebview elsewhere.
    for ico in ico_candidates:
        if ico.is_file():
            log.info(f"[icon] Using existing .ico: {ico}")
            return str(ico), str(ico)

    # 3. Render the SVG. Preferred over PNG because logo-app.svg is the
    #    canonical app icon and any random .png may be the wrong image.
    for svg in svg_candidates:
        if svg.is_file():
            rendered = _render_svg_to_png(svg, 256)
            if rendered:
                png_path = Path(rendered)
                _cleanup_icon.append(rendered)
                raster = str(png_path)
                ico_path = None
                if sys.platform == "win32":
                    ico_path = _png_to_ico(raster)
                    if ico_path:
                        _cleanup_icon.append(ico_path)
                        log.info(f"[icon] Built multi-size ICO: {ico_path}")
                    else:
                        log.warning("[icon] SVG rendered but PNG→ICO failed. "
                                    "Install Pillow: pip install pillow")
                return raster, ico_path
            # Non-Windows can load SVG directly.
            if sys.platform != "win32":
                log.info(f"[icon] No SVG renderer found — using SVG directly: {svg}")
                return str(svg), None
            log.warning(
                "[icon] Could not render SVG. Install one of:\n"
                "    pip install svglib reportlab    (recommended on Windows)\n"
                "    pip install cairosvg            (Linux/macOS)\n"
                "    pip install resvg-py\n"
                "  Or drop a real icon.ico next to client.py."
            )
            break  # don't repeat the warning for every svg candidate

    # 4/5. Existing PNG.
    for png in png_candidates:
        if png.is_file():
            raster = str(png)
            ico_path = None
            if sys.platform == "win32":
                ico_path = _png_to_ico(raster)
                if ico_path:
                    _cleanup_icon.append(ico_path)
                    log.info(f"[icon] Built multi-size ICO: {ico_path}")
                else:
                    log.warning("[icon] Could not build .ico — window will "
                                "use the default. Install Pillow: pip install pillow")
            log.info(f"[icon] Using PNG: {png}")
            return raster, ico_path

    log.warning("[icon] No icon source found — window will use the default.")
    return None, None


ICON_PATH, WINDOWS_ICO_PATH = _find_best_icon()
# =========================================================================
# Windows native icon swap
# =========================================================================
def set_windows_icon(window_title: str, ico_path: str):
    """
    Set the titlebar + taskbar icon of the top-level window `window_title`.
    Only runs on Windows and only accepts a real .ico file.
    """
    if sys.platform != "win32" or not ico_path:
        return
    ico_path = os.path.abspath(ico_path)
    if not os.path.exists(ico_path):
        log.warning(f"[icon] not found: {ico_path}")
        return
    if not ico_path.lower().endswith(".ico"):
        log.warning(f"[icon] not an .ico file: {ico_path}")
        return

    # Explicit argtypes prevent 64-bit pointer truncation.
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes   = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype    = ctypes.c_void_p
    user32.LoadImageW.argtypes    = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.LoadImageW.restype     = ctypes.c_void_p
    user32.SendMessageW.argtypes  = [ctypes.c_void_p, ctypes.c_uint,
                                     ctypes.c_void_p, ctypes.c_void_p]
    user32.SendMessageW.restype   = ctypes.c_void_p
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype  = ctypes.c_int

    hwnd = user32.FindWindowW(None, window_title)
    if not hwnd:
        log.warning(f"[icon] window '{window_title}' not found")
        return

    IMAGE_ICON      = 1
    LR_LOADFROMFILE = 0x0010
    WM_SETICON      = 0x0080
    ICON_SMALL      = 0
    ICON_BIG        = 1
    SM_CXSMICON     = 49
    SM_CYSMICON     = 50

    cx = user32.GetSystemMetrics(SM_CXSMICON) or 16
    cy = user32.GetSystemMetrics(SM_CYSMICON) or 16

    hicon_small = user32.LoadImageW(None, ico_path, IMAGE_ICON, cx, cy, LR_LOADFROMFILE)
    # (0, 0) with no LR_DEFAULTSIZE → LoadImage picks the largest icon in the file.
    hicon_big   = user32.LoadImageW(None, ico_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE)

    if hicon_small:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
    if hicon_big:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)

    if not hicon_small and not hicon_big:
        err = ctypes.get_last_error()
        log.warning(f"[icon] LoadImageW failed (err={err}) for {ico_path}")
    else:
        log.info(f"[icon] Applied window icon: {ico_path}")

def _wait_for_window(title: str, timeout: float = 5.0) -> bool:
    """Poll for the top-level window to exist."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype  = ctypes.c_void_p
    deadline = time.time() + timeout
    while time.time() < deadline:
        if user32.FindWindowW(None, title):
            return True
        time.sleep(0.1)
    return False

# =========================================================================
# Backend detection
# =========================================================================
def get_available_backends():
    backends = {}
    backends['cuda'] = torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip)
    backends['rocm'] = hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available()
    backends['mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    try:
        import intel_extension_for_pytorch
        backends['xpu'] = hasattr(torch, 'xpu') and torch.xpu.is_available()
    except ImportError:
        backends['xpu'] = False
    try:
        import torch_directml
        backends['directml'] = torch_directml.is_available()
    except ImportError:
        backends['directml'] = False
    backends['cpu'] = True
    return backends

def get_best_default_backend():
    avail = get_available_backends()
    for bk in ['cuda', 'rocm', 'mps', 'xpu', 'cpu']:
        if avail.get(bk):
            return bk
    return 'cpu'

def detect_training_backend(force=None):
    if force and force != "auto":
        n = force.lower()
        if n == "cpu":
            return torch.device('cpu'), "CPU"
        if n == "cuda" and torch.cuda.is_available():
            return torch.device('cuda'), "CUDA"
        if n == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
            return torch.device('cuda'), "ROCM"
        if n == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps'), "MPS"
        if n == "xpu":
            try:
                import intel_extension_for_pytorch
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    return torch.device('xpu'), "XPU"
            except ImportError:
                pass
        if n == "directml":
            try:
                import torch_directml
                if torch_directml.is_available():
                    return torch_directml.device(0), "DIRECTML"
            except ImportError:
                pass
    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip):
        return torch.device('cuda'), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
        return torch.device('cuda'), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), "MPS"
    try:
        import intel_extension_for_pytorch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu'), "XPU"
    except ImportError:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(0), "DIRECTML"
    except ImportError:
        pass
    return torch.device('cpu'), "CPU"

def auto_detect_vram_budget():
    global train_backend
    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            fb, tb = torch.cuda.mem_get_info(0)
            memory_config.update({"ram_gb": round(max(1.0, min(fb / 1024 ** 3, tb / 1024 ** 3 - 2.0) - 1.0), 2), "is_auto_detected": True})
            return
        except Exception:
            pass
    if train_backend == "MPS":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = int(out) / 1024 ** 3
            memory_config.update({"ram_gb": round(max(1.0, total_gb * 0.75 - 2.0), 2), "is_auto_detected": True})
            return
        except Exception:
            pass
    try:
        import psutil
        memory_config.update({"ram_gb": round(max(1.0, psutil.virtual_memory().available / 1024 ** 3 - 2.0), 2), "is_auto_detected": True})
    except Exception:
        pass

def has_bitsandbytes():
    try:
        import bitsandbytes
        return True
    except Exception:
        return False

def estimate_vram_bytes(bs, sl, u8):
    mb = EXPECTED_MODEL_SIZE * 2
    ob = EXPECTED_MODEL_SIZE * (4 if u8 else 12)
    gb = EXPECTED_MODEL_SIZE * 4
    ab = N_LAYERS * (bs * sl * DIM * 2 + bs * N_HEADS * 64 * sl * 4)
    lb = bs * LOSS_CHUNK * VOCAB_SIZE * 4
    eb = bs * sl * DIM * 2
    return int(mb + ob + gb + ab + lb + eb + 0.8 * 1024 ** 3)

def recommend_batch_size(sl=2048):
    u8 = has_bitsandbytes()
    budget = int(memory_config["ram_gb"] * 1024 ** 3)
    best = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(bs, sl, u8) <= budget:
            best = bs
        else:
            break
    return best

def precompute_freqs(dim, sl, dev):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)

def _is_directml(t):
    try:
        return t.device.type == 'privateuseone'
    except Exception:
        return False

def _is_directml_device(d):
    try:
        return d.type == 'privateuseone'
    except Exception:
        return False

def _is_accel_error(msg):
    """Detect DirectML/CUDA errors that warrant a fallback to CPU."""
    m = msg.lower()
    return any(k in m for k in (
        "privateuse1", "unbox expects", "not implemented for",
        "not supported", "internal assert", "dml",
        "scatter", "compilegraph", "out of memory", "allocation",
        "device_type", "not implemented for unique",
    ))

def _plan_gpu_layers(device, backend_name, ram_gb):
    """
    Decide how many transformer blocks to place on the accelerator.
    Everything else (embeddings, lm_head, remaining blocks) stays on CPU.
    """
    try:
        if backend_name == "CPU" or device.type == 'cpu':
            return 0
    except Exception:
        return 0

    ram_bytes = int(max(0.5, ram_gb) * 1024**3)
    budget = int(ram_bytes * 0.80)  # 20% safety margin

    params_per_block = (
        DIM * 2
        + DIM * (N_HEADS * HEAD_DIM)
        + DIM * (N_KV_HEADS * HEAD_DIM) * 2
        + DIM * DIM
        + DIM * 2
        + DIM * MLP_HIDDEN * 2
        + MLP_HIDDEN * DIM
    )
    # fp32 weights + fp32 grad + fp32 Adam m + fp32 Adam v = 4× param count × 4 bytes
    bytes_per_block = params_per_block * 4 * 4

    if backend_name == "DIRECTML":
        # DirectML is functional but fragile on this model.
        # Cap at 2 blocks and rely on runtime fallback if anything breaks.
        n_fit = int(budget // bytes_per_block)
        n = max(0, min(2, min(N_LAYERS, n_fit)))
        log.info(f"[PLACEMENT] DirectML: budget {ram_gb:.1f} GB → {n} block(s) on GPU")
        return n

    n_fit = int(budget // bytes_per_block)
    n_fit = max(0, min(N_LAYERS, n_fit))
    if n_fit < 1:
        log.warning(f"[PLACEMENT] {backend_name}: budget {ram_gb:.1f} GB too small for any block on GPU")
        return 0
    log.info(f"[PLACEMENT] {backend_name}: budget {ram_gb:.1f} GB → {n_fit} block(s) on GPU")
    return n_fit

class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = N_HEADS
        self.nkv = N_KV_HEADS
        self.nrep = N_HEADS // N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, cos, sin, use_chunked=True):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.nh, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        ct, st = cos[:, :, :T, :], sin[:, :, :T, :]
        q = q * ct + rotate_half(q) * st
        k = k * ct + rotate_half(k) * st
        k = k.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        v = v.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        if use_chunked:
            return self._chunked(q, k, v, B, T, C)
        a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
        a = a.masked_fill(torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, T, C))

    def _chunked(self, q, k, v, B, T, C, cs=64):
        # Some backends (DirectML) cannot materialize expanded views into CPU memory.
        # Force .contiguous() before moving to CPU to avoid that.
        use_cpu = _is_directml(q)
        if use_cpu:
            q_ = q.contiguous().cpu()
            k_ = k.contiguous().cpu()
            v_ = v.contiguous().cpu()
        else:
            q_, k_, v_ = q, k, v
        chunks = []
        for i in range(0, T, cs):
            e = min(i + cs, T)
            aw = (q_[:, :, i:e, :] @ k_[:, :, :e, :].transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            ri = torch.arange(i, e, device=q_.device).unsqueeze(1)
            ci = torch.arange(e, device=q_.device).unsqueeze(0)
            aw = aw.masked_fill(~(ci <= ri).unsqueeze(0).unsqueeze(0), float('-inf'))
            chunks.append(F.softmax(aw, dim=-1, dtype=torch.float32).to(q_.dtype) @ v_[:, :, :e, :])
        out = torch.cat(chunks, dim=2).transpose(1, 2).contiguous().view(B, T, C)
        if use_cpu:
            out = out.to(q.device)
        return self.wo(out)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w2 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w3 = nn.Linear(MLP_HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(DIM)
        self.attn = GroupedQueryAttention()
        self.ln_2 = nn.LayerNorm(DIM)
        self.mlp = SwiGLU()

    def forward(self, x, cos, sin, use_chunked=True):
        x = x + self.attn(self.ln_1(x), cos, sin, use_chunked)
        return x + self.mlp(self.ln_2(x))

class EngramMemory(nn.Module):
    """Always runs on CPU. Output is moved to `device` for the caller."""
    def __init__(self, bn, device):
        super().__init__()
        self.device = device
        self.table = nn.Embedding(ENG_NUM_BUCKETS, DIM, sparse=True).to('cpu')
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # Everything CPU-side, then a single transfer of the result.
        idx_cpu = idx.detach().cpu()
        px = torch.cat([torch.zeros_like(idx_cpu[:, :1]), idx_cpu[:, :-1]], dim=1)
        hi = (px * 1000003 + idx_cpu) % ENG_NUM_BUCKETS
        uq, inv = torch.unique(hi.reshape(-1), return_inverse=True)
        cr = self.table(uq)
        out = cr[inv].view(idx.shape[0], idx.shape[1], DIM)
        return out.to(self.device)

class SotaGPT(nn.Module):
    """
    Layer-offloaded transformer. `gpu_layers` = number of leading blocks
    placed on `device`. Everything else (embeddings, remaining blocks,
    lm_head, ln_f) stays on CPU.
    """
    def __init__(self, bn, device, gpu_layers=None):
        super().__init__()
        self.main_device = device
        self.cpu_device = torch.device('cpu')
        self.backend_name = bn

        if gpu_layers is None:
            gpu_layers = _plan_gpu_layers(device, bn, memory_config["ram_gb"])
        self.gpu_layers = max(0, min(N_LAYERS, int(gpu_layers)))
        self.gpu_layer_set = set(range(self.gpu_layers))
        self.all_on_gpu = (self.gpu_layers == N_LAYERS) and (device.type != 'cpu')
        self.any_on_gpu = (self.gpu_layers > 0) and (device.type != 'cpu')

        self.wte = nn.Embedding(VOCAB_SIZE, DIM)
        self.engram = EngramMemory(bn, device)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        if MODEL_CONFIG["weightTying"]:
            self.wte.weight = self.lm_head.weight

        cm_main, sm_main = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos_main", cm_main)
        self.register_buffer("freqs_sin_main", sm_main)

        cm_cpu, sm_cpu = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, self.cpu_device)
        self.register_buffer("freqs_cos_cpu", cm_cpu)
        self.register_buffer("freqs_sin_cpu", sm_cpu)

        for i, block in enumerate(self.blocks):
            if self.all_on_gpu or i in self.gpu_layer_set:
                block.to(device)
            else:
                block.to(self.cpu_device)

        if self.all_on_gpu:
            self.wte.to(device)
            self.ln_f.to(device)
            self.lm_head.to(device)
        else:
            # Head stays on CPU. Keeps VRAM free for block offloading
            # and avoids the huge vocab matmul living on the accelerator.
            self.wte.to(self.cpu_device)
            self.ln_f.to(self.cpu_device)
            self.lm_head.to(self.cpu_device)

        self.engram.table.to('cpu')

        log.info(
            f"🧠 SotaGPT: {self.gpu_layers}/{N_LAYERS} blocks on {device.type}, "
            f"head on {self.ln_f.weight.device.type}"
        )

    def _block_device(self, i):
        if self.all_on_gpu or i in self.gpu_layer_set:
            return self.main_device
        return self.cpu_device

    def forward_features(self, idx, use_chunked=True, use_ckpt=True):
        """Runs embeddings + all blocks + final LN. Returns [B, T, DIM]."""
        # wte can be on CPU while idx arrives on DirectML. torch-directml
        # can't mix CPU weight + DML index, so normalise devices first.
        wte_dev = self.wte.weight.device
        if idx.device != wte_dev:
            idx = idx.to(wte_dev)
        x = self.wte(idx)
        first_dev = self._block_device(0) if len(self.blocks) else self.cpu_device
        if x.device != first_dev:
            x = x.to(first_dev)
        cur_dev = x.device

        for i, block in enumerate(self.blocks):
            target = self._block_device(i)
            if target != cur_dev:
                x = x.to(target)
                cur_dev = target

            if cur_dev == self.cpu_device:
                cos, sin = self.freqs_cos_cpu, self.freqs_sin_cpu
            else:
                cos, sin = self.freqs_cos_main, self.freqs_sin_main

            if use_ckpt:
                x = checkpoint(block, x, cos, sin, use_chunked, use_reentrant=False)
            else:
                x = block(x, cos, sin, use_chunked)

        head_dev = self.ln_f.weight.device
        if x.device != head_dev:
            x = x.to(head_dev)
        return self.ln_f(x)

    def forward(self, idx, use_chunked=True, use_ckpt=True):
        x = self.forward_features(idx, use_chunked, use_ckpt)
        return self.lm_head(x)

    def to_cpu_only(self):
        """Hard fallback: put everything on CPU. Optimizer references stay valid."""
        self.gpu_layers = 0
        self.gpu_layer_set = set()
        self.all_on_gpu = False
        self.any_on_gpu = False
        self.main_device = self.cpu_device
        for block in self.blocks:
            block.to(self.cpu_device)
        self.wte.to(self.cpu_device)
        self.ln_f.to(self.cpu_device)
        self.lm_head.to(self.cpu_device)
        self.engram.table.to('cpu')
        self.engram.device = self.cpu_device

    def get_base_weights(self):
        return np.concatenate([
            p.detach().float().flatten().cpu().numpy()
            for n, p in self.named_parameters() if not n.startswith('engram.')
        ])

    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = 1
                for d in p.shape:
                    s *= int(d)
                p.data.copy_(ft[o:o + s].view(p.shape).to(p.device))
                o += s

def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for a in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                ChunkedEncodingError, http.client.IncompleteRead):
            if a < retries - 1:
                time.sleep(2 ** (a + 1))
            else:
                raise

class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500

    def __init__(self, repo_id, ci, ss=10 * 1024 * 1024, tps=65, slot=0, at=None):
        self.repo_id = repo_id
        self.ci = ci
        self.sub_size = ss
        self.tps = tps
        self.auth_token = at
        self._fmt = "chunk_{:04d}.bin"
        self.chunk_size = self._discover(ci)
        self.off = (slot * ss) % self.chunk_size
        self.data = None
        self.n = 0
        self.steps_used = 0
        self._pft = None
        self._pfr = None
        self._lock = threading.Lock()
        self._load(self.off)
        self._prefetch(self.off + ss)

    def _url(self):
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + self._fmt.format(self.ci)

    def _discover(self, idx):
        for fmt in ("chunk_{:04d}.bin", "chunk_{:d}.bin"):
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + fmt.format(idx)
            for _ in range(2):
                try:
                    r = requests.head(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200:
                        sz = int(r.headers.get('content-length', 0))
                        if sz > 0:
                            self._fmt = fmt
                            return sz
                except Exception:
                    time.sleep(1)
        raise RuntimeError(f"No chunk {idx}")

    def _slice(self, off):
        end = min(off + self.sub_size, self.chunk_size) - 1
        return _fetch_with_retry(self._url(), headers={'Range': f'bytes={off}-{end}'}, timeout=120).content

    def _set(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32)
        self.data = tk
        self.n = len(tk) // self.tps
        self.steps_used = 0

    def _load(self, off):
        self._set(self._slice(off))
        self.off = off

    def _prefetch(self, off):
        if off >= self.chunk_size:
            return
        with self._lock:
            self._pfr = None

        def w():
            try:
                r = self._slice(off)
                with self._lock:
                    self._pfr = (off, r)
            except Exception:
                pass
        self._pft = threading.Thread(target=w, daemon=True)
        self._pft.start()

    def needs_new_subchunk(self):
        return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, srv=None, fmt="bf16"):
        no = self.off + self.sub_size
        if no < self.chunk_size:
            if self._pft:
                self._pft.join(timeout=120)
                self._pft = None
            with self._lock:
                r = self._pfr
                self._pfr = None
            if r and r[0] == no:
                self.off = no
                self._set(r[1])
            else:
                self._load(no)
            self._prefetch(no + self.sub_size)
            return True
        self._req(srv, fmt)
        self.chunk_size = self._discover(self.ci)
        self.off = 0
        self._load(0)
        self._prefetch(self.sub_size)
        return True

    def _req(self, srv, fmt="bf16"):
        if not srv:
            return False
        try:
            h = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(f"{srv}/fl/task?format={fmt}&skip_weights=true", headers=h, timeout=30)
            if r.status_code != 200:
                return False
            ct = (r.headers.get("content-type") or "").lower()
            if "json" in ct:
                md = r.json()
            else:
                raw = r.content
                if len(raw) < 4:
                    return False
                ml = struct.unpack('<I', raw[:4])[0]
                if len(raw) < 4 + ml:
                    return False
                md = json.loads(raw[4:4 + ml].decode())
            ni = md.get("datasetConfig", {}).get("chunkIdx", self.ci)
            if ni != self.ci:
                self.ci = ni
                return True
        except Exception:
            pass
        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0:
            self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed)
        s = rng.randint(0, self.n, size=bs) * self.tps
        inp = np.stack([self.data[x:x + self.tps - 1] for x in s])
        tgt = np.stack([self.data[x + 1:x + self.tps] for x in s])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def decompress_weights(raw, fmt="bf16"):
    if fmt == "fp16":
        return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()

def do_login(srv, u, p):
    try:
        r = requests.post(f"{srv}/auth/login", json={"username": u, "password": p}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token"), None
        try:
            msg = r.json().get("detail", "")
        except Exception:
            msg = r.text
        if "invalid" in msg.lower() or "credentials" in msg.lower():
            return None, "invalid_credentials"
        return None, "login_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except Exception:
        return None, "network"

def do_register(srv, u, email, p):
    try:
        r = requests.post(f"{srv}/auth/register", json={"username": u, "password": p, "email": email}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token"), None
        try:
            msg = r.json().get("detail", "")
        except Exception:
            msg = r.text
        msg_l = msg.lower()
        if "username" in msg_l and "taken" in msg_l:
            return None, "username_taken"
        if "email" in msg_l and "registered" in msg_l:
            return None, "email_taken"
        if "password" in msg_l:
            return None, "weak_password"
        if "invalid" in msg_l:
            return None, "invalid_input"
        return None, "register_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except Exception:
        return None, "network"

class HeartbeatManager:
    def __init__(self, srv, h, cr, se, emit):
        self.srv = srv
        self.h = h
        self.cr = cr
        self.stop_training = threading.Event()
        self.shutdown = threading.Event()
        self.se = se
        self.emit = emit
        self._t = threading.Thread(target=self._r, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self.shutdown.set()

    def should_stop(self):
        return self.stop_training.is_set()

    def _r(self):
        while not self.shutdown.is_set() and not self.se.is_set():
            try:
                requests.get(f"{self.srv}/fl/heartbeat", headers=self.h, timeout=10)
                r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                if r.status_code == 200:
                    st = r.json()
                    if st.get("current_round", self.cr) != self.cr:
                        self.emit('log', "Next round. Stopping.")
                        self.stop_training.set()
                    rm = (st.get("max_round_hours", 2) - st.get("round_elapsed_hours", 0)) * 60
                    if rm <= 3:
                        self.emit('log', f"ULTIMATUM: {rm:.0f} min left!")
                        self.stop_training.set()
            except Exception:
                pass
            self.shutdown.wait(timeout=15)

def wait_for_round(srv, h, se, emit):
    overlay_on = False

    def show_wait():
        nonlocal overlay_on
        if not overlay_on:
            emit('overlay_show', {'title_key': 'wait_coord', 'subtitle_key': 'wait_coord_sub', 'indeterminate': True})
            overlay_on = True

    def hide():
        nonlocal overlay_on
        if overlay_on:
            emit('overlay_hide', {})
            overlay_on = False

    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/round_status", headers=h, timeout=30)
            if r.status_code == 200:
                st = r.json()
                if st.get("is_aggregating"):
                    emit('status', 'waiting')
                    show_wait()
                    for _ in range(30):
                        if se.is_set():
                            hide()
                            return None
                        time.sleep(1)
                    continue
                if not st.get("in_cooldown", False):
                    hide()
                    return st
                emit('status', 'waiting')
                show_wait()
                for _ in range(30):
                    if se.is_set():
                        hide()
                        return None
                    time.sleep(1)
            else:
                time.sleep(10)
        except Exception:
            time.sleep(10)
    hide()
    return None

def fetch_task_metadata(srv, h, se, emit):
    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/task?format=bf16&skip_weights=true", headers=h, timeout=30)
            if r.headers.get("X-Status") == "wait":
                for _ in range(10):
                    if se.is_set():
                        return None
                    time.sleep(1)
                continue
            if r.status_code == 200:
                ct = (r.headers.get("content-type") or "").lower()
                if "json" in ct:
                    md = r.json()
                    if md.get("status") == "wait":
                        for _ in range(10):
                            if se.is_set():
                                return None
                            time.sleep(1)
                        continue
                    return md
                raw = r.content
                if len(raw) >= 4:
                    ml = struct.unpack('<I', raw[:4])[0]
                    if len(raw) >= 4 + ml:
                        return json.loads(raw[4:4 + ml].decode())
                raise RuntimeError("Invalid task metadata response")
            if r.status_code in (503, 504):
                time.sleep(10)
                continue
            time.sleep(5)
        except Exception as e:
            if se.is_set():
                return None
            emit('log', f"Task metadata fetch failed: {e}")
            time.sleep(10)
    return None

def _weight_cache_path(revision, filename):
    rev = str(revision or "main").strip().replace("/", "_").replace("\\", "_")
    d = CHECKPOINT_DIR / "hf_weights" / rev
    d.mkdir(parents=True, exist_ok=True)
    return d / filename

def _download_file_with_progress(url, dest, emit, stop_evt, expected_size=None, label="file"):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        sz = dest.stat().st_size
        if expected_size is not None and sz == int(expected_size):
            log.info(f"Using cached weight file: {dest} ({sz} bytes)")
            return dest
        if expected_size is None and sz > 10_000_000:
            log.info(f"Using cached weight file: {dest} ({sz} bytes, unverified)")
            return dest
        try:
            dest.unlink()
        except Exception:
            pass
    last_err = None
    for attempt in range(3):
        if stop_evt.is_set():
            raise KeyboardInterrupt("Stop requested")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        try:
            log.info(f"Downloading {label}: {url}")
            r = requests.get(url, stream=True, timeout=(15, 3600), allow_redirects=True)
            r.raise_for_status()
            content_length = int(r.headers.get("content-length", 0) or 0)
            total = int(expected_size) if expected_size else content_length
            done = 0
            last_emit = 0.0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if stop_evt.is_set():
                        raise KeyboardInterrupt("Stop requested")
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_emit > 0.25:
                        emit("overlay_progress", {"done": done, "total": total})
                        last_emit = now
            if expected_size and done != int(expected_size):
                raise RuntimeError(f"Size mismatch for {label}: got {done}, expected {expected_size}")
            os.replace(tmp, dest)
            log.info(f"Finished downloading {label}: {dest}")
            return dest
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if attempt < 2:
                emit("log", f"Download retry {attempt + 1} for {label}: {e}")
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {label}: {last_err}")

def _load_safetensors_flat(path, expected_numel, label="weights"):
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise RuntimeError("Missing dependency: safetensors.") from e
    tensors = load_file(str(path))
    if "weights" in tensors:
        t = tensors["weights"]
    elif "flat" in tensors:
        t = tensors["flat"]
    elif len(tensors) == 1:
        t = next(iter(tensors.values()))
    else:
        raise RuntimeError(f"{label}: cannot identify tensor key. keys={list(tensors.keys())[:20]}")
    arr = t.to(torch.float32).contiguous().clone().cpu().numpy()
    del tensors, t
    gc.collect()
    if arr.size != int(expected_numel):
        raise RuntimeError(f"{label} size mismatch: got {arr.size}, expected {expected_numel}.")
    _ = float(arr[0]) + float(arr[-1])
    return arr

def fetch_weights_from_hf(md, se, emit):
    wcfg = md.get("weights") or {}
    if wcfg.get("source") != "huggingface":
        raise RuntimeError("Task metadata does not advertise Hugging Face weights.")
    revision = wcfg.get("revision") or "main"
    files = wcfg.get("files") or {}
    model_info = files.get("model") or {}
    engram_info = files.get("engram") or {}
    model_url = model_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/model_bf16.safetensors"
    engram_url = engram_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/engram_bf16.safetensors"
    model_size = model_info.get("size")
    engram_size = engram_info.get("size")
    if not model_size:
        try:
            hr = requests.head(model_url, allow_redirects=True, timeout=30)
            model_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            model_size = None
    if not engram_size:
        try:
            hr = requests.head(engram_url, allow_redirects=True, timeout=30)
            engram_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            engram_size = None
    model_path = _weight_cache_path(revision, "model_bf16.safetensors")
    engram_path = _weight_cache_path(revision, "engram_bf16.safetensors")
    emit("status", "downloading")
    emit("overlay_show", {"title_key": "dl_weights", "indeterminate": False})
    _download_file_with_progress(model_url, model_path, emit, se, expected_size=model_size, label="model_bf16.safetensors")
    _download_file_with_progress(engram_url, engram_path, emit, se, expected_size=engram_size, label="engram_bf16.safetensors")
    emit("overlay_hide", {})
    emit("status", "preparing")
    emit("log", "Loading weights from safetensors...")
    iw = _load_safetensors_flat(model_path, EXPECTED_MODEL_SIZE, "model weights")
    ie = _load_safetensors_flat(engram_path, EXPECTED_ENGRAM_SIZE, "engram weights")
    return iw, ie

def fetch_task(srv, h, prec, se, emit):
    while not se.is_set():
        try:
            emit('status', 'downloading')
            emit('overlay_show', {'title_key': 'dl_weights', 'indeterminate': False})
            r = requests.get(f"{srv}/fl/task?format={prec}", headers=h, timeout=(15, 3600), stream=True)
            if r.status_code != 200:
                r.close()
                emit('overlay_hide', {})
                emit('log', f"/fl/task HTTP {r.status_code}")
                time.sleep(10)
                continue
            if r.headers.get("X-Status") == "wait":
                r.close()
                emit('overlay_hide', {})
                for _ in range(10):
                    if se.is_set():
                        return None, None
                    time.sleep(1)
                continue
            cl_total = int(r.headers.get('content-length', 0) or 0)
            total = cl_total if cl_total > 0 else EXPECTED_WEIGHT_BYTES
            chunks = []
            done = 0
            last = 0.0
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if not chunk:
                    continue
                chunks.append(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 0.25:
                    emit('overlay_progress', {'done': done, 'total': total})
                    last = now
            r.close()
            emit('overlay_progress', {'done': done, 'total': total})
            emit('overlay_hide', {})
            emit('status', 'preparing')
            raw = b"".join(chunks)
            del chunks
            if len(raw) < 4:
                emit('log', f"/fl/task returned only {len(raw)} bytes; retrying")
                time.sleep(5)
                continue
            ml = struct.unpack('<I', raw[:4])[0]
            if 4 + ml > len(raw):
                emit('log', f"Truncated header; retrying")
                time.sleep(5)
                continue
            md = json.loads(raw[4:4 + ml].decode())
            wb = raw[4 + ml:]
            emit('log', f"/fl/task inline: metadata OK, weights={len(wb):,} bytes")
            if len(wb) < EXPECTED_WEIGHT_BYTES:
                emit('log', f"⚠️ inline truncated at {len(wb):,}; retrying")
                time.sleep(10)
                continue
            return md, wb
        except Exception as e:
            emit('overlay_hide', {})
            if se.is_set():
                return None, None
            emit('log', f"Task fetch failed: {e}")
            time.sleep(15)

def _chunk_ce(m, xs, ys):
    return F.cross_entropy(m.lm_head(xs).reshape(-1, VOCAB_SIZE), ys.reshape(-1))

def _fwl(m, x, y, sl, ua, ad, ls=1.0, preview=False):
    """
    Forward + loss. Uses model.forward_features to respect layer placement.
    Wraps compute in try/except so callers can fall back to CPU on failure.
    """
    use_ckpt = m.all_on_gpu and m.main_device.type == 'cuda'

    def fwd():
        return m.forward_features(x, use_chunked=True, use_ckpt=use_ckpt)

    nc = max(1, math.ceil(sl / LOSS_CHUNK))
    if ua:
        with torch.autocast(device_type='cuda', dtype=ad):
            xe = fwd()
            head_dev = m.lm_head.weight.device
            if xe.device != head_dev:
                xe = xe.to(head_dev)
            y_dev = y.to(head_dev)
            lo = sum(
                checkpoint(_chunk_ce, m, xe[:, i:i + LOSS_CHUNK, :], y_dev[:, i:i + LOSS_CHUNK], use_reentrant=False)
                for i in range(0, sl, LOSS_CHUNK)
            ) / nc
    else:
        xe = fwd()
        head_dev = m.lm_head.weight.device
        if xe.device != head_dev:
            xe = xe.to(head_dev)
        y_dev = y.to(head_dev)
        lo = sum(
            checkpoint(_chunk_ce, m, xe[:, i:i + LOSS_CHUNK, :], y_dev[:, i:i + LOSS_CHUNK], use_reentrant=False)
            for i in range(0, sl, LOSS_CHUNK)
        ) / nc

    cards = []
    if preview:
        try:
            with torch.no_grad():
                B, T, C = xe.shape
                items = []
                for b in range(min(B, 3)):
                    items.append((b, T - 1))
                for offset in (24, 48, 72):
                    p = T - offset - 1
                    if p >= 9:
                        items.append((0, p))
                items = items[:6]
                for b, p in items:
                    if p < 9 or p >= T:
                        continue
                    start = max(0, p - 9)
                    ctx_ids = x[b, start:p + 1].tolist()
                    target_id = int(y[b, p].item())
                    hidden = xe[b, p, :].unsqueeze(0).to(m.lm_head.weight.dtype)
                    logits = m.lm_head(hidden).float()
                    probs = torch.softmax(logits[0], dim=-1)
                    prob, pred = torch.max(probs, dim=0)
                    pred_id = int(pred.item())
                    cards.append({
                        'context': decode_ids(ctx_ids),
                        'target': decode_one(target_id),
                        'pred': decode_one(pred_id),
                        'match': bool(pred_id == target_id),
                        'prob': safe_float(prob.item()),
                        'target_id': int(target_id),
                        'pred_id': int(pred_id),
                    })
        except Exception as e:
            log.warning(f"Preview extraction failed: {e}")
            cards = []
    return lo * ls, cards

def _sha256_hex(data):
    return hashlib.sha256(data).hexdigest()

def _post_json_with_retry(url, payload, h, emit, se, timeout=60, retries=4):
    last_err = None
    for attempt in range(retries):
        if se.is_set():
            raise KeyboardInterrupt("Stop requested")
        try:
            r = requests.post(url, json=payload, headers=h, timeout=timeout)
            if r.status_code == 503:
                emit("log", "Server busy, waiting...")
                last_err = RuntimeError("server busy 503")
                time.sleep(10)
                continue
            r.raise_for_status()
            return r
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                emit("log", f"POST retry {attempt + 1}: {e}")
                time.sleep(min(20, 2 ** (attempt + 1)))
    raise RuntimeError(f"POST {url} failed: {last_err}")

def _chunked_submit_once(srv, h, compressed_data, meta, emit, se, chunk_size):
    total = len(compressed_data)
    chunks = max(1, math.ceil(total / chunk_size))
    checksum = _sha256_hex(compressed_data)
    init_payload = {**meta, "totalBytes": total, "chunkSize": chunk_size, "chunks": chunks, "checksum": checksum}
    emit("log", f"Initializing chunked upload: {chunks} chunks, {total / (1024 * 1024):.1f} MB total")
    r = _post_json_with_retry(f"{srv}/fl/submit/init", init_payload, h, emit, se, timeout=60, retries=4)
    j = r.json()
    upload_id = j.get("uploadId")
    if not upload_id:
        raise RuntimeError("Server did not return uploadId")
    for idx in range(chunks):
        if se.is_set():
            raise KeyboardInterrupt("Stop requested")
        start = idx * chunk_size
        end = min(total, start + chunk_size)
        part = compressed_data[start:end]
        part_sha = _sha256_hex(part)
        attempt = 0
        while True:
            try:
                rr = requests.post(
                    f"{srv}/fl/submit/chunk?uploadId={upload_id}&index={idx}",
                    data=part,
                    headers={**h, "Content-Type": "application/octet-stream", "X-Chunk-Sha256": part_sha},
                    timeout=300,
                )
                if rr.status_code == 413:
                    raise RuntimeError("CHUNK_TOO_LARGE")
                rr.raise_for_status()
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if "CHUNK_TOO_LARGE" in str(e):
                    raise
                attempt += 1
                if se.is_set() or attempt >= MAX_CHUNK_RETRY:
                    raise
                emit("log", f"Chunk {idx + 1}/{chunks} retry {attempt}: {e}")
                time.sleep(min(20, 2 ** attempt))
        emit("overlay_progress", {"done": end, "total": total})
        emit("log", f"  ✓ chunk {idx + 1}/{chunks} ({end / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)")
    emit("log", "Finalizing chunked upload...")
    return _post_json_with_retry(f"{srv}/fl/submit/finalize", {"uploadId": upload_id}, h, emit, se, timeout=900, retries=2)

def chunked_submit_delta(srv, h, compressed_data, meta, emit, se):
    chunk_size = UPLOAD_CHUNK_BYTES
    while True:
        try:
            return _chunked_submit_once(srv, h, compressed_data, meta, emit, se, chunk_size)
        except Exception as e:
            msg = str(e)
            if ("CHUNK_TOO_LARGE" in msg or "413" in msg) and chunk_size > 8 * 1024 * 1024:
                chunk_size = max(8 * 1024 * 1024, chunk_size // 2)
                emit("log", f"Chunk too large. Reducing to {chunk_size / (1024 * 1024):.0f} MB and retrying.")
                continue
            raise

def stream_with_progress(data, emit_cb, total_size, stop_evt):
    chunk_size = 128 * 1024
    uploaded = 0
    last_emit = 0.0
    for i in range(0, len(data), chunk_size):
        if stop_evt.is_set():
            raise KeyboardInterrupt("Stop requested")
        chunk = data[i:i + chunk_size]
        uploaded += len(chunk)
        now = time.time()
        if now - last_emit > 0.25:
            emit_cb("overlay_progress", {"done": uploaded, "total": total_size})
            last_emit = now
        yield chunk
    emit_cb("overlay_progress", {"done": total_size, "total": total_size})

def _build_model(bn, device, iw, ie, emit):
    """Build a SotaGPT with layer placement and load base + engram weights."""
    model = SotaGPT(bn, device)
    model.engram.table.to('cpu')

    try:
        print(f"[IW-PROBE] iw: shape={getattr(iw, 'shape', None)} size={getattr(iw, 'size', None)}", flush=True)
        print(f"[IW-PROBE] ie: shape={getattr(ie, 'shape', None)} size={getattr(ie, 'size', None)}", flush=True)
    except Exception:
        pass

    model.load_base_weights(iw)
    model.engram.table.weight.data.copy_(torch.from_numpy(ie).view(model.engram.table.weight.shape))
    model.engram.table.to('cpu')
    model.train()
    return model

def run_single_round_wrapper(srv, at, se, emit):
    global train_device, train_backend
    h = {"Authorization": f"Bearer {at}"} if at else {}
    emit('status', 'waiting')
    rs = wait_for_round(srv, h, se, emit)
    if not rs:
        return
    cr = rs["current_round"]
    hb = HeartbeatManager(srv, h, cr, se, emit)
    hb.start()
    model = ob = oe = x = y = lo = iw = ie = iew = ds = None
    micro_step = 0
    lval = 10.0
    tt = 0
    tid = ""
    tried_cpu_fallback = False
    try:
        emit('log', "Fetching task metadata...")
        md = fetch_task_metadata(srv, h, se, emit)
        if md is None:
            return
        tid = md.get("taskId")
        gs = md.get("globalStep")
        sl = MAX_SEQ_LEN
        if not tid:
            emit("log", "Task metadata missing taskId.")
            return
        iw = ie = None
        if USE_HF_WEIGHTS:
            wcfg = md.get("weights") or {}
            if wcfg.get("source") == "huggingface":
                try:
                    emit('log', "Downloading weights from Hugging Face...")
                    iw, ie = fetch_weights_from_hf(md, se, emit)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    import traceback as _tb
                    emit('log', f"Hugging Face weight download failed: {e}")
                    print(f"[HF-FAIL-TRACEBACK]\n{_tb.format_exc()}\n", flush=True)
                    emit('log', "Falling back to coordinator weight stream...")
                    iw = ie = None
        if iw is None or ie is None:
            emit('log', "Downloading weights from coordinator...")
            md2, wb = fetch_task(srv, h, "bf16", se, emit)
            if md2 is None:
                return
            md = md2
            tid = md.get("taskId")
            gs = md.get("globalStep")
            if not tid:
                emit("log", "Coordinator task metadata missing taskId.")
                return
            bbl = EXPECTED_MODEL_SIZE * 2
            ebl = EXPECTED_ENGRAM_SIZE * 2
            needed = bbl + ebl
            if len(wb) < needed:
                raise RuntimeError(f"Incomplete coordinator payload: got {len(wb)}, expected {needed}")
            iw = decompress_weights(wb[:bbl], "bf16")
            ie = decompress_weights(wb[bbl:bbl + ebl], "bf16")
            del wb
            gc.collect()

        emit('status', 'preparing')
        emit('log', "Preparing model...")
        dc = md.get("datasetConfig", {})
        sc = md.get("shardConfig", {})
        ds = StreamingShardDataset(
            dc.get("repoId", DATASET_REPO_ID),
            dc.get("chunkIdx", 0),
            dc.get("subChunkSize", 10 * 1024 * 1024),
            dc.get("tokensPerSample", sl + 1),
            sc.get("slot", 0),
            at,
        )
        bs = recommend_batch_size(sl)

        # ---- Build model with layer placement, retry on OOM ----
        while bs >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                gc.collect()
                model = _build_model(train_backend, train_device, iw, ie, emit)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if model:
                        del model
                        model = None
                    if train_backend in ("CUDA", "ROCM"):
                        try: torch.cuda.empty_cache()
                        except Exception: pass
                    gc.collect()
                    bs = max(1, bs // 2)
                else:
                    raise

        if model is None:
            emit('log', "Failed to initialize model.")
            return

        mbt = bs * sl
        as_ = max(1, round(131072 / mbt))
        ls = 1.0 / as_

        def ha():
            nonlocal as_, ls
            as_ = max(1, as_ // 2)
            ls = 1.0 / as_
            ob.zero_grad(set_to_none=True)
            gc.collect()
            if train_backend in ("CUDA", "ROCM"):
                try: torch.cuda.empty_cache()
                except Exception: pass
            emit('log', f"OOM -> accum now {as_}x")

        bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
        ep = list(model.engram.parameters())
        try:
            import bitsandbytes as bnb
            ob = bnb.optim.AdamW8bit(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        except Exception:
            ob = torch.optim.AdamW(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        try:
            oe = torch.optim.SparseAdam(ep, lr=1e-4)
        except Exception:
            oe = torch.optim.AdamW(ep, lr=1e-4, weight_decay=0.01)

        iew = model.engram.table.weight.data.cpu().clone()

        # Autocast only for CUDA + bf16 support
        ua = train_backend in ("CUDA", "ROCM") and torch.cuda.is_bf16_supported()
        ad = torch.bfloat16 if ua else None

        emit('log', f"Backend: {train_backend} | autocast: {ua} | "
                   f"bs={bs} | accum={as_} | dtype={'bf16' if ua else 'fp32'}")

        emit('log', "Calibrating speed...")
        cs_ = time.time()
        cal_global_steps = 3
        target_micro_steps = 100
        emit('status', 'calibrating')
        cal_done = 0
        while cal_done < cal_global_steps:
            if hb.should_stop() or se.is_set():
                return
            ob.zero_grad(set_to_none=True)
            try:
                for mi in range(as_):
                    if hb.should_stop() or se.is_set():
                        break
                    if ds.needs_new_subchunk():
                        ds.advance(srv, "bf16")
                    sd = (abs(hash("cal")) % 10000) + cal_done * 1000 + mi
                    x, y = ds.get_batch(bs, seed=sd)
                    x, y = x.to(train_device), y.to(train_device)
                    want_preview = (micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0)

                    # ---- DirectML/GPU runtime fallback ----
                    try:
                        lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                    except RuntimeError as re:
                        if (not tried_cpu_fallback) and _is_accel_error(str(re)):
                            emit('log', f"⚠️ {train_backend} op failed at fwd: {str(re)[:120]}")
                            emit('log', "🔻 Falling back to CPU-only model...")
                            model.to_cpu_only()
                            # Rebuild optimizer over same params — objects unchanged
                            bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
                            ep = list(model.engram.parameters())
                            try:
                                import bitsandbytes as bnb
                                ob = bnb.optim.AdamW8bit(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
                            except Exception:
                                ob = torch.optim.AdamW(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
                            try:
                                oe = torch.optim.SparseAdam(ep, lr=1e-4)
                            except Exception:
                                oe = torch.optim.AdamW(ep, lr=1e-4, weight_decay=0.01)
                            ua = False; ad = None
                            tried_cpu_fallback = True
                            emit('log', "✅ Retrying on CPU-only model")
                            lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                        else:
                            raise

                    lval = float(lo.item())
                    micro_step += 1
                    if cards:
                        emit('model_preview', {'step': micro_step, 'cards': cards})
                    if cal_done >= 1 and mi == 0:
                        elapsed = time.time() - cs_
                        sps_micro = elapsed / micro_step
                        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
                        remaining = max(60, (rh * 3600) - elapsed - (UPLOAD_BUFFER_MIN * 60))
                        target_micro_steps = micro_step + int(remaining / sps_micro)
                    elapsed_cal = time.time() - cs_
                    current_tps = tt / max(elapsed_cal, 1.0)
                    time_left_mins = 0
                    if cal_done >= 1 and micro_step > 0:
                        elapsed = time.time() - cs_
                        sps_micro = elapsed / micro_step
                        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
                        remaining = max(60, (rh * 3600) - elapsed - (UPLOAD_BUFFER_MIN * 60))
                        time_left_mins = remaining / 60.0
                    emit('cal_stats', {
                        'step': micro_step, 'total': target_micro_steps,
                        'loss': safe_float(lval * as_, 10.0),
                        'microbatch': micro_step,
                        'tps': safe_float(current_tps),
                        'time_left': safe_float(time_left_mins),
                    })
                    lo.backward()
                    oe.step()
                    oe.zero_grad(set_to_none=True)
                    tt += x.numel()
                    del lo, x, y
                if hb.should_stop() or se.is_set():
                    return
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                ha()
                continue
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            ob.step()
            ob.zero_grad(set_to_none=True)
            gc.collect()
            cal_done += 1
        if se.is_set():
            return
        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
        remaining = max(60, (rh * 3600) - (time.time() - cs_) - (UPLOAD_BUFFER_MIN * 60))
        dl = time.time() + remaining
        if micro_step > 0:
            sps_micro = (time.time() - cs_) / micro_step
            target_micro_steps = micro_step + int(remaining / sps_micro)
        else:
            target_micro_steps = 1000
        emit('status', 'training')
        emit('log', f"Target: ~{target_micro_steps} micro-steps")
        last_stats_emit = 0.0
        while time.time() < dl:
            if hb.should_stop() or se.is_set():
                break
            ob.zero_grad(set_to_none=True)
            als = 0.0
            cc = 0
            try:
                for mi in range(as_):
                    if hb.should_stop() or se.is_set() or time.time() >= dl:
                        break
                    if ds.needs_new_subchunk():
                        ds.advance(srv, "bf16")
                    sd = (abs(hash("t")) % 10000) + micro_step * 1000 + mi
                    x, y = ds.get_batch(bs, seed=sd)
                    x, y = x.to(train_device), y.to(train_device)
                    want_preview = (micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0)

                    try:
                        lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                    except RuntimeError as re:
                        if (not tried_cpu_fallback) and _is_accel_error(str(re)):
                            emit('log', f"⚠️ {train_backend} op failed at fwd: {str(re)[:120]}")
                            emit('log', "🔻 Falling back to CPU-only model...")
                            model.to_cpu_only()
                            bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
                            ep = list(model.engram.parameters())
                            try:
                                import bitsandbytes as bnb
                                ob = bnb.optim.AdamW8bit(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
                            except Exception:
                                ob = torch.optim.AdamW(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
                            try:
                                oe = torch.optim.SparseAdam(ep, lr=1e-4)
                            except Exception:
                                oe = torch.optim.AdamW(ep, lr=1e-4, weight_decay=0.01)
                            ua = False; ad = None
                            tried_cpu_fallback = True
                            lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                        else:
                            raise

                    lval = float(lo.item()) * as_
                    lo.backward()
                    oe.step()
                    oe.zero_grad(set_to_none=True)
                    als += lval
                    tt += x.numel()
                    cc += 1
                    micro_step += 1
                    if cards:
                        emit('model_preview', {'step': micro_step, 'cards': cards})
                    rem_time = max(1, dl - time.time())
                    elapsed_total = time.time() - cs_
                    if micro_step > 5:
                        sps_micro = elapsed_total / micro_step
                        future_steps = int(rem_time / sps_micro)
                        current_target = micro_step + future_steps
                        if current_target > target_micro_steps:
                            target_micro_steps = current_target
                    if time.time() - last_stats_emit > 0.25 or time.time() >= dl:
                        last_stats_emit = time.time()
                        lv = als / cc if cc > 0 else 10.0
                        ctps = tt / max(time.time() - cs_, 1)
                        emit('stats', {
                            'step': micro_step, 'target': target_micro_steps,
                            'loss': safe_float(lv, 10.0), 'tps': safe_float(ctps),
                            'global_step': gs, 'round': cr,
                            'time_left': safe_float(max(0, (dl - time.time()) / 60)),
                        })
                    del lo, x, y
                if hb.should_stop() or se.is_set():
                    break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                ha()
                continue
            if cc == 0:
                break
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            ob.step()
            ob.zero_grad(set_to_none=True)
        if se.is_set():
            emit('log', "Stopped by user. Skipping upload.")
            return
        fl = float(lval) if lval else 10.0
        emit('log', f"Done: {micro_step} micro-steps, loss {fl:.4f}")
        emit('status', 'uploading')
        dbf = torch.from_numpy(model.get_base_weights() - iw).to(torch.bfloat16).view(torch.uint16).numpy()
        ed = model.engram.table.weight.data.cpu() - iew
        rn = ed.abs().sum(dim=1)
        k = max(1, int(len(rn) * 0.10))
        tv, ti = torch.topk(rn, k)
        ai = ti[tv > 1e-8]
        si = ai.cpu().numpy().astype(np.uint32) if len(ai) else np.array([], dtype=np.uint32)
        sv = ed[ai].to(torch.bfloat16).view(torch.uint16).numpy() if len(ai) else np.array([], dtype=np.uint16)
        has_engram = bool(len(si) > 0)
        engram_sparse_count = int(len(si))
        pl = json.dumps({
            "taskId": tid, "loss": fl, "localSteps": micro_step, "tokensProcessed": tt,
            "loraRank": 0, "isDelta": True, "weightFormat": "bf16",
            "hasEngram": has_engram, "engramSparseCount": engram_sparse_count,
        }).encode()
        bi = struct.pack('<I', len(pl)) + pl + np.ascontiguousarray(dbf).tobytes() + np.ascontiguousarray(si).tobytes() + np.ascontiguousarray(sv).tobytes()
        compressed_data = gzip.compress(bi, compresslevel=6)
        upload_size_mb = len(compressed_data) / (1024 * 1024)
        del dbf, ed, si, sv, pl, bi
        gc.collect()
        meta = {
            "taskId": tid, "loss": fl, "localSteps": micro_step, "tokensProcessed": tt,
            "loraRank": 0, "isDelta": True, "weightFormat": "bf16",
            "hasEngram": has_engram, "engramSparseCount": engram_sparse_count,
        }
        emit('overlay_show', {'title_key': 'uploading', 'indeterminate': False})
        emit('log', f"Uploading {upload_size_mb:.1f} MB delta to server...")
        r = None
        try:
            if len(compressed_data) <= MAX_DIRECT_UPLOAD_BYTES:
                try:
                    r = requests.post(
                        f"{srv}/fl/submit",
                        headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **h},
                        data=compressed_data,
                        timeout=600,
                    )
                    if r.status_code == 413:
                        emit("log", "Direct upload rejected. Switching to chunked...")
                        r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
                except Exception as e:
                    if len(compressed_data) > 8 * 1024 * 1024:
                        emit("log", f"Direct upload failed ({e}). Trying chunked...")
                        r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
                    else:
                        raise
            else:
                r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
            if r is not None and r.status_code == 200:
                emit("log", "Submitted successfully!")
            elif r is not None:
                emit("log", f"Submit failed: {r.text[:200]}")
        except KeyboardInterrupt:
            emit("log", "Upload aborted by user.")
        except Exception as e:
            emit("log", f"Upload failed: {e}")
        finally:
            emit('overlay_hide', {})
            try:
                del compressed_data
            except Exception:
                pass
            gc.collect()
    finally:
        hb.stop()
        model = ob = oe = x = y = lo = iw = ie = iew = ds = None
        gc.collect()
        if train_backend in ("CUDA", "ROCM"):
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#fafaf9;--bg-soft:#fff;--bg-softer:#f0f0ee;--text:#111113;--text-dim:#3a3a40;--text-muted:#6b6b73;--border:#e4e4e1;--border-strong:#d0d0cb;--accent:#16a34a;--accent-dim:#15803d;--warn:#b45309;--err:#dc2626;--err-soft:rgba(220,38,38,.06);--err-border:rgba(220,38,38,.28);--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;--r:6px}
body.dark{--bg:#1a1a1e;--bg-soft:#242428;--bg-softer:#2e2e33;--text:#e8e8ec;--text-dim:#b8b8be;--text-muted:#888890;--border:#3a3a40;--border-strong:#505058;--accent:#22c55e;--accent-dim:#4ade80;--warn:#fbbf24;--err:#f87171;--err-soft:rgba(248,113,113,.08);--err-border:rgba(248,113,113,.35)}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;width:100%}
body{background:var(--bg);color:var(--text);font-family:var(--sans);display:flex;flex-direction:column;overflow:hidden;font-size:14px;line-height:1.5}
header{flex:0 0 auto;height:54px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 clamp(12px,2vw,20px);gap:10px}
.logo{display:flex;align-items:center;gap:9px;min-width:0}
.logo-img{height:22px;width:auto}
.logo-fallback{width:24px;height:24px;border:1px solid var(--border-strong);display:none;place-items:center;font-size:13px;font-weight:600}
.logo-text{font-size:15px;font-weight:600;white-space:nowrap}
.logo-ver{font-family:var(--mono);font-size:10px;color:var(--text-muted);border:1px solid var(--border);padding:1px 5px;border-radius:3px}
.header-right{display:flex;align-items:center;gap:8px}
#user-info{font-family:var(--mono);font-size:11px;color:var(--text-muted);max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.theme-toggle{width:30px;height:30px;display:flex;align-items:center;justify-content:center;background:var(--bg-soft);color:var(--text);border:1px solid var(--border);border-radius:4px;cursor:pointer;font-size:14px;padding:0}
.theme-toggle:hover{border-color:var(--border-strong)}
.theme-toggle .icon-sun{display:none}
body.dark .theme-toggle .icon-sun{display:inline}
body.dark .theme-toggle .icon-moon{display:none}
body:not(.dashboard-active) #backend-select{display:none}
.backend-select,.lang-select{font-family:var(--sans);font-size:12px;background:var(--bg-soft);color:var(--text);border:1px solid var(--border);padding:5px 8px;border-radius:4px;cursor:pointer;outline:none}
.backend-select:hover,.lang-select:hover{border-color:var(--border-strong)}
.backend-select:disabled{opacity:.5;cursor:not-allowed}
main{flex:1 1 auto;min-height:0;overflow-y:auto;padding:clamp(10px,2vw,18px) clamp(12px,2vw,20px)}
.view{display:none;height:100%;min-height:0}
.view.active{display:flex;flex-direction:column}
#view-login.active{flex:1 1 auto;align-items:center;justify-content:center;padding:10px 0;min-height:0}
.login-box{width:min(380px,100%);max-height:100%;overflow-y:auto;background:var(--bg-soft);border:1px solid var(--border);padding:clamp(14px,2.5vw,24px);border-radius:8px;display:flex;flex-direction:column}
.auth-screen{display:none;flex-direction:column}
.auth-screen.active{display:flex}
.login-box h2{font-size:clamp(17px,3vw,21px);font-weight:600;text-align:center;margin-bottom:3px}
.login-box h2 .em{color:var(--accent);font-weight:700}
.login-sub{font-size:12px;color:var(--text-dim);text-align:center;margin-bottom:14px}
.fg{margin-bottom:9px}
.fg label{display:block;font-size:9px;color:var(--text-muted);margin-bottom:3px;font-family:var(--mono);text-transform:uppercase}
input{width:100%;font-family:inherit;font-size:12.5px;background:var(--bg-softer);color:var(--text);border:1px solid var(--border);padding:8px 10px;border-radius:4px;outline:none}
input:focus{border-color:var(--accent)}
@font-face{font-family:"Twemoji Country Flags";unicode-range:U+1F1E6-1F1FF,U+1F3F4,U+E0062-E0063,U+E0065,U+E0067,U+E006C,U+E006E,U+E0073-E0074,U+E0077,U+E007F;src:url("https://cdn.jsdelivr.net/npm/country-flag-emoji-polyfill@0.1/dist/TwemojiCountryFlags.woff2") format("woff2");font-display:swap}
:root{--sans:"Twemoji Country Flags",-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; /* ...rest unchanged */}
button{font-family:inherit;font-size:12.5px;font-weight:500;cursor:pointer;border-radius:4px}
.btn-primary{background:var(--text);color:var(--bg);border:1px solid var(--text);padding:9px 14px}
.btn-primary:hover:not(:disabled){opacity:.85}
.btn-danger{background:transparent;color:var(--err);border:1px solid var(--err);padding:9px 14px}
.btn-danger:hover:not(:disabled){background:var(--err);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
.auth-btn{width:100%;margin-top:2px}
.error-box{display:none;margin-top:8px;padding:8px 10px;background:var(--err-soft);border:1px solid var(--err-border);border-left:3px solid var(--err);border-radius:4px;font-size:12px;color:var(--err)}
.error-box.show{display:block}
.auth-switch{margin-top:12px;font-size:12px;color:var(--text-muted);text-align:center}
.auth-switch a{color:var(--accent-dim);cursor:pointer;text-decoration:none}
.anon-hint{margin-top:10px;padding-top:10px;border-top:1px solid var(--border);font-size:11px;color:var(--text-muted);text-align:center}
.anon-hint a{color:var(--text-dim);cursor:pointer;text-decoration:none}
.dash{display:flex;flex-direction:column;gap:12px;flex:1 1 auto;min-height:0}
.dash-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.dash-title{font-size:17px;font-weight:600}
.status-badge{font-family:var(--mono);font-size:10px;padding:3px 10px;border-radius:100px;background:rgba(22,163,74,.08);color:var(--accent-dim);border:1px solid rgba(22,163,74,.35)}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.stat-card{background:var(--bg-soft);border:1px solid var(--border);padding:12px 14px;border-radius:var(--r);min-width:0}
.stat-val{font-family:var(--mono);font-size:clamp(17px,2.4vw,22px);font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stat-val.accent{color:var(--accent-dim)}
.stat-lbl{font-size:11px;color:var(--text-muted);margin-top:2px}
.panel{background:var(--bg-soft);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px}
.panel.flex{flex:1 1 auto;display:flex;flex-direction:column;min-height:160px}
.panel-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px;font-family:var(--mono);font-size:11px;color:var(--text-muted);flex-wrap:wrap}
.panel-head .pv{color:var(--text)}
.progress-track{height:6px;background:var(--bg-softer);border:1px solid var(--border);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;width:0%;background:var(--accent);transition:width .4s ease}
.dash-body{display:grid;grid-template-columns:1fr;gap:12px;flex:1 1 auto;min-height:0}
@media(min-width:800px){.dash-body{grid-template-columns:1fr 1fr}}
.col{display:flex;flex-direction:column;gap:12px;min-height:0}
.panel-content{flex:1 1 auto;min-height:0;overflow-y:auto}
.graph-container{position:relative;height:230px}
#loss-graph{position:absolute;inset:0;width:100%;height:100%;display:block;transition:opacity .2s}
#loss-graph.hidden-graph{opacity:0;pointer-events:none}
.graph-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-family:var(--mono);font-size:11px;transition:opacity .2s}
.graph-empty.hidden-empty{opacity:0;pointer-events:none}
.pred-lines{display:flex;flex-direction:column;gap:6px;font-family:var(--mono);font-size:12px}
.preview-empty{color:var(--text-muted);text-align:center;padding:24px 0;font-size:11px}
.pred-line{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:6px 8px;border-radius:4px;border:1px solid var(--border);background:var(--bg-softer);animation:lineIn .24s ease}
@keyframes lineIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.pred-line.good{border-left:3px solid var(--accent)}
.pred-line.bad{border-left:3px solid var(--err)}
.pred-main{display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0;flex:1}
.pred-text{color:var(--text-dim);font-size:12px;word-break:break-word}
.pred-token{font-weight:600;padding:1px 5px;border-radius:3px;font-size:12px}
.good .pred-token{background:rgba(22,163,74,.10);color:var(--accent-dim)}
.bad .pred-token{background:rgba(220,38,38,.08);color:var(--err)}
.pred-side{display:flex;align-items:center;gap:8px;color:var(--text-muted);font-size:10px;white-space:nowrap}
.pred-badge{font-weight:600}
.good .pred-badge{color:var(--accent-dim)}
.bad .pred-badge{color:var(--err)}
.status-strip{font-family:var(--mono);font-size:11px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 0}
.ss-event{display:block;width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.controls{display:flex;gap:8px;justify-content:flex-end;align-items:center;flex-wrap:wrap}
.overlay{position:fixed;inset:0;background:rgba(250,250,249,.86);backdrop-filter:blur(3px);display:flex;align-items:center;justify-content:center;z-index:50;opacity:0;pointer-events:none;transition:opacity .25s}
body.dark .overlay{background:rgba(26,26,30,.86)}
.overlay.show{opacity:1;pointer-events:auto}
.overlay-card{width:min(420px,88vw);background:var(--bg-soft);border:1px solid var(--border);border-radius:8px;padding:22px 24px}
.overlay-title{font-size:15px;font-weight:600;margin-bottom:4px}
.overlay-subtitle{font-size:12.5px;color:var(--text-muted);margin-bottom:14px;display:none}
.overlay-subtitle.show{display:block}
.overlay-bar{height:10px;background:var(--bg-softer);border:1px solid var(--border);border-radius:5px;overflow:hidden;position:relative}
.overlay-fill{height:100%;width:0%;background:linear-gradient(90deg,#15803d,#16a34a 55%,#4ade80);transition:width .25s ease}
.overlay-bar.indet .overlay-fill{width:35%;transition:none;animation:indet 1.15s ease-in-out infinite}
@keyframes indet{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
.overlay-sub{margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--text-muted);display:flex;justify-content:space-between;gap:8px}
.motd-overlay{position:fixed;top:24px;left:50%;transform:translateX(-50%) translateY(-20px);width:min(520px,92vw);background:var(--bg-soft);border:1px solid var(--border);border-left:4px solid var(--warn);border-radius:6px;padding:16px 20px;box-shadow:0 12px 32px rgba(0,0,0,.08);z-index:200;opacity:0;pointer-events:none;transition:opacity .4s,transform .4s;display:flex;flex-direction:column;gap:12px}
.motd-overlay.show{opacity:1;pointer-events:auto;transform:translateX(-50%) translateY(0)}
.motd-header{display:flex;align-items:center;justify-content:space-between;gap:10px}
.motd-title{font-size:11px;font-weight:600;color:var(--warn);display:flex;align-items:center;gap:8px;font-family:var(--mono);text-transform:uppercase}
.motd-close{background:transparent;border:1px solid var(--border);color:var(--text-muted);cursor:pointer;font-size:12px;padding:4px 8px;border-radius:4px;font-family:var(--mono)}
.motd-close:hover{color:var(--text);border-color:var(--border-strong)}
.motd-text{font-size:13px;color:var(--text-dim);line-height:1.65;white-space:pre-wrap;word-break:break-word;max-height:240px;overflow-y:auto}
@media(max-width:520px){header{height:auto;flex-wrap:wrap;padding:8px 12px}#user-info{display:none}.controls button{flex:1}}
</style></head>
<body>
<header>
  <div class="logo">
    <img src="__LOGO__" id="logo-img" data-light="__LOGO__" data-dark="__LOGO_DARK__" class="logo-img" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='grid';">
    <div class="logo-fallback">C</div>
    <span class="logo-text">CrowdGPT</span>
    <span class="logo-ver">v__VERSION__</span>
  </div>
  <div class="header-right">
    <span id="user-info"></span>
    <select class="backend-select" id="backend-select" disabled></select>
    <select class="lang-select" id="lang-select">
      <option value="en">🇬🇧 English</option>
      <option value="fr">🇫🇷 Français</option>
      <option value="es">🇪🇸 Español</option>
      <option value="de">🇩🇪 Deutsch</option>
      <option value="tr">🇹🇷 Türkçe</option>
    </select>
    <button class="theme-toggle" id="theme-toggle"><span class="icon-moon">🌙</span><span class="icon-sun">☀️</span></button>
  </div>
</header>
<main>
  <div id="view-login" class="view active">
    <div class="login-box">
      <div class="auth-screen active" id="screen-login">
        <h2><span data-i18n="login_pre"></span><span class="em">CrowdGPT</span></h2>
        <p class="login-sub" data-i18n="login_sub"></p>
        <div class="fg"><label data-i18n="server_url"></label><input type="text" id="login-server" value="https://server.crowdgpt.net"></div>
        <div class="fg"><label data-i18n="username"></label><input type="text" id="login-user" autocomplete="username"></div>
        <div class="fg"><label data-i18n="password"></label><input type="password" id="login-pass" autocomplete="current-password"></div>
        <div class="error-box" id="login-error"></div>
        <button id="btn-login" class="btn-primary auth-btn" data-i18n="login_btn"></button>
        <div class="auth-switch"><span data-i18n="no_account"></span> <a id="go-to-register" data-i18n="register_link"></a></div>
        <div class="anon-hint"><span data-i18n="anon_preface"></span> <a id="btn-anon" data-i18n="anon_link"></a></div>
      </div>
      <div class="auth-screen" id="screen-register">
        <h2><span data-i18n="register_title_pre"></span><span class="em">CrowdGPT</span></h2>
        <p class="login-sub" data-i18n="register_sub"></p>
        <div class="fg"><label data-i18n="server_url"></label><input type="text" id="reg-server" value="https://server.crowdgpt.net"></div>
        <div class="fg"><label data-i18n="username"></label><input type="text" id="reg-user" autocomplete="username"></div>
        <div class="fg"><label data-i18n="email"></label><input type="email" id="reg-email" autocomplete="email"></div>
        <div class="fg"><label data-i18n="password"></label><input type="password" id="reg-pass" autocomplete="new-password"></div>
        <div class="error-box" id="reg-error"></div>
        <button id="btn-register" class="btn-primary auth-btn" data-i18n="register_btn"></button>
        <div class="auth-switch"><span data-i18n="have_account"></span> <a id="go-to-login" data-i18n="login_link"></a></div>
      </div>
    </div>
  </div>
  <div id="view-dashboard" class="view">
    <div class="dash">
      <div class="dash-head">
        <span class="dash-title" data-i18n="dash_title"></span>
        <span class="status-badge" id="status-indicator"></span>
      </div>
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-val accent" id="stat-loss">—</div><div class="stat-lbl" data-i18n="loss"></div></div>
        <div class="stat-card"><div class="stat-val" id="stat-tps">—</div><div class="stat-lbl" data-i18n="tokens_sec"></div></div>
        <div class="stat-card"><div class="stat-val" id="stat-time">—</div><div class="stat-lbl" data-i18n="time_left"></div></div>
      </div>
      <div class="panel">
        <div class="panel-head"><span><span data-i18n="round"></span> <span class="pv" id="stat-round">—</span> · <span data-i18n="step"></span> <span class="pv" id="stat-step-cur">0</span>/<span class="pv" id="stat-step-tot">0</span></span><span class="pv" id="progress-text">0%</span></div>
        <div class="progress-track"><div class="progress-fill" id="progress-bar"></div></div>
      </div>
      <div class="dash-body">
        <div class="col"><div class="panel flex">
          <div class="panel-head"><span data-i18n="loss_history"></span><span class="pv" id="graph-last"></span></div>
          <div class="graph-container"><canvas id="loss-graph"></canvas><div class="graph-empty" id="graph-empty" data-i18n="waiting_data"></div></div>
        </div></div>
        <div class="col"><div class="panel flex">
          <div class="panel-head"><span data-i18n="model_thoughts"></span><span class="pv" id="preview-step">—</span></div>
          <div class="panel-content"><div id="model-preview" class="pred-lines"><div class="preview-empty" data-i18n="waiting_preview"></div></div></div>
        </div></div>
      </div>
      <div class="status-strip"><span class="ss-event" id="status-event"></span></div>
      <div class="controls">
        <button id="btn-start" class="btn-primary" data-i18n="start"></button>
        <button id="btn-stop" class="btn-danger" data-i18n="stop" disabled></button>
      </div>
    </div>
  </div>
</main>
<div class="overlay" id="overlay"><div class="overlay-card">
  <div class="overlay-title" id="overlay-title"></div>
  <div class="overlay-subtitle" id="overlay-subtitle"></div>
  <div class="overlay-bar" id="overlay-bar"><div class="overlay-fill" id="overlay-fill"></div></div>
  <div class="overlay-sub"><span id="overlay-pct"></span><span id="overlay-mb"></span></div>
</div></div>
<div class="motd-overlay" id="motd-overlay">
  <div class="motd-header">
    <div class="motd-title">⚠️ <span data-i18n="motd_title">System Notice</span></div>
    <button class="motd-close" id="motd-close" data-i18n="close">Close</button>
  </div>
  <div class="motd-text" id="motd-text"></div>
</div>
<script>
const I18N={en:{login_pre:"Sign in to ",login_sub:"Welcome back to the swarm.",register_title_pre:"Join ",register_sub:"Create your account.",server_url:"Server address",username:"Username",password:"Password",email:"Email",login_btn:"Log in",register_btn:"Create account",no_account:"No account?",register_link:"Create one",have_account:"Already registered?",login_link:"Log in",anon_preface:"Stay anonymous?",anon_link:"Skip and contribute anonymously",dash_title:"Training",loss:"Loss",tokens_sec:"Tokens/sec",time_left:"Time left",round:"Round",step:"Step",loss_history:"Loss history",waiting_data:"Waiting for data…",model_thoughts:"Model Predictions",waiting_preview:"Waiting for first prediction…",idle:"Idle",training:"Training…",waiting:"Waiting…",uploading:"Uploading…",downloading:"Downloading…",preparing:"Preparing…",calibrating:"Calibrating…",dl_weights:"Downloading weights…",wait_coord:"Waiting for coordinator…",wait_coord_sub:"This can take a few minutes.",start:"Start training",stop:"Stop",err_invalid_credentials:"Invalid username or password.",err_username_taken:"Username taken.",err_email_taken:"Email already registered.",err_weak_password:"Password must be ≥6 chars.",err_invalid_input:"Check your input.",err_login_failed:"Login failed.",err_register_failed:"Registration failed.",err_network:"Cannot reach server.",motd_title:"System Notice",close:"Close"},fr:{login_pre:"Connexion à ",login_sub:"Bon retour parmi nous.",register_title_pre:"Rejoignez ",register_sub:"Créez votre compte.",server_url:"Adresse",username:"Nom d'utilisateur",password:"Mot de passe",email:"E-mail",login_btn:"Se connecter",register_btn:"Créer le compte",no_account:"Pas de compte ?",register_link:"Créer un compte",have_account:"Déjà inscrit ?",login_link:"Se connecter",anon_preface:"Rester anonyme ?",anon_link:"Continuer anonymement",dash_title:"Entraînement",loss:"Perte",tokens_sec:"Tokens/sec",time_left:"Reste",round:"Manche",step:"Étape",loss_history:"Historique",waiting_data:"En attente…",model_thoughts:"Prédictions",waiting_preview:"En attente…",idle:"Inactif",training:"Entraînement…",waiting:"Attente…",uploading:"Envoi…",downloading:"Téléchargement…",preparing:"Préparation…",calibrating:"Calibration…",dl_weights:"Téléchargement des poids…",wait_coord:"Attente du coordinateur…",wait_coord_sub:"Cela peut prendre quelques minutes.",start:"Démarrer",stop:"Arrêter",err_invalid_credentials:"Identifiants invalides.",err_username_taken:"Nom pris.",err_email_taken:"E-mail déjà enregistré.",err_weak_password:"Mot de passe ≥6 caractères.",err_invalid_input:"Vérifiez la saisie.",err_login_failed:"Échec de connexion.",err_register_failed:"Échec d'inscription.",err_network:"Serveur injoignable.",motd_title:"Avis système",close:"Fermer"},es:{login_pre:"Inicia sesión en ",login_sub:"Bienvenido de vuelta.",register_title_pre:"Únete a ",register_sub:"Crea tu cuenta.",server_url:"Servidor",username:"Usuario",password:"Contraseña",email:"Correo",login_btn:"Iniciar sesión",register_btn:"Crear cuenta",no_account:"¿Sin cuenta?",register_link:"Crear una",have_account:"¿Registrado?",login_link:"Iniciar sesión",anon_preface:"¿Anónimo?",anon_link:"Continuar anónimamente",dash_title:"Entrenamiento",loss:"Pérdida",tokens_sec:"Tokens/seg",time_left:"Queda",round:"Ronda",step:"Paso",loss_history:"Historial",waiting_data:"Esperando…",model_thoughts:"Predicciones",waiting_preview:"Esperando…",idle:"Inactivo",training:"Entrenando…",waiting:"Esperando…",uploading:"Subiendo…",downloading:"Descargando…",preparing:"Preparando…",calibrating:"Calibrando…",dl_weights:"Descargando pesos…",wait_coord:"Esperando coordinador…",wait_coord_sub:"Puede tardar unos minutos.",start:"Iniciar",stop:"Detener",err_invalid_credentials:"Credenciales inválidas.",err_username_taken:"Usuario en uso.",err_email_taken:"Correo ya registrado.",err_weak_password:"Contraseña ≥6 caracteres.",err_invalid_input:"Revisa los datos.",err_login_failed:"Falló el inicio de sesión.",err_register_failed:"Falló el registro.",err_network:"Servidor inaccesible.",motd_title:"Aviso",close:"Cerrar"},de:{login_pre:"Anmelden bei ",login_sub:"Willkommen zurück.",register_title_pre:"Tritt ",register_sub:"Erstelle dein Konto.",server_url:"Serveradresse",username:"Benutzername",password:"Passwort",email:"E-Mail",login_btn:"Anmelden",register_btn:"Konto erstellen",no_account:"Kein Konto?",register_link:"Erstellen",have_account:"Bereits registriert?",login_link:"Anmelden",anon_preface:"Anonym bleiben?",anon_link:"Anonym fortfahren",dash_title:"Training",loss:"Verlust",tokens_sec:"Tokens/Sek",time_left:"Rest",round:"Runde",step:"Schritt",loss_history:"Verlauf",waiting_data:"Warte…",model_thoughts:"Vorhersagen",waiting_preview:"Warte…",idle:"Bereit",training:"Training…",waiting:"Warten…",uploading:"Upload…",downloading:"Herunterladen…",preparing:"Vorbereitung…",calibrating:"Kalibrierung…",dl_weights:"Gewichte werden geladen…",wait_coord:"Warte auf Koordinator…",wait_coord_sub:"Kann ein paar Minuten dauern.",start:"Training starten",stop:"Stopp",err_invalid_credentials:"Ungültig.",err_username_taken:"Benutzername vergeben.",err_email_taken:"E-Mail bereits registriert.",err_weak_password:"Passwort ≥6 Zeichen.",err_invalid_input:"Eingabe prüfen.",err_login_failed:"Anmeldung fehlgeschlagen.",err_register_failed:"Registrierung fehlgeschlagen.",err_network:"Server nicht erreichbar.",motd_title:"Systemmeldung",close:"Schließen"},tr:{login_pre:"Giriş yap: ",login_sub:"Sürüye hoş geldin.",register_title_pre:"Katıl: ",register_sub:"Hesabını oluştur.",server_url:"Sunucu",username:"Kullanıcı adı",password:"Şifre",email:"E-posta",login_btn:"Giriş yap",register_btn:"Hesap oluştur",no_account:"Hesap yok mu?",register_link:"Oluştur",have_account:"Kayıtlı mısın?",login_link:"Giriş yap",anon_preface:"Anonim kal?",anon_link:"Anonim devam et",dash_title:"Eğitim",loss:"Kayıp",tokens_sec:"Token/sn",time_left:"Kalan",round:"Tur",step:"Adım",loss_history:"Geçmiş",waiting_data:"Bekleniyor…",model_thoughts:"Tahminler",waiting_preview:"Bekleniyor…",idle:"Boşta",training:"Eğitiliyor…",waiting:"Bekleniyor…",uploading:"Yükleniyor…",downloading:"İndiriliyor…",preparing:"Hazırlanıyor…",calibrating:"Kalibre ediliyor…",dl_weights:"Ağırlıklar yükleniyor…",wait_coord:"Koordinatör bekleniyor…",wait_coord_sub:"Birkaç dakika sürebilir.",start:"Eğitimi başlat",stop:"Durdur",err_invalid_credentials:"Geçersiz.",err_username_taken:"Kullanıcı adı alınmış.",err_email_taken:"E-posta kayıtlı.",err_weak_password:"Şifre ≥6 karakter.",err_invalid_input:"Girdiyi kontrol et.",err_login_failed:"Giriş başarısız.",err_register_failed:"Kayıt başarısız.",err_network:"Sunucuya ulaşılamıyor.",motd_title:"Sistem Uyarısı",close:"Kapat"}};
let lang='en',statusKey='idle';
const lossHistory=[];const MAXP=80;
function t(k){return (I18N[lang]&&I18N[lang][k])||I18N.en[k]||k}
function applyT(){document.querySelectorAll('[data-i18n]').forEach(e=>e.textContent=t(e.getAttribute('data-i18n')));document.getElementById('status-indicator').textContent=t(statusKey)}
document.getElementById('lang-select').addEventListener('change',e=>{lang=e.target.value;applyT()});
function showAuthScreen(name){document.querySelectorAll('.auth-screen').forEach(s=>s.classList.remove('active'));document.getElementById('screen-'+name).classList.add('active');document.getElementById('login-error').classList.remove('show');document.getElementById('reg-error').classList.remove('show')}
document.getElementById('go-to-register').addEventListener('click',e=>{e.preventDefault();showAuthScreen('register')});
document.getElementById('go-to-login').addEventListener('click',e=>{e.preventDefault();showAuthScreen('login')});
function callApi(n,...a){if(window.pywebview&&window.pywebview.api&&window.pywebview.api[n]){window.pywebview.api[n](...a);return true}return false}
window.addEventListener('pywebviewready',()=>callApi('get_backends'));
document.getElementById('backend-select').addEventListener('change',e=>{const nb=e.target.value;if(nb===window._currentBackend)return;callApi('set_backend',nb);});
document.getElementById('btn-login').addEventListener('click',()=>{const srv=document.getElementById('login-server').value.trim();const u=document.getElementById('login-user').value.trim();const p=document.getElementById('login-pass').value;if(!u||!p){showLoginError('err_invalid_credentials');return}document.getElementById('login-error').classList.remove('show');document.getElementById('btn-login').disabled=true;callApi('login',srv,u,p)});
document.getElementById('btn-register').addEventListener('click',()=>{const srv=document.getElementById('reg-server').value.trim();const u=document.getElementById('reg-user').value.trim();const email=document.getElementById('reg-email').value.trim();const p=document.getElementById('reg-pass').value;if(!u||!email||!p){showRegError('err_invalid_input');return}document.getElementById('reg-error').classList.remove('show');document.getElementById('btn-register').disabled=true;callApi('register',srv,u,email,p)});
document.getElementById('btn-anon').addEventListener('click',e=>{e.preventDefault();callApi('login_anon')});
document.getElementById('btn-start').addEventListener('click',()=>{if(callApi('start')){document.getElementById('btn-start').disabled=true;document.getElementById('btn-stop').disabled=false}});
document.getElementById('btn-stop').addEventListener('click',()=>{callApi('stop');document.getElementById('btn-start').disabled=false;document.getElementById('btn-stop').disabled=true});
function showLoginError(code){const el=document.getElementById('login-error');el.textContent=t(code);el.classList.add('show');document.getElementById('btn-login').disabled=false}
function showRegError(code){const el=document.getElementById('reg-error');el.textContent=t(code);el.classList.add('show');document.getElementById('btn-register').disabled=false}
function mb(b){return (b/1048576).toFixed(1)}
function overlayShow(d){document.getElementById('overlay').classList.add('show');document.getElementById('overlay-title').textContent=t(d.title_key);const s=document.getElementById('overlay-subtitle');if(d.subtitle_key){s.textContent=t(d.subtitle_key);s.classList.add('show')}else{s.classList.remove('show')}const b=document.getElementById('overlay-bar');b.classList.toggle('indet',!!d.indeterminate);document.getElementById('overlay-fill').style.width=d.indeterminate?'':'0%'}
function overlayHide(){document.getElementById('overlay').classList.remove('show')}
function drawGraph(){const c=document.getElementById('loss-graph');if(!c)return;const empty=document.getElementById('graph-empty');if(lossHistory.length<2){c.classList.add('hidden-graph');if(empty)empty.classList.remove('hidden-empty');return}c.classList.remove('hidden-graph');if(empty)empty.classList.add('hidden-empty');const rect=c.getBoundingClientRect();const W=rect.width,H=rect.height;if(W<10||H<10){requestAnimationFrame(drawGraph);return}const ctx=c.getContext('2d');const dpr=window.devicePixelRatio||1;c.width=Math.round(W*dpr);c.height=Math.round(H*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);const data=lossHistory.slice(-MAXP);const mn=Math.min(...data),mx=Math.max(...data),range=(mx-mn)||1;const pad=6,pH=H-pad*2,pW=W-pad*2,st=pW/(data.length-1);const isDark=document.body.classList.contains('dark');ctx.strokeStyle=isDark?'rgba(255,255,255,.06)':'rgba(17,17,19,.07)';for(let g=1;g<4;g++){const y=pad+pH*g/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(pad+pW,y);ctx.stroke()}const pt=i=>[pad+i*st,pad+pH-(data[i]-mn)/range*pH];ctx.beginPath();data.forEach((v,i)=>{const[x,y]=pt(i);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.lineTo(pad+pW,pad+pH);ctx.lineTo(pad,pad+pH);ctx.closePath();const gr=ctx.createLinearGradient(0,0,0,H);gr.addColorStop(0,isDark?'rgba(34,197,94,.20)':'rgba(22,163,74,.16)');gr.addColorStop(1,'rgba(22,163,74,0)');ctx.fillStyle=gr;ctx.fill();ctx.beginPath();data.forEach((v,i)=>{const[x,y]=pt(i);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=isDark?'#22c55e':'#16a34a';ctx.lineWidth=1.5;ctx.stroke()}
window.handleEvent=function(ev,data){
if(ev==='login_success'){document.getElementById('view-login').classList.remove('active');document.getElementById('view-dashboard').classList.add('active');document.body.classList.add('dashboard-active');document.getElementById('user-info').textContent=data.username;document.getElementById('btn-login').disabled=false;document.getElementById('btn-register').disabled=false;requestAnimationFrame(drawGraph)}
else if(ev==='login_error'){const code=data.code||'err_invalid_credentials';const map={invalid_credentials:'err_invalid_credentials',login_failed:'err_login_failed',username_taken:'err_username_taken',email_taken:'err_email_taken',weak_password:'err_weak_password',invalid_input:'err_invalid_input',register_failed:'err_register_failed',network:'err_network'};const tk=map[code]||'err_login_failed';const loginActive=document.getElementById('screen-login').classList.contains('active');if(loginActive)showLoginError(tk);else showRegError(tk)}
else if(ev==='log'){document.getElementById('status-event').textContent=data}
else if(ev==='backends'){const sel=document.getElementById('backend-select');sel.innerHTML='';window._currentBackend=data.current;const order=[{key:'cuda',label:'CUDA (NVIDIA)'},{key:'rocm',label:'ROCm (AMD)'},{key:'mps',label:'MPS (Apple)'},{key:'xpu',label:'XPU (Intel)'},{key:'directml',label:'DirectML (Windows)'},{key:'cpu',label:'CPU'}];order.forEach(bk=>{const opt=document.createElement('option');opt.value=bk.key;opt.textContent=bk.label;opt.disabled=!data.available[bk.key];if(bk.key===data.current)opt.selected=true;sel.appendChild(opt)});sel.disabled=false}
else if(ev==='overlay_show'){overlayShow(data)}
else if(ev==='overlay_progress'){const bar=document.getElementById('overlay-bar');if(data.total>0){bar.classList.remove('indet');const pct=Math.min(100,data.done/data.total*100);document.getElementById('overlay-fill').style.width=pct+'%';document.getElementById('overlay-pct').textContent=pct.toFixed(1)+'%';document.getElementById('overlay-mb').textContent=mb(data.done)+' / '+mb(data.total)+' MB'}}
else if(ev==='overlay_hide'){overlayHide()}
else if(ev==='cal_stats'||ev==='stats'){const loss=Number(data.loss);const tps=Number(data.tps);const step=Number(data.step);const target=ev==='cal_stats'?Number(data.total):Number(data.target);const tl=Number(data.time_left);if(isFinite(loss)){document.getElementById('stat-loss').textContent=loss.toFixed(4);document.getElementById('graph-last').textContent=loss.toFixed(4);if(loss>0){lossHistory.push(loss);if(lossHistory.length>MAXP*2)lossHistory.splice(0,lossHistory.length-MAXP*2)}}if(isFinite(tps))document.getElementById('stat-tps').textContent=Math.round(tps);if(isFinite(tl))document.getElementById('stat-time').textContent=Math.round(tl)+'m';document.getElementById('stat-round').textContent=ev==='cal_stats'?'—':(data.round!=null?data.round:'—');document.getElementById('stat-step-cur').textContent=isFinite(step)?step:0;document.getElementById('stat-step-tot').textContent=isFinite(target)?target:0;const pct=(isFinite(step)&&isFinite(target)&&target>0)?Math.min(100,(step/target)*100):0;document.getElementById('progress-bar').style.width=pct+'%';document.getElementById('progress-text').textContent=pct.toFixed(1)+'%';requestAnimationFrame(drawGraph)}
else if(ev==='model_preview'){const container=document.getElementById('model-preview');const stepEl=document.getElementById('preview-step');if(!container||!stepEl)return;stepEl.textContent='Step '+data.step;let html='';if(data.cards&&data.cards.length>0){data.cards.forEach(card=>{const cls=card.match?'good':'bad';const badge=card.match?'✓':'✗';const prob=(typeof card.prob==='number'&&isFinite(card.prob))?(card.prob*100).toFixed(1)+'%':'';html+='<div class="pred-line '+cls+'"><div class="pred-main"><span class="pred-text">'+escapeHtml(card.context)+'</span><span class="pred-token">'+escapeHtml(card.pred)+'</span></div><div class="pred-side"><span class="pred-badge">'+badge+'</span><span>target: '+escapeHtml(card.target)+'</span><span>'+prob+'</span></div></div>'})}else{html='<div class="preview-empty">'+escapeHtml(t('waiting_preview'))+'</div>'}container.innerHTML=html}
else if(ev==='status'){statusKey=data;document.getElementById('status-indicator').textContent=t(statusKey);if(data==='training'||data==='idle'||data==='calibrating'||data==='preparing')overlayHide();if(data==='idle'){document.getElementById('btn-start').disabled=false;document.getElementById('btn-stop').disabled=true}}
};
function escapeHtml(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
window.addEventListener('resize',()=>requestAnimationFrame(drawGraph));
function updateLogo(){try{const img=document.getElementById('logo-img');if(!img)return;const isDark=document.body.classList.contains('dark');const src=isDark?img.getAttribute('data-dark'):img.getAttribute('data-light');if(src&&src.length>30)img.src=src}catch(e){}}
(function(){try{const saved=localStorage.getItem('crowdgpt-theme');const prefersDark=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches;if(saved==='dark'||(!saved&&prefersDark))document.body.classList.add('dark');updateLogo()}catch(e){}const tbtn=document.getElementById('theme-toggle');if(tbtn){tbtn.addEventListener('click',function(){document.body.classList.toggle('dark');try{localStorage.setItem('crowdgpt-theme',document.body.classList.contains('dark')?'dark':'light')}catch(e){}updateLogo();requestAnimationFrame(drawGraph)})}})();
applyT();
(function(){const key='crowdgpt_motd_last_day';const today=new Date().toISOString().slice(0,10);let last=null;try{last=localStorage.getItem(key)}catch(e){}if(last===today)return;fetch('https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/motd.txt').then(r=>r.ok?r.text():'').then(text=>{text=(text||'').trim();if(!text)return;document.getElementById('motd-text').textContent=text;document.getElementById('motd-overlay').classList.add('show');try{localStorage.setItem(key,today)}catch(e){}}).catch(()=>{})})();
document.getElementById('motd-close').addEventListener('click',()=>document.getElementById('motd-overlay').classList.remove('show'));
</script></body></html>"""

class Api:
    def __init__(self):
        self.window = None
        self.stop_event = threading.Event()
        self.thread = None
        self.auth_token = None
        self.username = None
        self.server_url = DEFAULT_SERVER
        self._lp = 0.0
        self._backend_file = CHECKPOINT_DIR / "selected_backend.json"
        saved = None
        try:
            if self._backend_file.exists():
                saved = json.loads(self._backend_file.read_text()).get("backend")
        except Exception:
            saved = None
        self.selected_backend = saved or get_best_default_backend()
        log.info(f"[CLIENT] Selected backend: {self.selected_backend}")

    def emit(self, ev, data):
        if not self.window:
            return
        if ev in ('log', 'status'):
            try:
                print(f"[EMIT {ev}] {data}", flush=True)
            except Exception:
                pass
        if ev == 'preview_loss':
            n = time.time()
            if n - self._lp < 0.4:
                return
            self._lp = n
        try:
            clean_data = sanitize_for_json(data)
            self.window.evaluate_js(f"window.handleEvent({json.dumps(ev)}, {json.dumps(clean_data)});")
        except Exception as e:
            log.warning(f"Emit failed for {ev}: {e}")

    def get_backends(self):
        available = get_available_backends()
        self.emit('backends', {'available': available, 'current': self.selected_backend})

    def set_backend(self, backend_name):
        log.info(f"[CLIENT] set_backend called with: {backend_name} (was {self.selected_backend})")
        self.selected_backend = backend_name
        try:
            self._backend_file.write_text(json.dumps({"backend": backend_name}))
        except Exception as e:
            log.warning(f"Failed to save backend choice: {e}")
        self.get_backends()
        if self.thread and self.thread.is_alive():
            self.stop_event.set()

    def login(self, s, u, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_login, args=(u, p), daemon=True).start()

    def _do_login(self, u, p):
        tok, err = do_login(self.server_url, u, p)
        if tok:
            self.auth_token = tok
            self.username = u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'login_failed'})

    def register(self, s, u, email, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_register, args=(u, email, p), daemon=True).start()

    def _do_register(self, u, email, p):
        tok, err = do_register(self.server_url, u, email, p)
        if tok:
            self.auth_token = tok
            self.username = u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'register_failed'})

    def login_anon(self):
        self.auth_token = None
        self.username = "anonymous"
        self.emit('login_success', {'username': 'anonymous'})

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._rs, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _check_updates_between_rounds(self):
        try:
            if check_for_update():
                self.emit('log', "🔄 Updated to a new version — restarting...")
                self.emit('status', 'idle')
                time.sleep(2)
                _restart_self()
        except Exception as e:
            log.warning(f"Update check failed: {e}")

    def _rs(self):
        global train_device, train_backend
        try:
            train_device, train_backend = detect_training_backend(self.selected_backend)
            self.emit('meta', {'backend': train_backend})
            self.emit('log', f"Backend: {train_backend} ({train_device})")
            auto_detect_vram_budget()
            self.emit('log', f"VRAM: {memory_config['ram_gb']:.1f} GB")
        except Exception as e:
            self.emit('log', f"Error: {e}")
            self.emit('status', 'idle')
            return
        round_count = 0
        while not self.stop_event.is_set():
            round_count += 1
            self.emit('log', f"{'='*40}")
            self.emit('log', f"Starting Round Cycle #{round_count}")
            self.emit('log', f"{'='*40}")
            try:
                run_single_round_wrapper(self.server_url, self.auth_token, self.stop_event, self.emit)
            except KeyboardInterrupt:
                self.stop_event.set()
                break
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"\n[ROUND-FAIL-TRACEBACK]\n{tb}\n", flush=True)
                for ln in tb.splitlines()[-14:]:
                    self.emit('log', ln)
                self.emit('log', f"Round failed: {e}")
            if self.stop_event.is_set():
                break
            self._check_updates_between_rounds()
            self.emit('status', 'waiting')
            self.emit('log', "✅ Round complete. Auto-relaunching next round in 10s...")
            for _ in range(10):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
        self.emit('status', 'idle')
        self.emit('log', "👋 Training stopped by user.")

def startup():
    """Runs on a background thread once the webview is live."""
    # Give the OS a beat to actually create the HWND, then poll for it.
    if sys.platform == "win32" and WINDOWS_ICO_PATH:
        _wait_for_window(WINDOW_TITLE, timeout=5.0)
        time.sleep(0.15)
        set_windows_icon(WINDOW_TITLE, WINDOWS_ICO_PATH)

    try:
        scr = webview.screens[0]
        aw, ah = scr.width, scr.height
        w = max(480, min(1040, int(aw * 0.80)))
        h = max(360, min(800, int(ah * 0.80)))
        w = min(w, aw - 16)
        h = min(h, ah - 48)
        w = max(320, w)
        h = max(240, h)
        window.resize(w, h)
        window.move(max(0, (aw - w) // 2), max(0, (ah - h) // 2))
    except Exception:
        pass

def _create_window_safe(title, html, icon_path, kwargs):
    """
    pywebview's Windows backend raises a *native* exception (not TypeError)
    when `icon=` points at a non-raster file (e.g. an SVG). Guard against that.
    """
    usable = False
    if icon_path and os.path.exists(icon_path):
        low = icon_path.lower()
        if low.endswith((".png", ".ico", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")):
            usable = True
    if usable:
        try:
            return webview.create_window(title, html=html, icon=icon_path, **kwargs)
        except Exception as e:
            log.warning(f"[icon] create_window(icon=...) failed, retrying without: {e}")
    try:
        return webview.create_window(title, html=html, **kwargs)
    except TypeError:
        kwargs.pop("min_size", None)
        return webview.create_window(title, html=html, **kwargs)

if __name__ == "__main__":
    print(f"CrowdGPT client v{CLIENT_VERSION}")

    # Give the taskbar a stable identity so it uses our icon, not Python's.
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Vxtzq.CrowdGPT.1")
        except Exception:
            pass

    try:
        if check_for_update():
            print("Update applied. Restarting...")
            _restart_self()
    except Exception as e:
        log.warning(f"Auto-update check error: {e}")

    api = Api()
    start_tokenizer_loader()

    kwargs = {"js_api": api, "width": 1000, "height": 720, "min_size": (320, 240)}
    html_out = (HTML.replace("__LOGO__", LOGO_URI_LIGHT)
                    .replace("__LOGO_DARK__", LOGO_URI_DARK)
                    .replace("__VERSION__", CLIENT_VERSION))

    window = _create_window_safe("CrowdGPT", html_out, ICON_PATH, kwargs)
    api.window = window
    webview.start(startup, debug=False)
