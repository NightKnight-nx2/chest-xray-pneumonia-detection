import os
import sys
import datetime
import tkinter as tk
from tkinter import filedialog, ttk

# Threshold for pneumonia detection (logit space).
# Lower = higher recall (fewer missed pneumonia), at cost of more false positives.
# sigmoid(-0.4)  ≈  0.40  →  flag pneumonia if model gives ≥40% confidence
PNEUMONIA_THRESHOLD = -0.4
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ─────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_pytorch_model.pth")
LOG_PATH   = os.path.join(BASE_DIR, "analysis_log.txt")

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────
def load_model(device):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model.to(device)


def preprocess(pil_img):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0)


# ─────────────────────────────────────────────────
#  Grad-CAM
# ─────────────────────────────────────────────────
def compute_gradcam(model, tensor, device):
    """Always compute Grad-CAM w.r.t. the PNEUMONIA class (positive logit).
    This consistently shows which lung regions drive the pneumonia score,
    regardless of what the model ultimately predicted."""
    tensor = tensor.to(device)
    target_layer = model.layer4[-1]

    activation, gradient = {}, {}

    def fwd_hook(m, inp, out):
        activation["val"] = out.detach()

    def bwd_hook(m, grad_in, grad_out):
        gradient["val"] = grad_out[0].detach()

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    # Forward
    output = model(tensor)
    # Always backprop toward PNEUMONIA score (output[:, 0])
    # For a predicted-NORMAL case this reveals subtle pneumonia features the model DID detect
    loss = output[:, 0]

    # Backward
    model.zero_grad()
    loss.backward()

    fh.remove()
    bh.remove()

    act  = activation["val"][0]         # (C, H, W)
    grad = gradient["val"][0]           # (C, H, W)

    # Use only positive gradient contributions (ReLU on gradients too)
    grad = torch.relu(grad)
    weights = grad.mean(dim=(1, 2))     # (C,)
    cam = (weights[:, None, None] * act).sum(dim=0)  # (H, W)
    cam = torch.relu(cam)

    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_heatmap(pil_img, cam):
    """Blend Grad-CAM heatmap onto original image, return PIL."""
    img_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlayed = (heatmap_color * 0.45 + img_np * 0.55).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlayed)


# ─────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────
def log_result(img_path, label, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp}\t{os.path.basename(img_path)}\t{label} ({confidence:.1f}%)\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


# ─────────────────────────────────────────────────
#  GUI Application
# ─────────────────────────────────────────────────
DARK_BG     = "#0D1117"
PANEL_BG    = "#161B22"
ACCENT      = "#6366F1"
TEXT_WHITE  = "#F0F6FC"
TEXT_GRAY   = "#8B949E"
NORMAL_CLR  = "#10B981"
PNEUMO_CLR  = "#F87171"
BORDER      = "#30363D"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pneumonia Detection — AI Desktop")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)
        self.minsize(900, 560)

        # Load model once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status_var = tk.StringVar(value=f"Model yükleniyor…  ({self.device})")
        self._build_ui()
        self.update_idletasks()
        self.after(100, self._load_model)

    # ── UI ──────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────
        hdr = tk.Frame(self, bg=DARK_BG, pady=16)
        hdr.pack(fill=tk.X, padx=24)

        tk.Label(
            hdr, text="Pneumonia Detection AI",
            font=("Segoe UI", 22, "bold"),
            fg=TEXT_WHITE, bg=DARK_BG,
        ).pack(side=tk.LEFT)

        tk.Label(
            hdr,
            text="ResNet50 · PyTorch · Grad-CAM",
            font=("Segoe UI", 10),
            fg=TEXT_GRAY, bg=DARK_BG,
        ).pack(side=tk.LEFT, padx=16, pady=6)

        # ── Status bar ──────────────────────────
        self.status_lbl = tk.Label(
            self, textvariable=self.status_var,
            font=("Segoe UI", 9), fg=TEXT_GRAY, bg=DARK_BG,
        )
        self.status_lbl.pack(anchor=tk.E, padx=24)

        # ── Image Row ───────────────────────────
        img_row = tk.Frame(self, bg=DARK_BG)
        img_row.pack(fill=tk.BOTH, expand=True, padx=24, pady=8)
        img_row.columnconfigure(0, weight=1)
        img_row.columnconfigure(1, weight=1)

        self.orig_panel  = self._make_img_panel(img_row, "Orijinal Görsel",   0)
        self.cam_panel   = self._make_img_panel(img_row, "Grad-CAM Isı Haritası", 1)

        # ── Result box ──────────────────────────
        result_frame = tk.Frame(self, bg=PANEL_BG, bd=0,
                                highlightbackground=BORDER, highlightthickness=1)
        result_frame.pack(fill=tk.X, padx=24, pady=8)

        inner = tk.Frame(result_frame, bg=PANEL_BG, padx=20, pady=14)
        inner.pack(fill=tk.X)

        self.predict_var = tk.StringVar(value="—")
        self.conf_var    = tk.StringVar(value="Bir röntgen görüntüsü yükleyin")

        self.predict_lbl = tk.Label(
            inner, textvariable=self.predict_var,
            font=("Segoe UI", 28, "bold"),
            fg=TEXT_WHITE, bg=PANEL_BG,
        )
        self.predict_lbl.pack(side=tk.LEFT, padx=(0, 24))

        self.conf_lbl = tk.Label(
            inner, textvariable=self.conf_var,
            font=("Segoe UI", 13),
            fg=TEXT_GRAY, bg=PANEL_BG,
        )
        self.conf_lbl.pack(side=tk.LEFT)

        # ── Progress bar ────────────────────────
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=PANEL_BG, background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT,
                        bordercolor=PANEL_BG, relief="flat")
        self.progress = ttk.Progressbar(
            self, orient="horizontal", mode="determinate",
            style="Accent.Horizontal.TProgressbar", length=200
        )
        # (hidden until used)

        # ── Button ──────────────────────────────
        btn_row = tk.Frame(self, bg=DARK_BG, pady=12)
        btn_row.pack(padx=24, anchor=tk.W)

        self.open_btn = tk.Button(
            btn_row,
            text="  📂  Röntgen Görüntüsü Seç",
            font=("Segoe UI", 12, "bold"),
            bg=ACCENT, fg="white",
            activebackground="#4F52C9", activeforeground="white",
            bd=0, padx=18, pady=10, cursor="hand2",
            command=self._open_image,
        )
        self.open_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.log_btn = tk.Button(
            btn_row,
            text="  📋  Analiz Geçmişi",
            font=("Segoe UI", 12),
            bg=PANEL_BG, fg=TEXT_GRAY,
            activebackground=BORDER, activeforeground=TEXT_WHITE,
            bd=0, padx=18, pady=10, cursor="hand2",
            highlightbackground=BORDER, highlightthickness=1,
            command=self._show_log,
        )
        self.log_btn.pack(side=tk.LEFT)

        # footer
        tk.Label(
            self,
            text=f"Model: {MODEL_PATH}  |  Log: {LOG_PATH}",
            font=("Segoe UI", 8), fg="#444C56", bg=DARK_BG,
        ).pack(side=tk.BOTTOM, pady=4)

    def _make_img_panel(self, parent, title, col):
        frame = tk.Frame(parent, bg=PANEL_BG,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.grid(row=0, column=col, padx=8, pady=4, sticky="nsew")

        tk.Label(
            frame, text=title,
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_GRAY, bg=PANEL_BG, pady=8,
        ).pack()

        lbl = tk.Label(frame, bg=PANEL_BG, text="Görüntü bekleniyor…",
                       fg=TEXT_GRAY, font=("Segoe UI", 10))
        lbl.pack(expand=True, fill=tk.BOTH, padx=8, pady=(0, 8))
        return lbl

    # ── Model ────────────────────────────────────
    def _load_model(self):
        try:
            self.model = load_model(self.device)
            self.status_var.set(f"✅  Model hazır  ({self.device.type.upper()})")
            self.open_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.status_var.set(f"❌  Model yüklenemedi: {e}")
            self.open_btn.config(state=tk.DISABLED)

    # ── File Picker ──────────────────────────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Röntgen Görüntüsü Seç",
            filetypes=[("Resim dosyaları", "*.jpeg *.jpg *.png *.bmp *.tiff")],
        )
        if not path:
            return
        self.status_var.set("Analiz ediliyor…")
        self.update_idletasks()
        self._run_inference(path)

    # ── Inference ────────────────────────────────
    def _run_inference(self, img_path):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            tensor  = preprocess(pil_img)

            with torch.no_grad():
                logit = self.model(tensor.to(self.device)).item()
            probability = 1 / (1 + np.exp(-logit))   # sigmoid

            import math
            # Use PNEUMONIA_THRESHOLD in logit space for higher recall
            is_pneumonia = logit > PNEUMONIA_THRESHOLD
            label        = "🔴  PNEUMONIA" if is_pneumonia else "🟢  NORMAL"
            conf         = probability * 100 if is_pneumonia else (1 - probability) * 100
            color        = PNEUMO_CLR if is_pneumonia else NORMAL_CLR

            # Grad-CAM (needs grad)
            cam = compute_gradcam(self.model, tensor, self.device)

            # Update UI
            self._show_image(self.orig_panel, pil_img.resize((IMG_SIZE, IMG_SIZE)))
            cam_pil = overlay_heatmap(pil_img, cam)
            self._show_image(self.cam_panel, cam_pil)

            self.predict_var.set(label)
            self.conf_var.set(f"Güven Skoru:  {conf:.1f}%")
            self.predict_lbl.config(fg=color)
            self.status_var.set(f"Analiz tamamlandı — {os.path.basename(img_path)}")

            log_result(img_path, "PNEUMONIA" if is_pneumonia else "NORMAL", conf)

        except Exception as e:
            self.status_var.set(f"Hata: {e}")
            import traceback; traceback.print_exc()

    def _show_image(self, panel, pil_img):
        # Scale to fit panel keeping aspect ratio
        pil_img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(pil_img)
        panel.config(image=photo, text="")
        panel.image = photo   # keep reference

    # ── Log Viewer ───────────────────────────────
    def _show_log(self):
        win = tk.Toplevel(self)
        win.title("Analiz Geçmişi")
        win.configure(bg=DARK_BG)
        win.geometry("700x400")

        tk.Label(
            win, text="Analiz Geçmişi  (analysis_log.txt)",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_WHITE, bg=DARK_BG, pady=10,
        ).pack()

        text_frame = tk.Frame(win, bg=PANEL_BG)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(
            text_frame, bg=PANEL_BG, fg=TEXT_WHITE,
            font=("Consolas", 10), bd=0, wrap=tk.NONE,
            yscrollcommand=scrollbar.set,
        )
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, encoding="utf-8") as f:
                content = f.read()
            text.insert(tk.END, content if content else "(Henüz kayıt yok)")
        else:
            text.insert(tk.END, "(Henüz analiz yapılmadı)")

        text.config(state=tk.DISABLED)


# ─────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        print("Lütfen önce 'train_pytorch.py' betiğini çalıştırın.")
        sys.exit(1)
    app = App()
    app.mainloop()
