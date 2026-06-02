"""Vision Checkout — Gradio app for Hugging Face Spaces.

Single-file Gradio app: upload a tray photo, run YOLO detection, generate a receipt.
"""
import random
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Fine_Tuned_SAUDI_model" / "weights" / "best.pt"
TEST_IMAGES_DIR = BASE_DIR / "models" / "Fine_Tuned_SAUDI_model" / "test_images"

CATEGORY_PRICES = {
    "puffed_food":      {"ar": "وجبات خفيفة",   "en": "Puffed snacks",   "price": 6.50,  "emoji": "🍿"},
    "dried_fruit":      {"ar": "فواكه مجففة",   "en": "Dried fruit",     "price": 12.00, "emoji": "🍇"},
    "dried_food":       {"ar": "أغذية مجففة",   "en": "Dried food",      "price": 8.00,  "emoji": "🥜"},
    "instant_drink":    {"ar": "مشروب سريع",    "en": "Instant drink",   "price": 5.00,  "emoji": "☕"},
    "instant_noodles":  {"ar": "نودلز سريع",    "en": "Instant noodles", "price": 4.50,  "emoji": "🍜"},
    "dessert":          {"ar": "حلويات",         "en": "Dessert",         "price": 4.00,  "emoji": "🍰"},
    "drink":            {"ar": "مشروبات",        "en": "Drinks",          "price": 3.50,  "emoji": "🥤"},
    "alcohol":          {"ar": "مشروب كحولي",   "en": "Alcohol",         "price": 0.00,  "emoji": "🚫"},
    "milk":             {"ar": "حليب",           "en": "Milk",            "price": 9.00,  "emoji": "🥛"},
    "canned_food":      {"ar": "معلبات",         "en": "Canned food",     "price": 5.00,  "emoji": "🥫"},
    "chocolate":        {"ar": "شوكولاتة",       "en": "Chocolate",       "price": 7.00,  "emoji": "🍫"},
    "gum":              {"ar": "علكة",           "en": "Gum",             "price": 2.50,  "emoji": "🫧"},
    "candy":            {"ar": "حلوى",           "en": "Candy",           "price": 3.00,  "emoji": "🍬"},
    "seasoner":         {"ar": "بهارات وتوابل", "en": "Seasoning",       "price": 6.00,  "emoji": "🧂"},
    "personal_hygiene": {"ar": "عناية شخصية",   "en": "Personal care",   "price": 15.00, "emoji": "🧴"},
    "tissue":           {"ar": "مناديل",         "en": "Tissues",         "price": 8.00,  "emoji": "🧻"},
    "stationery":       {"ar": "قرطاسية",        "en": "Stationery",      "price": 10.00, "emoji": "✏️"},
}

T = {
    "en": {
        "title": "Vision Checkout",
        "tagline": "Smart AI cashier — upload a tray photo and get a receipt.",
        "image_label": "Upload image",
        "confidence": "Confidence threshold",
        "store": "Store name",
        "vat": "Add VAT (15%)",
        "scan": "Scan & generate receipt",
        "detections": "Detections",
        "samples": "Try a sample image",
        "shuffle": "Shuffle samples",
        "prices_label": "Price list",
        "no_items": "No products detected. Try a lower confidence threshold.",
        "detected_count": "Detected {n} item(s) across {k} category(ies).",
        "subtotal": "Subtotal", "vat_label": "VAT (15%)", "total": "TOTAL",
        "currency": "SAR", "invoice": "Invoice",
        "thank_you": "Thank you for shopping.",
        "tagline_receipt": "AI-Powered Smart Checkout",
        "print": "Print receipt",
    },
    "ar": {
        "title": "الكاشير الذكي",
        "tagline": "ارفع صورة المنتجات واحصل على الفاتورة فوراً.",
        "image_label": "ارفع صورة",
        "confidence": "حد الثقة",
        "store": "اسم المتجر",
        "vat": "إضافة ضريبة القيمة المضافة (15٪)",
        "scan": "ابدأ الفحص وأنشئ الفاتورة",
        "detections": "الاكتشافات",
        "samples": "جرّب عيّنة عشوائية",
        "shuffle": "خلط العيّنات",
        "prices_label": "قائمة الأسعار",
        "no_items": "لم يتم اكتشاف أي منتج. جرّب تخفيض حد الثقة.",
        "detected_count": "تم اكتشاف {n} منتج عبر {k} فئة.",
        "subtotal": "المجموع الفرعي", "vat_label": "ضريبة القيمة المضافة (15٪)", "total": "الإجمالي",
        "currency": "ر.س", "invoice": "فاتورة",
        "thank_you": "شكراً لتسوقكم.",
        "tagline_receipt": "نظام كاشير ذكي بالذكاء الاصطناعي",
        "print": "طباعة الفاتورة",
    },
}

# ---------------------------------------------------------------------------
# Model & helpers
# ---------------------------------------------------------------------------
model = YOLO(str(MODEL_PATH))


def list_test_images():
    if not TEST_IMAGES_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(str(p) for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in exts)


ALL_SAMPLES = list_test_images()


def pick_random_samples(n=3):
    return random.sample(ALL_SAMPLES, min(n, len(ALL_SAMPLES))) if ALL_SAMPLES else []


def build_cart(detections):
    cart = {}
    for cls in detections:
        info = CATEGORY_PRICES.get(cls, {"ar": cls, "en": cls, "price": 0.0, "emoji": "📦"})
        if cls not in cart:
            cart[cls] = {"info": info, "count": 0}
        cart[cls]["count"] += 1
    return cart


def render_receipt(cart, store_name, show_vat, lang):
    txt = T[lang]
    direction = "rtl" if lang == "ar" else "ltr"
    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])

    subtotal = sum(it["info"]["price"] * it["count"] for it in cart.values())
    vat = subtotal * 0.15 if show_vat else 0
    total = subtotal + vat
    now = datetime.now()
    invoice_no = f"VC-{random.randint(10000, 99999)}"

    rows = "".join(
        f'<div class="row"><span>{it["info"]["emoji"]} {label(it["info"])} ×{it["count"]}</span>'
        f'<span>{it["info"]["price"] * it["count"]:.2f} {txt["currency"]}</span></div>'
        for it in cart.values()
    )
    vat_row = (
        f'<div class="row vat"><span>{txt["vat_label"]}</span>'
        f'<span>{vat:.2f} {txt["currency"]}</span></div>'
        if show_vat else ""
    )

    return f"""
<div id="vc-receipt-wrap" dir="{direction}">
<style>
#vc-receipt-wrap {{ display:flex; flex-direction:column; align-items:center; margin:8px 0; }}
#vc-receipt {{ background:#FFFEF5; border:1px solid #E8E0C8; border-radius:6px;
    padding:24px 22px; font-family:'Courier New',monospace; max-width:400px;
    width:100%; color:#1a1a2e; }}
#vc-receipt .hd {{ text-align:center; border-bottom:2px dashed #E8E0C8;
    padding-bottom:12px; margin-bottom:12px; }}
#vc-receipt .store {{ font-size:1.3rem; font-weight:700; color:#2D3561; letter-spacing:2px; }}
#vc-receipt .sub {{ font-size:0.72rem; color:#6c757d; margin-top:4px; }}
#vc-receipt .row {{ display:flex; justify-content:space-between; padding:5px 0;
    font-size:0.85rem; border-bottom:1px dotted #e0d8c0; }}
#vc-receipt .row.vat {{ color:#6c757d; font-size:0.78rem; }}
#vc-receipt .total {{ display:flex; justify-content:space-between;
    border-top:2px solid #2D3561; padding-top:10px; margin-top:8px;
    font-weight:700; color:#2D3561; }}
#vc-receipt .ft {{ text-align:center; margin-top:14px; padding-top:12px;
    border-top:2px dashed #E8E0C8; font-size:0.72rem; color:#6c757d; }}
#vc-receipt .barcode {{ font-size:1.8rem; letter-spacing:3px; color:#2D3561; margin:8px 0 4px; }}
#vc-print-btn {{ margin-top:14px; padding:10px 24px; background:#2D3561; color:white;
    border:none; border-radius:10px; font-weight:700; cursor:pointer; }}
#vc-print-btn:hover {{ opacity:0.9; }}
@media print {{
    body * {{ visibility:hidden !important; }}
    #vc-receipt, #vc-receipt * {{ visibility:visible !important; }}
    #vc-receipt {{ position:absolute; left:0; top:0; border:none; box-shadow:none; }}
    #vc-print-btn {{ display:none !important; }}
}}
</style>
<div id="vc-receipt">
  <div class="hd">
    <div class="store">🛒 {store_name}</div>
    <div class="sub">{txt['tagline_receipt']}</div>
    <div class="sub">{now.strftime('%Y-%m-%d')} &nbsp; {now.strftime('%H:%M:%S')}</div>
    <div class="sub">{txt['invoice']}: {invoice_no}</div>
  </div>
  {rows}
  <div class="row" style="margin-top:6px;"><span>{txt['subtotal']}</span>
    <span>{subtotal:.2f} {txt['currency']}</span></div>
  {vat_row}
  <div class="total"><span>{txt['total']}</span>
    <span>{total:.2f} {txt['currency']}</span></div>
  <div class="ft">
    <div class="barcode">||||| |||| ||||| ||||</div>
    <div>{invoice_no}</div>
    <div style="margin-top:8px;">{txt['thank_you']}</div>
  </div>
</div>
<button id="vc-print-btn" onclick="window.print()">{txt['print']}</button>
</div>
"""


def predict(image, conf, store_name, show_vat, lang):
    txt = T[lang]
    if image is None:
        return None, "", ""

    img = ImageOps.exif_transpose(image).convert("RGB")
    result = model.predict(img, conf=conf, device="cpu", verbose=False)[0]
    # result.plot() returns a BGR ndarray; flip channels and copy() so PIL gets a contiguous RGB array.
    annotated = Image.fromarray(result.plot()[:, :, ::-1].copy())

    classes = [model.names[int(b.cls)] for b in result.boxes]
    if not classes:
        return annotated, f"**{txt['no_items']}**", ""

    cart = build_cart(classes)
    n_items = sum(it["count"] for it in cart.values())
    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])

    summary_lines = [f"**{txt['detected_count'].format(n=n_items, k=len(cart))}**", ""]
    for it in cart.values():
        info = it["info"]
        summary_lines.append(
            f"- {info['emoji']} **{label(info)}** ×{it['count']} — "
            f"`{info['price'] * it['count']:.2f} {txt['currency']}`"
        )
    summary = "\n".join(summary_lines)
    receipt = render_receipt(cart, store_name, show_vat, lang)
    return annotated, summary, receipt


def price_list_markdown(lang):
    label = (lambda info: info["ar"]) if lang == "ar" else (lambda info: info["en"])
    cur = T[lang]["currency"]
    rows = [f"| | | |", "|---|---|---|"]
    for info in CATEGORY_PRICES.values():
        rows.append(f"| {info['emoji']} | {label(info)} | **{info['price']:.2f} {cur}** |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.fillable { max-width: none !important; width: 100% !important; }
.gradio-container .row { flex-wrap: nowrap !important; }
.gradio-container .row .column { min-width: 0 !important; }
.lang-toggle button { min-width: 90px; }

/* RTL mode */
.rtl-mode, .rtl-mode * { direction: rtl !important; }
.rtl-mode .gradio-container,
.rtl-mode label,
.rtl-mode .markdown,
.rtl-mode .prose { text-align: right !important; }

/* In RTL the labels reorder via direction:rtl (max on left, min on right).
   Flip the range track horizontally so the orange fill grows from the "0" side
   toward the thumb, and dragging LEFT increases the value. */
.rtl-mode .slider_input_container input[type="range"] { transform: scaleX(-1); }
"""


def toggle_lang(lang):
    new_lang = "ar" if lang == "en" else "en"
    t = T[new_lang]
    btn_label = "AR 🌐" if new_lang == "en" else "EN 🌐"
    rtl_class = "rtl-mode" if new_lang == "ar" else ""
    return (
        new_lang,
        gr.update(value=btn_label),
        gr.update(value=f"# {t['title']}\n{t['tagline']}"),
        gr.update(label=t["image_label"]),
        gr.update(label=t["confidence"]),
        gr.update(label=t["store"]),
        gr.update(label=t["vat"]),
        gr.update(value=t["scan"]),
        gr.update(label=t["detections"]),
        gr.update(label=t["samples"]),
        gr.update(value=t["shuffle"]),
        gr.update(label=t["prices_label"], value=price_list_markdown(new_lang)),
        gr.update(elem_classes=[rtl_class] if rtl_class else []),
    )


def shuffle_samples():
    return gr.update(samples=[[p] for p in pick_random_samples(3)])


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="orange"),
               title="Vision Checkout", fill_width=True) as demo:
    lang_state = gr.State("en")
    root = gr.Column(elem_classes=[])

    with root:
        with gr.Row():
            with gr.Column(scale=6):
                header_md = gr.Markdown(f"# {T['en']['title']}\n{T['en']['tagline']}")
            with gr.Column(scale=1, min_width=120, elem_classes=["lang-toggle"]):
                lang_btn = gr.Button("AR 🌐", size="sm")

        with gr.Row():
            # ── LEFT column: results (detections image + receipt) ──
            with gr.Column(scale=1):
                annotated_out = gr.Image(label=T["en"]["detections"], height=380,
                                         interactive=False)
                summary_out = gr.Markdown()
                receipt_out = gr.HTML()

            # ── RIGHT column: upload, parameters, samples ──
            with gr.Column(scale=1):
                image_in = gr.Image(label=T["en"]["image_label"], type="pil",
                                    sources=["upload", "clipboard"], height=380)

                with gr.Group():
                    conf_slider = gr.Slider(0.0, 1.0, value=0.25, step=0.05,
                                            label=T["en"]["confidence"])
                    store_in = gr.Textbox(label=T["en"]["store"], value="VISION MART")
                    vat_in = gr.Checkbox(label=T["en"]["vat"], value=True)

                scan_btn = gr.Button(T["en"]["scan"], variant="primary")

                with gr.Accordion(T["en"]["prices_label"], open=False) as prices_acc:
                    prices_md = gr.Markdown(price_list_markdown("en"))

                samples_gallery = gr.Dataset(
                    components=[image_in],
                    samples=[[p] for p in pick_random_samples(3)],
                    label=T["en"]["samples"],
                    samples_per_page=3,
                )
                shuffle_btn = gr.Button(T["en"]["shuffle"], size="sm")

    # ── wiring ──
    scan_btn.click(
        predict,
        inputs=[image_in, conf_slider, store_in, vat_in, lang_state],
        outputs=[annotated_out, summary_out, receipt_out],
    )

    lang_btn.click(
        toggle_lang,
        inputs=[lang_state],
        outputs=[lang_state, lang_btn, header_md, image_in, conf_slider, store_in,
                 vat_in, scan_btn, annotated_out, samples_gallery, shuffle_btn,
                 prices_md, root],
    )

    shuffle_btn.click(shuffle_samples, outputs=[samples_gallery])

    samples_gallery.click(lambda x: x[0], inputs=[samples_gallery], outputs=[image_in])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
