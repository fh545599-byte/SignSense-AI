import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE = Path(__file__).parent
IMG_SIZE = 32

# ---- Project Resources ----
GITHUB_URL = "https://github.com/fh545599-byte/SignSense-AI"

APP_URL = "https://signsense-ai-gbmrkxzxzmrgkewc7wgdfa.streamlit.app/"

REPORT_URL = "https://github.com/fh545599-byte/SignSense-AI/blob/main/Documentation/SignSense_AI%20Project%20Report.docx"

PPT_URL = "https://github.com/fh545599-byte/SignSense-AI/blob/main/Documentation/SignSense_AI%20Project%20Presentation.pptx"

YOUTUBE_URL = "https://youtu.be/kBSCaga3_IM"

COLAB_URL = "https://colab.research.google.com/drive/1cwysgM-d7CauJfMmzlO6WjpBZxXIzeDM"

DATASET_URL = "https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"

st.set_page_config(page_title="SignSense AI", page_icon="🚦", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Sora:wght@500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, li, label {font-family: 'DM Sans', sans-serif;}
h1, h2, h3 {font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.4rem; max-width: 1120px;}

[data-testid="stSidebar"] {border-right: 1px solid #22305A;}
.brand {font-family: 'Sora'; font-size: 1.45rem; font-weight: 700; color: #E8EEFF;}
.brand i {font-style: normal; color: #38BDF8;}
.brand-sub {color: #8B98B8; font-size: .85rem; margin-bottom: 1.2rem;}
.hist {display: flex; justify-content: space-between; font-size: .88rem; padding: .35rem 0; border-bottom: 1px solid #1C2848; color: #C5CFEA;}
.hist span:last-child {color: #38BDF8;}

.page-title {font-family: 'Sora'; font-size: 2.5rem; font-weight: 700; color: #E8EEFF; line-height: 1.1; margin: 0; letter-spacing: -0.03em;}
.page-sub {color: #8B98B8; font-size: 1.05rem; margin: .6rem 0 1.6rem 0; max-width: 620px;}

.card {background: #111A30; border: 1px solid #22305A; border-radius: 16px; padding: 1.4rem 1.5rem;}
.card + .card {margin-top: 1rem;}
.label {color: #8B98B8; font-size: .85rem; margin-bottom: .25rem;}
.pred-name {font-family: 'Sora'; font-size: 1.7rem; font-weight: 600; color: #E8EEFF; line-height: 1.2;}
.chip {display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 500; margin: .6rem .4rem 0 0;}
.row {display: flex; align-items: center; gap: 1.5rem;}

.bar-row {margin: .75rem 0 0 0;}
.bar-label {display: flex; justify-content: space-between; font-size: .92rem; color: #C5CFEA; margin-bottom: 4px;}
.bar-track {background: #1C2848; border-radius: 6px; height: 8px; overflow: hidden;}
.bar-fill {background: #3A4C7C; height: 8px; border-radius: 6px;}
.bar-fill.first {background: #38BDF8;}

.stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.4rem;}
.stat {background: #111A30; border: 1px solid #22305A; border-radius: 14px; padding: 1rem 1.2rem;}
.stat b {display: block; font-family: 'Sora'; font-size: 1.5rem; color: #E8EEFF;}
.stat span {color: #8B98B8; font-size: .85rem;}

.lib-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem;}
.lib {background: #111A30; border: 1px solid #22305A; border-radius: 14px; padding: .9rem 1.1rem;}
.lib .id {color: #8B98B8; font-size: .8rem;}
.lib .nm {color: #E8EEFF; font-weight: 500; margin-top: .15rem;}

.pixel img {image-rendering: pixelated; border-radius: 10px; border: 1px solid #22305A;}

[data-testid="stFileUploader"] section {background: #111A30; border: 1.5px dashed #3A4C7C; border-radius: 16px; padding: 1.8rem;}
[data-testid="stFileUploader"] button {background: #38BDF8; color: #06101F; border: 0; border-radius: 8px; font-weight: 600;}
.stTabs [data-baseweb="tab"] {font-family: 'Sora'; font-size: .95rem;}
.stDownloadButton button {background: transparent; border: 1px solid #38BDF8; color: #38BDF8; border-radius: 8px;}

@media (max-width: 800px) {
  .page-title {font-size: 1.9rem;}
  .stats {grid-template-columns: repeat(2, 1fr);}
  .lib-grid {grid-template-columns: 1fr;}
  .row {flex-direction: column; align-items: flex-start;}
}
</style>
""",
    unsafe_allow_html=True,
)

# ---- Sign categories (GTSRB class IDs) ----
CATEGORIES = {
    "Speed limit": ([0, 1, 2, 3, 4, 5, 7, 8], "#F87171", "Limits the maximum speed"),
    "Prohibitory": ([9, 10, 15, 16, 17], "#FB7185", "Prohibits an action or vehicle type"),
    "Warning": ([11] + list(range(18, 32)), "#FBBF24", "Warns about a hazard ahead"),
    "Priority": ([12, 13, 14], "#A78BFA", "Defines who has the right of way"),
    "Mandatory": (list(range(33, 41)), "#38BDF8", "Instructs the driver what to do"),
    "End of restriction": ([6, 32, 41, 42], "#34D399", "Cancels a previous restriction"),
}
CLASS_CATEGORY = {c: name for name, (ids, _, _) in CATEGORIES.items() for c in ids}


# ---------------- Model helpers ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(BASE / "model" / "signsense_model.keras")


@st.cache_resource
def load_graph():
    """Re-connect the loaded layers on a fresh Input so inner outputs can be read.
    Training-only augmentation layers are skipped. Weights are shared, nothing is retrained."""
    m = load_model()
    aug = (tf.keras.layers.RandomRotation, tf.keras.layers.RandomZoom, tf.keras.layers.RandomTranslation)
    kept = [l for l in m.layers if not isinstance(l, aug)]
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x, taps = inp, []
    for layer in kept:
        x = layer(x)
        taps.append(x)
    return kept, inp, taps


@st.cache_resource
def load_grad_model():
    kept, inp, taps = load_graph()
    idx = max(i for i, l in enumerate(kept) if isinstance(l, tf.keras.layers.Conv2D))
    return tf.keras.models.Model(inp, [taps[idx], taps[-1]])


@st.cache_resource
def load_multi():
    """Feature extractor (output of the Dense-256 layer), used only for unknown-sign detection."""
    kept, inp, taps = load_graph()
    return tf.keras.models.Model(inp, taps[-3])


@st.cache_resource
def load_ood():
    """Statistics of the known signs, used to detect signs the model has never seen."""
    path = BASE / "model" / "ood_stats.npz"
    if UNKNOWN_ID is not None or not path.exists():  # stats only apply to the 43-class model
        return None
    d = np.load(path)
    return {
        "means": d["means"].astype("float64"),
        "precision": d["precision"].astype("float64"),
        "thr": float(d["thr99"]),
    }


def infer(X: np.ndarray):
    """Class probabilities (from the original model) plus a distance-to-known-signs score.
    The distance is None if the stats file is missing or the check fails."""
    probs = load_model().predict(X, verbose=0)
    dist = None
    ood = load_ood()
    if ood is not None:
        try:
            f = load_multi().predict(X, verbose=0).astype("float64")
            d = np.stack([(((f - mu) @ ood["precision"]) * (f - mu)).sum(axis=1) for mu in ood["means"]], axis=1)
            dist = d.min(axis=1)
        except Exception:
            dist = None
    return probs, dist


def ood_ratio(dist):
    """How far the image is from known signs, as a multiple of the calibrated threshold (1.0 = at the limit)."""
    ood = load_ood()
    if ood is None or dist is None:
        return None
    return float(dist) / ood["thr"]


def is_unknown(dist) -> bool:
    ood = load_ood()
    if ood is None or dist is None or ood_factor is None:
        return False
    return bool(dist > ood["thr"] * ood_factor)


@st.cache_data
def load_class_names():
    with open(BASE / "model" / "class_names.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def prepare(img: Image.Image, enhance: bool, square: bool) -> Image.Image:
    """Clean up the user's image before it is resized for the model."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if square:
        side = min(img.size)
        img = ImageOps.fit(img, (side, side), Image.LANCZOS)
    if enhance:
        img = ImageOps.autocontrast(img, cutoff=1)
    return img


def to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype="float32")  # rescaling (/255) happens inside the model


def gradcam_overlay(img: Image.Image, class_idx: int, size: int = 256) -> Image.Image:
    """Heatmap showing which image regions pushed the model toward its prediction."""
    x = tf.constant(np.expand_dims(to_array(img), 0))
    grad_model = load_grad_model()
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    heat = Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.BICUBIC)
    colored = ImageOps.colorize(heat, black="#0A0F1F", mid="#38BDF8", white="#FBBF24", midpoint=128)
    base = img.resize((size, size), Image.LANCZOS)
    return Image.blend(base, colored, 0.45)


def gauge(conf: float, threshold: float, color: str = None) -> str:
    color = color or ("#34D399" if conf >= 85 else "#FBBF24" if conf >= threshold else "#FB7185")
    circ = 339.29
    off = circ * (1 - conf / 100)
    return f"""
<svg width="140" height="140" viewBox="0 0 140 140" style="flex-shrink:0">
  <circle cx="70" cy="70" r="54" stroke="#1C2848" stroke-width="12" fill="none"/>
  <circle cx="70" cy="70" r="54" stroke="{color}" stroke-width="12" fill="none" stroke-linecap="round"
          stroke-dasharray="{circ}" stroke-dashoffset="{off:.2f}" transform="rotate(-90 70 70)"/>
  <text x="70" y="75" text-anchor="middle" fill="#E8EEFF" font-size="26" font-weight="600"
        font-family="Sora, sans-serif">{conf:.1f}%</text>
  <text x="70" y="95" text-anchor="middle" fill="#8B98B8" font-size="11"
        font-family="DM Sans, sans-serif">confidence</text>
</svg>"""


def stretch(fn, *args, **kwargs):
    """Full-width element that works on both old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def chip(text, color):
    return f'<span class="chip" style="background:{color}22;color:{color};border:1px solid {color}55;">{text}</span>'


def link_line(label, url):
    return f"- **{label}:** [{url}]({url})" if url else f"- **{label}:** coming soon"


model = load_model()
class_names = load_class_names()
N_SIGNS = 43
UNKNOWN_ID = 43 if 43 in class_names else None  # extra "Not a traffic sign" class, if the model has it

for key, default in [("history", []), ("seen", set())]:
    if key not in st.session_state:
        st.session_state[key] = default

ood_factor = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('<div class="brand">Sign<i>Sense</i> AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Traffic sign recognition</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Recognize", "Batch analysis", "Sign library", "About the project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Settings**")
    enhance = st.checkbox("Auto-enhance contrast", value=False, help="Helps with dark or washed-out photos.")
    square = st.checkbox("Center-crop to square", value=True, help="Avoids squashing wide or tall photos.")
    threshold = st.slider(
        "Recognition threshold (%)", 30, 95, 85,
        help="If the model's confidence is below this value, the image is reported as 'Not recognized' instead of a possibly wrong answer.",
    )
    if load_ood() is not None:
        level = st.select_slider(
            "Unknown-sign rejection",
            options=["Off", "Lenient", "Balanced", "Strict"],
            value="Balanced",
            help="Reports signs that look different from the training photos as not recognized. Works best on real photos; very clean graphics or clipart may be flagged too.",
        )
        ood_factor = {"Off": None, "Lenient": 1.5, "Balanced": 1.0, "Strict": 0.7}[level]
    st.divider()
    st.markdown("**Recent predictions**")
    if st.session_state.history:
        for name, c in st.session_state.history[:6]:
            st.markdown(f'<div class="hist"><span>{name}</span><span>{c:.0f}%</span></div>', unsafe_allow_html=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.seen = set()
            st.rerun()
    else:
        st.caption("Nothing yet. Your predictions will appear here.")
    st.divider()
    st.markdown(f"[GitHub repository]({GITHUB_URL})")

# ======================= Recognize =======================
if page == "Recognize":
    st.markdown('<h1 class="page-title">Recognize any traffic sign</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Upload a photo or take one with your camera. '
        "The model identifies the sign among 43 categories and shows how confident it is.</p>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Input method", ["Upload image", "Use camera"], horizontal=True, label_visibility="collapsed"
    )

    data = None
    if mode == "Upload image":
        f = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        data = f.getvalue() if f else None
    elif mode == "Use camera":
        f = st.camera_input("Take a photo of a sign", label_visibility="collapsed")  # created only when selected
        data = f.getvalue() if f else None
    if data is None:
        st.markdown(
            '<p class="label" style="margin-top:.8rem">Supported formats: PNG, JPG, JPEG. '
            "A clear, close-up photo of a single sign works best. Only the 43 sign types in the Sign library are supported.</p>",
            unsafe_allow_html=True,
        )
    else:
        original = Image.open(io.BytesIO(data))
        image = prepare(original, enhance, square)
        with st.spinner("Analyzing image..."):
            probs, dist = infer(np.expand_dims(to_array(image), 0))
        probs = probs[0]
        dist = None if dist is None else float(dist[0])
        unknown = is_unknown(dist)
        ratio = ood_ratio(dist[0] if isinstance(dist, np.ndarray) else dist)
        top5 = np.argsort(probs[:N_SIGNS])[::-1][:5]
        best = int(top5[0])  # most likely real sign
        conf = float(probs[best]) * 100
        not_a_sign = UNKNOWN_ID is not None and int(probs.argmax()) == UNKNOWN_ID
        rejected = unknown or not_a_sign or conf < threshold
        cat = CLASS_CATEGORY.get(best, "Sign")
        cat_color, cat_desc = (CATEGORIES[cat][1], CATEGORIES[cat][2]) if cat in CATEGORIES else ("#38BDF8", "")

        key = hashlib.md5(data + bytes([enhance, square])).hexdigest()
        if key not in st.session_state.seen:
            st.session_state.seen.add(key)
            st.session_state.history.insert(0, ("Not recognized" if rejected else class_names[best], conf))

        st.write("")
        left, right = st.columns([1, 1.5], gap="large")
        with left:
            st.markdown('<div class="label">Your image</div>', unsafe_allow_html=True)
            stretch(st.image, image)
            st.markdown('<div class="label" style="margin-top:1rem">What the model sees (32x32)</div>', unsafe_allow_html=True)
            st.markdown('<div class="pixel">', unsafe_allow_html=True)
            st.image(image.resize((IMG_SIZE, IMG_SIZE)).resize((120, 120), Image.NEAREST), width=120)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            if rejected:
                if not_a_sign:
                    reason = "The model classified this image as not a traffic sign."
                elif unknown:
                    reason = "This image looks different from the training photos."
                else:
                    reason = f"The model's confidence ({conf:.1f}%) is below the {threshold}% recognition threshold."
                st.markdown(
                    f"""
<div class="card" style="border-color:#FB718566;">
  <div class="row">
    {gauge(conf, threshold, "#FB7185")}
    <div>
      <div class="label">Result</div>
      <div class="pred-name">Sign not recognized</div>
      {chip("Not one of the 43 known signs", "#FB7185")}
      <div class="label" style="margin-top:.6rem">{reason} The sign may not be in the dataset.
      Check the Sign library for supported signs, or try a sharper, closer photo.</div>
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.caption(f"Closest match (unreliable): {class_names[best]} ({conf:.1f}%)")
            else:
                st.markdown(
                    f"""
<div class="card">
  <div class="row">
    {gauge(conf, threshold)}
    <div>
      <div class="label">Detected sign</div>
      <div class="pred-name">{class_names[best]}</div>
      {chip(cat, cat_color)}
      <div class="label" style="margin-top:.6rem">{cat_desc}</div>
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            if UNKNOWN_ID is not None:
                st.caption(f"Probability that this is not a traffic sign: {probs[UNKNOWN_ID] * 100:.1f}%")
            if ratio is not None:
                limit = "off" if ood_factor is None else f"flagged above {ood_factor:.1f}"
                st.caption(f"Familiarity score: {ratio:.2f} ({limit}). Lower means the image looks more like the training signs.")

            st.caption(
                "SignSense recognizes 43 German (GTSRB) sign types. A sign outside this set is matched to the "
                "closest known class, so high confidence does not always mean a correct match."
            )

            t_top, t_cam, t_all = st.tabs(["Top 5", "Where the model looks", "All classes"])

            with t_top:
                bars = ""
                for rank, i in enumerate(top5):
                    p = float(probs[int(i)]) * 100
                    bars += (
                        f'<div class="bar-row"><div class="bar-label"><span>{class_names[int(i)]}</span>'
                        f"<span>{p:.2f}%</span></div>"
                        f'<div class="bar-track"><div class="bar-fill {"first" if rank == 0 else ""}" '
                        f'style="width:{max(p, 0.5):.2f}%"></div></div></div>'
                    )
                st.markdown(f'<div class="card">{bars}</div>', unsafe_allow_html=True)

            with t_cam:
                try:
                    overlay = gradcam_overlay(image, best)
                    c1, c2 = st.columns(2)
                    stretch(c1.image, image.resize((256, 256)), caption="Input")
                    stretch(c2.image, overlay, caption="Grad-CAM heatmap")
                    st.caption(
                        "Warm areas are the regions that influenced the prediction most. "
                        "If the glow sits on the sign itself, the model is looking at the right thing."
                    )
                except Exception as e:
                    st.info(f"Heatmap is not available for this model ({type(e).__name__}).")

            with t_all:
                df = pd.DataFrame({"Probability (%)": probs * 100}, index=[class_names[i] for i in range(len(probs))])
                st.bar_chart(df.sort_values("Probability (%)", ascending=False).head(15))
                st.caption("Showing the 15 most likely classes.")

            report = {
                "prediction": "Not recognized" if rejected else class_names[best],
                "closest_match": class_names[best],
                "unusual_image": unknown,
                "not_a_sign_probability_percent": None if UNKNOWN_ID is None else round(float(probs[UNKNOWN_ID]) * 100, 2),
                "familiarity_ratio": None if ratio is None else round(ratio, 2),
                "class_id": best,
                "category": cat,
                "confidence_percent": round(conf, 2),
                "top5": [
                    {"sign": class_names[int(i)], "probability_percent": round(float(probs[int(i)]) * 100, 2)}
                    for i in top5
                ],
            }
            st.write("")
            st.download_button(
                "Download result (JSON)", json.dumps(report, indent=2), file_name="signsense_result.json", mime="application/json"
            )

# ======================= Batch analysis =======================
elif page == "Batch analysis":
    st.markdown('<h1 class="page-title">Batch analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Upload several sign images at once and get a table of predictions you can download.</p>',
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        "Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if not files:
        st.markdown('<p class="label">Select up to 50 images.</p>', unsafe_allow_html=True)
    else:
        files = files[:50]
        images = [prepare(Image.open(io.BytesIO(f.getvalue())), enhance, square) for f in files]
        with st.spinner(f"Analyzing {len(images)} images..."):
            probs, dists = infer(np.stack([to_array(im) for im in images]))
        ids = probs[:, :N_SIGNS].argmax(axis=1)
        unk = [is_unknown(d) for d in (dists if dists is not None else [None] * len(ids))]
        confs = probs[:, :N_SIGNS].max(axis=1) * 100
        nas = (probs.argmax(axis=1) == UNKNOWN_ID) if UNKNOWN_ID is not None else np.zeros(len(ids), dtype=bool)
        rej = [bool(u) or bool(n) or c < threshold for u, n, c in zip(unk, nas, confs)]

        df = pd.DataFrame(
            {
                "File": [f.name for f in files],
                "Prediction": ["Not recognized" if r else class_names[int(i)] for i, r in zip(ids, rej)],
                "Category": ["-" if r else CLASS_CATEGORY.get(int(i), "Sign") for i, r in zip(ids, rej)],
                "Status": ["Not recognized" if r else "Recognized" for r in rej],
                "Confidence (%)": np.round(confs, 2),
            }
        )
        st.markdown(
            f"""
<div class="stats">
  <div class="stat"><b>{len(df)}</b><span>images analyzed</span></div>
  <div class="stat"><b>{confs.mean():.1f}%</b><span>average confidence</span></div>
  <div class="stat"><b>{sum(rej)}</b><span>not recognized</span></div>
  <div class="stat"><b>{df.loc[df["Status"] == "Recognized", "Prediction"].nunique()}</b><span>distinct signs</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        stretch(st.dataframe, df, hide_index=True)
        st.download_button("Download results (CSV)", df.to_csv(index=False), file_name="signsense_batch.csv", mime="text/csv")

        st.markdown("#### Gallery")
        cols = st.columns(5)
        for i, (im, name, c, r) in enumerate(zip(images, df["Prediction"], confs, rej)):
            with cols[i % 5]:
                stretch(st.image, im)
                st.caption("Not recognized" if r else f"{name} ({c:.0f}%)")

# ======================= Sign library =======================
elif page == "Sign library":
    st.markdown('<h1 class="page-title">Sign library</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">All 43 traffic sign classes the model can recognize, grouped by type.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1.3])
    query = c1.text_input("Search", placeholder="e.g. stop, speed, curve")
    picked = c2.multiselect("Category", list(CATEGORIES.keys()), default=list(CATEGORIES.keys()))

    cards = ""
    count = 0
    for cid in range(43):
        cat = CLASS_CATEGORY.get(cid, "Sign")
        if cat not in picked or query.lower() not in class_names[cid].lower():
            continue
        count += 1
        cards += (
            f'<div class="lib"><div class="id">Class {cid}</div>'
            f'<div class="nm">{class_names[cid]}</div>{chip(cat, CATEGORIES[cat][1])}</div>'
        )
    st.caption(f"{count} of 43 classes")
    st.markdown(f'<div class="lib-grid">{cards}</div>', unsafe_allow_html=True)

# ======================= About =======================
else:
    st.markdown('<h1 class="page-title">About the project</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">SignSense AI is a deep learning project that classifies traffic signs '
        "into 43 categories using a convolutional neural network.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="stats">
  <div class="stat"><b>43</b><span>sign classes</span></div>
  <div class="stat"><b>39,209</b><span>training images</span></div>
  <div class="stat"><b>98.44%</b><span>test accuracy</span></div>
  <div class="stat"><b>32x32</b><span>input size (RGB)</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    tab_data, tab_model, tab_perf, tab_links = st.tabs(["Dataset", "Model", "Performance", "Resources"])

    with tab_data:
        st.write(
            "The model is trained on GTSRB (German Traffic Sign Recognition Benchmark): "
            "39,209 training images across 43 classes, plus a separate official test set of 12,630 images."
        )
        st.markdown(f"[View the dataset on Kaggle]({DATASET_URL})")
        st.markdown("**Preprocessing**")
        st.write(
            "Images are resized to 32x32 RGB and normalized to the 0-1 range. "
            "Rotation, zoom, shift and contrast augmentation is applied during training."
        )
        if UNKNOWN_ID is not None:
            st.markdown("**Extra data for unrecognized images**")
            st.write(
                "A 44th class, Not a traffic sign, is trained with everyday photos (CIFAR-100) and "
                "generated sign-like images (other shapes, colours and pictograms that are not GTSRB signs), "
                "so the model can reject images instead of forcing a match."
            )

    with tab_model:
        st.markdown(
            "- 5 convolutional layers (32, 32, 64, 64, 128 filters)\n"
            "- BatchNormalization, MaxPooling and Dropout for stable training\n"
            "- Dense(256) layer with Dropout(0.5), then a softmax output layer\n"
            "- Adam optimizer with EarlyStopping and ReduceLROnPlateau\n"
            "- Trained on Google Colab (T4 GPU)"
        )
        if UNKNOWN_ID is not None:
            st.markdown(
                "- 44 outputs: the 43 GTSRB signs plus **Not a traffic sign**, so unrelated or unknown "
                "images are reported as not recognized"
            )
        with st.expander("Keras model summary"):
            lines = []
            model.summary(print_fn=lambda x: lines.append(x))
            st.code("\n".join(lines))

    with tab_perf:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", "98.44%")
        m2.metric("Macro precision", "98.10%")
        m3.metric("Macro recall", "97.75%")
        m4.metric("Macro F1-score", "97.82%")
        r1, r2, r3 = st.columns(3)
        r1.metric("Real signs wrongly rejected", "0.35%")
        r2.metric("Clean-graphic signs recognized", "98.47%")
        r3.metric("Unseen everyday photos rejected", "98.80%")
        st.caption(
            "Accuracy, precision, recall and F1 are measured on the official GTSRB test set, which was never "
            "seen during training. Rejection of everyday photos is measured on unseen CIFAR-100 images. "
            "Signs from other countries or sign types outside GTSRB may still occasionally be matched to a "
            "known class."
        )
        for name, title in [("training_curves.png", "Training curves"), ("confusion_matrix.png", "Confusion matrix")]:
            p = BASE / "assets" / name
            if p.exists():
                st.markdown(f"**{title}**")
                stretch(st.image, str(p))

    with tab_links:
        st.markdown(
            "\n".join(
                [
                    link_line("Live Application", APP_URL),
                    link_line("GitHub repository", GITHUB_URL),
                    link_line("Project report", REPORT_URL),
                    link_line("Presentation", PPT_URL),
                    link_line("YouTube video", YOUTUBE_URL),
                    link_line("Google Colab Notebook", COLAB_URL),
                    link_line("Dataset (Kaggle)", DATASET_URL),
                ]
            )
        )