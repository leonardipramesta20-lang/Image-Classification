import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# =====================================
# IDENTITAS
# =====================================
NAMA       = "Jayaghu Leonardi Pramesta"
NIM        = "032400037"
PRODI      = "Elektro Mekanika"
APP_TITLE  = "Deteksi Keretakan Beton — CNN Inspector"
APP_ICON   = "🏗️"

# =====================================
# FUNGSI GRAFIK PDF
# =====================================
def _make_donut_chart_img(total_retak, total_aman):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        [total_retak, total_aman],
        labels=["Retak", "Tidak Retak"],
        colors=["#E74C3C", "#1ABC9C"],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"linewidth": 2, "edgecolor": "white", "width": 0.6},
        textprops={"fontsize": 10}
    )
    ax.set_title("Distribusi Hasil Inspeksi", fontsize=11, fontweight="bold", pad=10)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def _make_bar_chart_img(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n[:20] + "…" if len(n) > 20 else n for n in df["File"].tolist()]
    confs = df["Confidence (%)"].tolist()
    clrs  = ["#E74C3C" if p == "Retak" else "#1ABC9C" for p in df["Prediksi"].tolist()]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(names) * 0.5)), facecolor="white")
    bars = ax.barh(names, confs, color=clrs, edgecolor="white", linewidth=0.8, height=0.6)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title("Confidence Per Citra", fontsize=11, fontweight="bold")
    ax.axvline(50, color="#95A5A6", linestyle="--", linewidth=1, label="Threshold 50%")
    ax.set_facecolor("#FAFAFA")
    for bar, v in zip(bars, confs):
        ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

# =====================================
# FUNGSI GENERATE PDF
# =====================================
def generate_pdf(df, total_img, total_retak, total_aman, persen_retak, persen_aman, catatan_inspeksi=""):
    from reportlab.platypus import Image as RLImage
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle("title",   parent=styles["Title"],   fontSize=15, alignment=TA_CENTER, spaceAfter=4, textColor=colors.HexColor("#1A252F"))
    style_sub     = ParagraphStyle("sub",     parent=styles["Normal"],  fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#7F8C8D"), spaceAfter=2)
    style_heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=11, spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#1A252F"))
    style_normal  = ParagraphStyle("normal",  parent=styles["Normal"],  fontSize=9,  textColor=colors.HexColor("#2C3E50"))
    style_note    = ParagraphStyle("note",    parent=styles["Normal"],  fontSize=9,  textColor=colors.HexColor("#555555"), leftIndent=8)

    story = []

    # HEADER
    story.append(Paragraph("LAPORAN INSPEKSI KERETAKAN BETON", style_title))
    story.append(Paragraph(f"CNN Inspector — {APP_TITLE}", style_sub))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1ABC9C")))
    story.append(Spacer(1, 0.3*cm))

    # IDENTITAS
    story.append(Paragraph("Identitas Inspektor", style_heading))
    tbl_identitas = Table([
        ["Nama Inspektor", NAMA],
        ["NIM",            NIM],
        ["Program Studi",  PRODI],
        ["Tanggal Laporan", datetime.now().strftime("%d %B %Y, %H:%M:%S")],
    ], colWidths=[4.5*cm, 11.5*cm])
    tbl_identitas.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("FONTNAME",       (0,0), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",      (0,0), (0,-1),  colors.HexColor("#1ABC9C")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#F0FEFA"), colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#D5F5ED")),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
    ]))
    story.append(tbl_identitas)
    story.append(Spacer(1, 0.3*cm))

    # RINGKASAN
    story.append(Paragraph("Ringkasan Hasil Inspeksi", style_heading))
    tbl_ringkasan = Table([
        ["Keterangan",           "Jumlah", "Persentase"],
        ["Total Citra Dianalisis", str(total_img),   "100%"],
        ["Terdeteksi Retak",      str(total_retak), f"{persen_retak:.1f}%"],
        ["Kondisi Normal",        str(total_aman),  f"{persen_aman:.1f}%"],
    ], colWidths=[8*cm, 4*cm, 4*cm])
    tbl_ringkasan.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#1A252F")),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("ALIGN",          (1,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0FEFA")]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#D5F5ED")),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
    ]))
    story.append(tbl_ringkasan)
    story.append(Spacer(1, 0.3*cm))

    # CATATAN INSPEKSI (fitur baru)
    if catatan_inspeksi.strip():
        story.append(Paragraph("Catatan Inspeksi", style_heading))
        story.append(Paragraph(catatan_inspeksi, style_note))
        story.append(Spacer(1, 0.3*cm))

    # GRAFIK
    story.append(Paragraph("Visualisasi Hasil Inspeksi", style_heading))
    try:
        pie_buf = _make_donut_chart_img(total_retak, total_aman)
        pie_img = RLImage(pie_buf, width=7*cm, height=5.6*cm)
        bar_buf = _make_bar_chart_img(df)
        bar_h   = max(4*cm, min(len(df) * 1.1*cm, 10*cm))
        bar_img = RLImage(bar_buf, width=9*cm, height=bar_h)
        chart_tbl = Table([[pie_img, bar_img]], colWidths=[8*cm, 9*cm])
        chart_tbl.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("LEFTPADDING",  (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(chart_tbl)
    except Exception:
        story.append(Paragraph("(Grafik tidak tersedia)", style_normal))
    story.append(Spacer(1, 0.4*cm))

    # DETAIL
    story.append(Paragraph("Detail Hasil Inspeksi Per Citra", style_heading))
    header = [["No", "Nama File", "Resolusi", "Ukuran (KB)", "Status", "Confidence (%)", "Waktu Inspeksi"]]
    rows   = [[str(i+1), row["File"], row.get("Resolusi", "-"),
               str(row.get("Ukuran (KB)", "-")), row["Prediksi"],
               f"{row['Confidence (%)']:.2f}%", row.get("Waktu Prediksi", "-")]
              for i, row in df.iterrows()]
    tbl_detail = Table(header + rows, colWidths=[0.8*cm, 4.5*cm, 2*cm, 2*cm, 2.2*cm, 2.5*cm, 3*cm])
    tbl_detail.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#1A252F")),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 7),
        ("ALIGN",          (0,0), (0,-1),  "CENTER"),
        ("ALIGN",          (2,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0FEFA")]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#D5F5ED")),
        ("LEFTPADDING",    (0,0), (-1,-1), 4),
        ("RIGHTPADDING",   (0,0), (-1,-1), 4),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
        *[("TEXTCOLOR", (4, i+1), (4, i+1), colors.HexColor("#E74C3C"))
          for i, row in df.iterrows() if row["Prediksi"] == "Retak"],
        *[("TEXTCOLOR", (4, i+1), (4, i+1), colors.HexColor("#1ABC9C"))
          for i, row in df.iterrows() if row["Prediksi"] == "Tidak_Retak"],
    ]))
    story.append(tbl_detail)
    story.append(Spacer(1, 0.5*cm))

    # KESIMPULAN OTOMATIS (fitur baru)
    story.append(Paragraph("Kesimpulan", style_heading))
    if persen_retak == 0:
        kesimpulan = f"Seluruh {total_img} citra yang dianalisis tidak menunjukkan indikasi keretakan. Kondisi beton dinyatakan dalam keadaan baik."
    elif persen_retak == 100:
        kesimpulan = f"Seluruh {total_img} citra menunjukkan adanya keretakan. Diperlukan pemeriksaan dan penanganan menyeluruh pada struktur beton."
    else:
        kesimpulan = (f"Dari {total_img} citra yang dianalisis, ditemukan {total_retak} citra ({persen_retak:.1f}%) "
                      f"terdeteksi retak dan {total_aman} citra ({persen_aman:.1f}%) dalam kondisi normal. "
                      f"Disarankan untuk melakukan pengecekan lebih lanjut pada area yang terindikasi retak.")
    story.append(Paragraph(kesimpulan, style_note))
    story.append(Spacer(1, 0.5*cm))

    # FOOTER
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#1ABC9C")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Laporan dibuat otomatis oleh <b>{APP_TITLE}</b> "
        f"pada {datetime.now().strftime('%d %B %Y pukul %H:%M:%S')} | {NAMA} — {NIM}",
        style_normal
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# Custom CSS — tema teal/dark
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1A252F; }
    [data-testid="stSidebar"] * { color: #ECF0F1 !important; }
    .stButton > button { border-radius: 8px; background-color: #1ABC9C; color: white; border: none; }
    .stButton > button:hover { background-color: #17A589; }
    div[data-testid="metric-container"] { background-color: #F0FEFA; border-radius: 10px; padding: 8px; border-left: 4px solid #1ABC9C; }
    .stProgress > div > div { background-color: #1ABC9C; }
</style>
""", unsafe_allow_html=True)

# =====================================
# SESSION STATE
# =====================================
if "history"            not in st.session_state: st.session_state.history            = []
if "images"             not in st.session_state: st.session_state.images             = []
if "model"              not in st.session_state: st.session_state.model              = None
if "labels"             not in st.session_state: st.session_state.labels             = ["Retak", "Tidak_Retak"]
if "last_predicted_key" not in st.session_state: st.session_state.last_predicted_key = None
if "uploader_key"       not in st.session_state: st.session_state.uploader_key       = 0
if "loaded_model_name"  not in st.session_state: st.session_state.loaded_model_name  = None
if "catatan_inspeksi"   not in st.session_state: st.session_state.catatan_inspeksi   = ""

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title(f"{APP_ICON} CNN Inspector")
st.sidebar.caption("Deteksi Keretakan Beton Berbasis AI")
st.sidebar.markdown("---")

menu = st.sidebar.radio("🗂️ Menu", ["🏠 Beranda", "🔍 Inspeksi", "📊 Statistik", "🗒️ Riwayat", "ℹ️ Tentang"])

st.sidebar.markdown("---")

# =====================================
# MODEL UPLOAD
# =====================================
uploaded_model = st.sidebar.file_uploader("📁 Upload Model (.h5)", type=["h5"])

if uploaded_model is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            tmp_path = tmp.name
        st.session_state.model = load_model(tmp_path, compile=False)
        os.unlink(tmp_path)

        try:
            output_layer = st.session_state.model.layers[-1]
            num_classes  = output_layer.output_shape[-1]
            if hasattr(output_layer, "class_names"):
                st.session_state.labels = output_layer.class_names
            elif num_classes == 2:
                st.session_state.labels = ["Retak", "Tidak_Retak"]
        except Exception:
            pass

        if st.session_state.loaded_model_name != uploaded_model.name:
            st.session_state.loaded_model_name = uploaded_model.name
            st.toast(f"✅ Model **{uploaded_model.name}** berhasil dimuat!", icon="🧠")

        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")

model  = st.session_state.model
labels = st.session_state.labels

if model is None:
    st.sidebar.warning("⚠️ Model belum diupload")
else:
    st.sidebar.success("🧠 Model aktif")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏗️ Info Model:**")
    try:
        total_layers = len(model.layers)
        total_params = model.count_params()
        trainable    = sum([p.numpy().size for p in model.trainable_weights])
        input_shape  = model.input_shape
        output_shape = model.output_shape
        st.sidebar.markdown(
            f"- Layers: `{total_layers}`\n"
            f"- Parameter: `{total_params:,}`\n"
            f"- Trainable: `{trainable:,}`\n"
            f"- Input: `{input_shape}`\n"
            f"- Output: `{output_shape}`"
        )
    except Exception:
        st.sidebar.caption("Info arsitektur tidak tersedia.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏷️ Urutan Label:**")
    for i, lbl in enumerate(labels):
        st.sidebar.markdown(f"- Index `{i}` → `{lbl}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚙️ Koreksi Label:**")
    if st.sidebar.toggle("🔄 Balik urutan label"):
        labels = labels[::-1]
        st.session_state.labels = labels

# =====================================
# BERANDA
# =====================================
if menu == "🏠 Beranda":
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("Sistem inspeksi keretakan permukaan beton menggunakan **Convolutional Neural Network (CNN)**.")
    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("⚙️ Fitur Aplikasi")
        fitur = {
            "🧠 Upload Model CNN (.h5)": "Load model langsung dari browser",
            "🖼️ Multi-Citra Sekaligus":  "Analisis banyak gambar dalam satu sesi",
            "📊 Dashboard Statistik":    "Visualisasi distribusi & confidence interaktif",
            "🗒️ Riwayat Sesi":           "Lihat & hapus riwayat inspeksi per batch",
            "📝 Catatan Inspeksi":       "Tambahkan catatan lapangan ke laporan PDF",
            "📄 Laporan PDF Otomatis":   "Ekspor PDF dengan grafik & kesimpulan otomatis",
            "📥 Export CSV":             "Download hasil dalam format spreadsheet",
            "🔄 Koreksi Label":          "Toggle jika prediksi terbalik",
        }
        for k, v in fitur.items():
            st.markdown(f"**{k}** — {v}")

    with col_right:
        st.subheader("👤 Profil Inspektor")
        st.markdown(f"""
        | Keterangan | Detail |
        |---|---|
        | **Nama** | {NAMA} |
        | **NIM** | {NIM} |
        | **Program Studi** | {PRODI} |
        """)
        st.info("💡 Upload model .h5 via sidebar untuk mulai inspeksi.")

        st.subheader("📌 Panduan Singkat")
        st.markdown("""
        1. Upload model `.h5` di sidebar
        2. Buka menu **Inspeksi**
        3. Upload gambar beton
        4. Klik **Mulai Inspeksi**
        5. Download laporan PDF / CSV
        """)

# =====================================
# INSPEKSI
# =====================================
if menu == "🔍 Inspeksi":
    st.title("🔍 Inspeksi Keretakan Beton")

    uploaded_images = st.file_uploader(
        "📂 Upload Citra Beton (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_images:
        st.session_state.images = uploaded_images

    images = st.session_state.images

    # Catatan inspeksi (fitur baru)
    st.session_state.catatan_inspeksi = st.text_area(
        "📝 Catatan Inspeksi (opsional — akan muncul di PDF)",
        value=st.session_state.catatan_inspeksi,
        placeholder="Contoh: Inspeksi dilakukan pada pilar utama jembatan, kondisi cuaca cerah...",
        height=80
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("🗑️ Reset Sesi"):
            st.session_state.images             = []
            st.session_state.history            = []
            st.session_state.last_predicted_key = None
            st.session_state.uploader_key       += 1
            st.session_state.catatan_inspeksi   = ""
            st.toast("✅ Sesi berhasil direset!", icon="🗑️")
            st.rerun()

    if model is not None and len(images) > 0:
        results = []

        st.markdown("---")
        st.subheader("📸 Hasil Inspeksi Citra")
        st.info(f"🏷️ Label aktif: Index 0 = **{labels[0]}**, Index 1 = **{labels[1]}** — Jika terbalik, aktifkan toggle di sidebar.")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            col_count = st.slider("Grid Columns", 2, 4, 3)
        with col_s2:
            threshold = st.slider("⚙️ Threshold (%)", 30, 90, 50,
                help="Batas confidence klasifikasi. Default 50%.")
        with col_s3:
            sort_mode = st.selectbox("🔃 Urutkan Berdasar", ["Urutan Upload", "Confidence Tertinggi", "Confidence Terendah"])

        cols = st.columns(col_count)

        for i, img_file in enumerate(images):
            image         = Image.open(img_file).convert("RGB")
            lebar, tinggi = image.size
            ukuran_file   = img_file.size / 1024
            waktu_pred    = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            with st.spinner("🔬 Menganalisis..."):
                img_arr    = img_to_array(image.resize((150, 150)))
                img_arr    = np.expand_dims(img_arr, axis=0)
                output     = model.predict(img_arr, verbose=0)[0]
                thresh_val = threshold / 100.0

                if len(output) == 1:
                    sigmoid_val = float(output[0])
                    if sigmoid_val >= thresh_val:
                        label = labels[1]
                        conf  = sigmoid_val * 100
                    else:
                        label = labels[0]
                        conf  = (1 - sigmoid_val) * 100
                else:
                    idx   = np.argmax(output)
                    label = labels[idx]
                    conf  = float(output[idx]) * 100

            results.append({
                "File"           : img_file.name,
                "Ukuran (KB)"    : round(ukuran_file, 1),
                "Resolusi"       : f"{lebar}x{tinggi}",
                "Prediksi"       : label,
                "Confidence (%)" : round(conf, 2),
                "Waktu Prediksi" : waktu_pred,
                "_img"           : image,
            })

        df_raw = pd.DataFrame(results)

        # Sorting (fitur baru)
        if sort_mode == "Confidence Tertinggi":
            df_raw = df_raw.sort_values("Confidence (%)", ascending=False).reset_index(drop=True)
        elif sort_mode == "Confidence Terendah":
            df_raw = df_raw.sort_values("Confidence (%)").reset_index(drop=True)

        for i, row in df_raw.iterrows():
            with cols[i % col_count]:
                if row["Prediksi"] == "Retak":
                    st.error(f"⚠️ RETAK — {row['Confidence (%)']:.2f}%")
                else:
                    st.success(f"✔ NORMAL — {row['Confidence (%)']:.2f}%")
                st.image(row["_img"], use_container_width=True)
                st.progress(row["Confidence (%)"] / 100)
                st.caption(f"📁 {row['File']}")
                st.caption(f"📐 {row['Resolusi']}  |  💾 {row['Ukuran (KB)']} KB  |  🕐 {row['Waktu Prediksi']}")

        df = df_raw.drop(columns=["_img"])

        current_key = str(sorted([f.name for f in images]))
        if current_key != st.session_state.last_predicted_key:
            st.session_state.history.append(df)
            st.session_state.last_predicted_key = current_key

        st.markdown("---")
        st.subheader("📋 Ringkasan Inspeksi")

        total_img    = len(df)
        total_retak  = len(df[df["Prediksi"] == "Retak"])
        total_aman   = len(df[df["Prediksi"] == "Tidak_Retak"])
        persen_retak = (total_retak / total_img * 100) if total_img > 0 else 0
        persen_aman  = (total_aman  / total_img * 100) if total_img > 0 else 0

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("🖼️ Total Citra", total_img)
        col_r2.metric("⚠️ Retak",       f"{total_retak} ({persen_retak:.1f}%)")
        col_r3.metric("✅ Normal",       f"{total_aman} ({persen_aman:.1f}%)")

        if total_retak == 0:
            st.success(f"✅ Seluruh **{total_img}** citra dalam kondisi **normal**, tidak ditemukan keretakan.")
        elif total_aman == 0:
            st.error(f"🚨 Seluruh **{total_img}** citra **terdeteksi retak** — diperlukan penanganan segera.")
        else:
            st.warning(f"🔎 Ditemukan **{total_retak} retak ({persen_retak:.1f}%)** dan **{total_aman} normal ({persen_aman:.1f}%)** dari {total_img} citra.")

        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode(),
                               f"inspeksi_beton_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               "text/csv", use_container_width=True)
        with col_dl2:
            pdf_buffer = generate_pdf(
                df, total_img, total_retak, total_aman, persen_retak, persen_aman,
                catatan_inspeksi=st.session_state.catatan_inspeksi
            )
            st.download_button("📄 Download Laporan PDF", pdf_buffer,
                               f"laporan_inspeksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               "application/pdf", use_container_width=True)
    else:
        st.info("ℹ️ Upload model (.h5) via sidebar dan pilih gambar beton untuk memulai inspeksi.")

# =====================================
# STATISTIK
# =====================================
if menu == "📊 Statistik":
    st.title("📊 Dashboard Statistik Inspeksi")

    if len(st.session_state.history) == 0:
        st.warning("⚠️ Belum ada data. Lakukan inspeksi di menu Inspeksi terlebih dahulu.")
    else:
        df     = pd.concat(st.session_state.history, ignore_index=True)
        total  = len(df)
        retak  = len(df[df["Prediksi"] == "Retak"])
        normal = len(df[df["Prediksi"] == "Tidak_Retak"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Citra", total)
        col2.metric("Retak",       retak)
        col3.metric("Normal",      normal)
        col4.metric("Rata-rata Confidence", f"{df['Confidence (%)'].mean():.1f}%")

        st.markdown("---")
        st.subheader("📊 Visualisasi Distribusi")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            donut_fig = go.Figure(data=[go.Pie(
                labels=["Retak", "Normal"],
                values=[retak, normal],
                hole=0.5,
                marker=dict(colors=["#E74C3C", "#1ABC9C"], line=dict(color="white", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13),
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>%{percent}<extra></extra>"
            )])
            donut_fig.update_layout(
                title=dict(text="Distribusi Hasil Inspeksi", font=dict(size=14), x=0.5),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=50, b=20, l=10, r=10), height=320
            )
            st.plotly_chart(donut_fig, use_container_width=True)

        with chart_col2:
            vc = df["Prediksi"].value_counts().reset_index()
            vc.columns = ["Prediksi", "Jumlah"]
            bar_fig = go.Figure(data=[go.Bar(
                x=vc["Prediksi"],
                y=vc["Jumlah"],
                marker_color=["#E74C3C" if p == "Retak" else "#1ABC9C" for p in vc["Prediksi"]],
                text=vc["Jumlah"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Jumlah: %{y}<extra></extra>"
            )])
            bar_fig.update_layout(
                title=dict(text="Jumlah Per Kelas", font=dict(size=14), x=0.5),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#EEEEEE"),
                margin=dict(t=50, b=20, l=10, r=10), height=320
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("📈 Distribusi Confidence")
        hist_fig = px.histogram(
            df, x="Confidence (%)", color="Prediksi", nbins=20,
            color_discrete_map={"Retak": "#E74C3C", "Tidak_Retak": "#1ABC9C"},
            barmode="overlay", opacity=0.8,
        )
        hist_fig.update_layout(
            legend=dict(title="Prediksi"),
            margin=dict(t=20, b=20, l=10, r=10), height=280
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        st.subheader("🔎 Filter & Tabel Data")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            min_conf = st.slider("Min. Confidence (%)", 0, 100, 50)
        with fc2:
            filter_label = st.selectbox("Filter Status", ["Semua", "Retak", "Tidak_Retak"])
        with fc3:
            sort_col = st.selectbox("Urutkan Berdasar", ["Waktu Prediksi", "Confidence (%)"])

        filtered = df[df["Confidence (%)"] >= min_conf]
        if filter_label != "Semua":
            filtered = filtered[filtered["Prediksi"] == filter_label]
        filtered = filtered.sort_values(sort_col, ascending=(sort_col == "Waktu Prediksi"))

        st.caption(f"Menampilkan {len(filtered)} dari {len(df)} data")
        st.dataframe(filtered, use_container_width=True)

# =====================================
# RIWAYAT (fitur baru)
# =====================================
if menu == "🗒️ Riwayat":
    st.title("🗒️ Riwayat Sesi Inspeksi")

    if len(st.session_state.history) == 0:
        st.info("ℹ️ Belum ada riwayat. Lakukan inspeksi terlebih dahulu.")
    else:
        st.markdown(f"Total batch tersimpan: **{len(st.session_state.history)}**")
        for idx, batch_df in enumerate(st.session_state.history):
            with st.expander(f"📦 Batch #{idx+1} — {len(batch_df)} citra | {batch_df['Waktu Prediksi'].iloc[0]}"):
                retak_n  = len(batch_df[batch_df["Prediksi"] == "Retak"])
                normal_n = len(batch_df[batch_df["Prediksi"] == "Tidak_Retak"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(batch_df))
                c2.metric("Retak", retak_n)
                c3.metric("Normal", normal_n)
                st.dataframe(batch_df, use_container_width=True)

        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.history = []
            st.session_state.last_predicted_key = None
            st.toast("✅ Riwayat dihapus!", icon="🗑️")
            st.rerun()

# =====================================
# TENTANG
# =====================================
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("---")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.subheader("🔬 Deskripsi Proyek")
        st.markdown(f"""
        **{APP_TITLE}** adalah sistem inspeksi otomatis berbasis
        **Convolutional Neural Network (CNN)** untuk mendeteksi dan
        mengklasifikasikan keretakan pada permukaan beton melalui
        analisis citra digital. Dibangun menggunakan Streamlit dan TensorFlow/Keras.
        """)

        st.subheader("⚙️ Fitur Teknis")
        st.markdown("""
        - Multi-citra upload & inspeksi sekaligus
        - Sorting hasil berdasarkan confidence
        - Riwayat per batch dengan ekspansi detail
        - Catatan inspeksi lapangan (tersimpan ke PDF)
        - Kesimpulan otomatis pada laporan PDF
        - Auto-detect label order (sigmoid & softmax)
        - Manual label swap toggle
        - Threshold confidence adjustable
        - Export PDF (grafik donut + bar) & CSV
        - Custom tema UI teal/dark
        """)

    with col_a2:
        st.subheader("👤 Identitas Inspektor")
        st.markdown(f"""
        | Keterangan | Detail |
        |---|---|
        | **Nama** | {NAMA} |
        | **NIM** | {NIM} |
        | **Program Studi** | {PRODI} |
        """)

        st.subheader("🛠️ Teknologi yang Digunakan")
        st.markdown("""
        | Library | Kegunaan |
        |---|---|
        | Streamlit | Framework UI |
        | TensorFlow / Keras | Model CNN |
        | Plotly | Grafik interaktif |
        | Matplotlib | Grafik PDF |
        | ReportLab | Generate PDF |
        | Pillow | Image processing |
        """)

        st.success(f"🚀 {APP_TITLE} — Siap Deploy di Streamlit Cloud.")
