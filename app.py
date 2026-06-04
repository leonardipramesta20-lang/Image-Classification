import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Image Classification App",
    page_icon="📸",
    layout="wide"
)

# Informasi Mahasiswa
st.sidebar.markdown("### 📝 Identitas Mahasiswa")
st.sidebar.info("""
**Nama:** JAYAGHU LEONARDI PRAMESTA
**NIM:** 032400037
**Prodi:** ELEKTRO MEKANIKA
**Praktikum:** IMAGE CLASSIFICATION
""")

# Pilihan Menu Utama
st.sidebar.markdown("### 🎛️ Menu Utama")
menu = st.sidebar.radio("Pilih Halaman:", ["Pelatihan Model (Training)", "Prediksi Gambar Baru (Prediction)"])

# Target size sesuai dengan rancangan notebook
TARGET_SIZE = (128, 128)

# ====================================================================
# HALAMAN 1: PELATIHAN MODEL
# ====================================================================
if menu == "Pelatihan Model (Training)":
    st.title("🏋️ Pelatihan Model CNN (Image Classification)")
    st.write("Halaman ini digunakan untuk mengunggah dataset zip, mengonfigurasi parameter, dan melatih model CNN secara interaktif.")
    
    st.header("1. Persiapan Dataset")
    st.write("Unggah dataset Anda dalam format `.zip` yang berisi struktur folder `train` dan `validation` (masing-masing memiliki subfolder kelas contohnya `positive` dan `negative`).")
    
    uploaded_zip = st.file_uploader("Unggah Dataset (.zip)", type=["zip"])
    
    if uploaded_zip is not None:
        dataset_extract_path = "extracted_dataset"
        
        with st.spinner("Mengekstrak dataset..."):
            with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                zip_ref.extractall(dataset_extract_path)
        st.success("Dataset berhasil diekstrak!")
        
        # Deteksi otomatis jalur direktori di dalam zip
        base_dir = dataset_extract_path
        extracted_dirs = os.listdir(dataset_extract_path)
        if len(extracted_dirs) == 1 and os.path.isdir(os.path.join(dataset_extract_path, extracted_dirs[0])):
            base_dir = os.path.join(dataset_extract_path, extracted_dirs[0])
            
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')
        
        st.header("2. Parameter Pelatihan")
        epochs = st.slider("Jumlah Epochs", min_value=1, max_value=50, value=20)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=32)
        
        if st.button("🚀 Mulai Pelatihan Model"):
            st.header("3. Proses Pelatihan")
            
            # Arsitektur model berdasarkan Jupyter Notebook asli
            model = Sequential([
                Conv2D(128, (3,3), activation='relu', input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
                MaxPooling2D(pool_size=(2,2)),
                
                Conv2D(64, (3,3), activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            st.info("Arsitektur CNN berhasil diinisialisasi.")
            
            # Augmentasi Data menggunakan ImageDataGenerator
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            validation_datagen = ImageDataGenerator(rescale=1./255)
            
            try:
                train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=TARGET_SIZE,
                    batch_size=batch_size,
                    class_mode='binary'
                )

                validation_generator = validation_datagen.flow_from_directory(
                    validation_dir,
                    target_size=TARGET_SIZE,
                    batch_size=batch_size,
                    class_mode='binary'
                )
                
                st.session_state['class_indices'] = train_generator.class_indices
                st.write(f"Label Kelas Terdeteksi: `{train_generator.class_indices}`")
                
                # Callback untuk menampilkan progress di UI Streamlit
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        st.write(f"🔹 **Epoch {epoch+1}/{epochs}** - Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.4f} | Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_accuracy']:.4f}")

                with st.spinner("Model sedang dilatih..."):
                    history = model.fit(
                        train_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        callbacks=[StreamlitCallback()],
                        verbose=0
                    )
                st.success("Pelatihan selesai!")
                
                model_filename = "image_classification_model.h5"
                model.save(model_filename)
                
                st.session_state['history'] = history.history
                st.session_state['model_path'] = model_filename
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data/melatih model: {e}")
                st.warning("Pastikan folder di dalam zip Anda memiliki struktur yang benar (terdapat folder 'train' dan 'validation').")
        
        # Visualisasi Metrik Pelatihan jika ada history
        if 'history' in st.session_state:
            st.header("4. Visualisasi Hasil")
            history_dict = st.session_state['history']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history_dict['accuracy'], label='Train Accuracy', color='#1f77b4')
            ax1.plot(history_dict['val_accuracy'], label='Val Accuracy', color='#ff7f0e')
            ax1.set_title('Akurasi')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(history_dict['loss'], label='Train Loss', color='#1f77b4')
            ax2.plot(history_dict['val_loss'], label='Val Loss', color='#ff7f0e')
            ax2.set_title('Loss')
            ax2.legend()
            ax2.grid(True)
            
            st.pyplot(fig)

# ====================================================================
# HALAMAN 2: PREDIKSI GAMBAR BARU
# ====================================================================
elif menu == "Prediksi Gambar Baru (Prediction)":
    st.title("🔍 Prediksi Gambar Baru")
    st.write("Gunakan model yang telah dilatih atau unggah model `.h5` eksternal Anda untuk melakukan klasifikasi gambar.")
    
    st.header("1. Muat Model")
    model_option = st.radio("Pilih sumber model:", ["Gunakan model hasil pelatihan sesi ini", "Unggah file model (.h5) kustom"])
    
    model = None
    model_loaded = False
    
    if model_option == "Gunakan model hasil pelatihan sesi ini":
        if 'model_path' in st.session_state and os.path.exists(st.session_state['model_path']):
            model = load_model(st.session_state['model_path'])
            model_loaded = True
            st.success("Model dari sesi pelatihan aktif berhasil dimuat!")
        else:
            st.warning("Model belum dilatih pada sesi ini. Selesaikan pelatihan terlebih dahulu atau gunakan pilihan unggah model.")
    else:
        uploaded_model_file = st.file_uploader("Unggah file model (.h5)", type=["h5"])
        if uploaded_model_file is not None:
            with open("temp_uploaded_model.h5", "wb") as f:
                f.write(uploaded_model_file.read())
            model = load_model("temp_uploaded_model.h5")
            model_loaded = True
            st.success("Model kustom berhasil dimuat!")
            
    if model_loaded:
        st.header("2. Unggah Gambar")
        uploaded_img = st.file_uploader("Pilih gambar untuk diklasifikasi...", type=["jpg", "jpeg", "png"])
        
        if uploaded_img is not None:
            img_display = Image.open(uploaded_img)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img_display, caption="Gambar Input", use_container_width=True)
                
            with col2:
                st.subheader("Hasil Prediksi")
                with st.spinner("Menganalisis gambar..."):
                    img_resized = img_display.resize(TARGET_SIZE)
                    img_array = img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    prediction = model.predict(img_array)
                    probability = prediction[0][0]
                    
                    # Mapping label default jika class_indices tidak tersedia
                    if 'class_indices' in st.session_state:
                        class_mapping = st.session_state['class_indices']
                        idx_to_class = {v: k for k, v in class_mapping.items()}
                    else:
                        idx_to_class = {0: "Negative (Kelas 0)", 1: "Positive (Kelas 1)"}
                    
                    if probability > 0.5:
                        predicted_idx = 1
                        confidence = probability
                    else:
                        predicted_idx = 0
                        confidence = 1 - probability
                        
                    predicted_label = idx_to_class.get(predicted_idx, "Unknown")
                    
                    st.metric(label="Hasil Klasifikasi", value=predicted_label)
                    st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence * 100:.2f}%")
                    st.progress(float(probability))
                    st.caption(f"Nilai mentah output sigmoid: {probability:.4f}")
