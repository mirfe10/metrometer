import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Sayfa Yapılandırması
st.set_page_config(page_title="MetroMeter", layout="wide")

# --- O SİNİR BOZUCU LİNK (ANCHOR) İŞARETLERİNİ VE GÖZ KIRPMALARINI GİZLEYEN CSS SİHRETTİ ---
st.markdown("""
    <style>
    .viewerBadge_link__1SuwF, a.header-anchor {
        display: none !important;
    }
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }
    [data-testid="stMarkdown"] a {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Gün İsimleri Sözlükleri
GUN_ISIMLERI = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
GUN_INPUT_MAP = {"Pazartesi": 0, "Sali": 1, "Carsamba": 2, "Persembe": 3, "Cuma": 4, "Cumartesi": 5, "Pazar": 6}


# Ondalık Saati Normal Saate Çeviren Fonksiyon (Arayüz Gösterimleri İçin)
def format_saat(ondalik_saat):
    saat_kismi = int(ondalik_saat)
    dakika_kismi = int(round((ondalik_saat % 1) * 60))
    if dakika_kismi == 60:
        saat_kismi += 1
        dakika_kismi = 0
    return f"{saat_kismi:02d}:{dakika_kismi:02d}"


# HAZIR CSV DOSYALARININ ADLARI
DB_KULLANICILAR = "buyuk_kullanici_veritabani.csv"
DB_YOLCULUKLAR = "ogrenilebilir_yolcu_verisi.csv"

HAT_DURAKLARI = {
    "M2": ["Yenikapi", "Vezneciler", "Taksim", "Mecidiyekoy", "Levent", "Haciosman"],
    "M7": ["Yildiz", "Mecidiyekoy", "Kagithane", "Alibeykoy", "Mahmutbey"]
}
TUM_DURAKLAR = sorted(list(set(HAT_DURAKLARI["M2"] + HAT_DURAKLARI["M7"])))


# Yoğunluk Durum Fonksiyonu
def yogunluk_hesapla(yolcu_sayisi):
    if yolcu_sayisi <= 2:
        return "Sakin 🟢", "#2ecc71"
    elif yolcu_sayisi <= 5:
        return "Yoğun 🟡", "#f1c40f"
    else:
        return "Aşırı Yoğun 🔴", "#e74c3c"


# ==========================================
# 1. VERİ VE MODEL HAZIRLIĞI
# ==========================================
@st.cache_resource
def modeli_hazirla():
    if not os.path.exists(DB_KULLANICILAR) or not os.path.exists(DB_YOLCULUKLAR):
        st.error(f"❌ HATA: CSV dosyaları bulunamadı! Lütfen dosyaların bu scriptle aynı klasörde olduğundan emin ol.")
        st.stop()

    df_kullanicilar = pd.read_csv(DB_KULLANICILAR)
    df_egitim = pd.read_csv(DB_YOLCULUKLAR)

    df_egitim['Gun_Indeksi'] = pd.to_datetime(df_egitim['Tarih']).dt.dayofweek

    le_id = LabelEncoder()
    le_hat = LabelEncoder()
    le_durak = LabelEncoder()

    df_egitim['ID_Num'] = le_id.fit_transform(df_egitim['Kullanici_ID'])
    df_egitim['Hat_Num'] = le_hat.fit_transform(df_egitim['Binis_Hatti'])

    le_durak.fit(TUM_DURAKLAR)
    df_egitim['Binis_Num'] = le_durak.transform(df_egitim['Binis_Duragi'])

    X = df_egitim[['ID_Num', 'Saat_Num', 'Gun_Indeksi', 'Hat_Num', 'Binis_Num']]
    y = df_egitim['Hedef_Duragi']

    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
    model.fit(X, y)

    return model, le_id, le_hat, le_durak, df_kullanicilar, df_egitim


# Modeli yükle
with st.spinner("🧠 Veriler işleniyor ve Yapay Zeka eğitiliyor..."):
    model, le_id, le_hat, le_durak, df_kullanicilar, df_egitim = modeli_hazirla()

# ==========================================
# 2. ARAYÜZ (SOL PANEL)
# ==========================================
st.title("📊 MetroMeter: İstasyon Yoğunluk & Rota Analiz Merkezi")
st.markdown(
    "Bu panelde yolcuların seçilen hatta göre değişen rutinlerini görebilir and istasyonların anlık yolcu yoğunluğunu izleyebilirsiniz.")
st.markdown("---")

with st.sidebar:
    st.header("👤 1. Yolcu Seçimi")
    yolcu_listesi = df_kullanicilar.apply(lambda x: f"{x['Isim']} ({x['Kullanici_ID']})", axis=1).tolist()
    secilen_yolcu_str = st.selectbox("Bir yolcu seçin:", yolcu_listesi)

    secilen_uid = secilen_yolcu_str.split('(')[1].replace(')', '')
    yolcu_bilgi = df_kullanicilar[df_kullanicilar['Kullanici_ID'] == secilen_uid].iloc[0]

    st.markdown("---")
    st.header("⚙️ 2. Senaryo Ayarları")

    secilen_hat = st.radio("Biniş Hattı:", ["M2", "M7"], index=0 if yolcu_bilgi['Hat'] == "M2" else 1)

    # Dinamik Hatta Göre Detaylı Rutin Raporu
    yolcu_hat_gecmisi = df_egitim[
        (df_egitim['Kullanici_ID'] == secilen_uid) & (df_egitim['Binis_Hatti'] == secilen_hat)]

    if not yolcu_hat_gecmisi.empty:
        rutin_binis = yolcu_hat_gecmisi['Binis_Duragi'].mode()[0]
        rutin_hedef = yolcu_hat_gecmisi['Hedef_Duragi'].mode()[0]
        en_yogun_gun_indeksi = yolcu_hat_gecmisi['Gun_Indeksi'].mode()[0]
        en_yogun_gun_adi = GUN_ISIMLERI[en_yogun_gun_indeksi]

        yolcu_hat_gecmisi['Saat_Bloku'] = yolcu_hat_gecmisi['Saat_Num'].apply(lambda x: int(np.floor(x)))
        en_yogun_saat_bloku = yolcu_hat_gecmisi['Saat_Bloku'].mode()[0]
        saat_araligi_str = f"{en_yogun_saat_bloku:02d}:00 - {en_yogun_saat_bloku + 1:02d}:00"

        hat_saat_ort = round(yolcu_hat_gecmisi['Saat_Num'].mean(), 2)
        hat_saat_ort_str = format_saat(hat_saat_ort)

        box_status = "normal"
    else:
        rutin_binis = "Kayıt Yok"
        rutin_hedef = "Kayıt Yok"
        en_yogun_gun_adi = "Kayıt Yok"
        saat_araligi_str = "Kayıt Yok"
        hat_saat_ort_str = "08:00"
        hat_saat_ort = 8.0
        box_status = "empty"

    # --- DETAYLI MAVİ KUTU RAPORU ---
    if box_status == "normal":
        st.info(f"""
        **📊 {yolcu_bilgi['Isim']} - {secilen_hat} Hattı Güncel Rutini:**
        * **Rutin Giriş Durağı:** {rutin_binis}
        * **Rutin Hedef Durağı:** {rutin_hedef}
        * **En Sık Kullandığı Gün:** {en_yogun_gun_adi}
        * **En Sık Kullandığı Saat Aralığı:** {saat_araligi_str}
        """)
    else:
        st.warning(
            f"⚠️ {yolcu_bilgi['Isim']} geçmişte **{secilen_hat}** hattını hiç rutin olarak kullanmamış! Bu seçim tamamen rutin dışıdır.")

    secilen_gun = st.selectbox("Gün seçin:",
                               ["Pazartesi", "Sali", "Carsamba", "Persembe", "Cuma", "Cumartesi", "Pazar"])
    gun_indeksi = GUN_INPUT_MAP[secilen_gun]

    # --- AYRI SAAT VE DAKİKA SEÇİM KUTULARI ---
    st.markdown("**⏰ Turnike Geçiş Zamanı Ayarı:**")
    col_s1, col_s2 = st.columns(2)

    varsayilan_saat = int(hat_saat_ort)
    varsayilan_dakika = int(round((hat_saat_ort % 1) * 60))
    if varsayilan_dakika == 60:
        varsayilan_saat += 1
        varsayilan_dakika = 0

    with col_s1:
        saat_listesi = [f"{h:02d}" for h in range(6, 24)]
        v_saat_str = f"{varsayilan_saat:02d}"
        v_saat_index = saat_listesi.index(v_saat_str) if v_saat_str in saat_listesi else 2
        secilen_saat_saat = st.selectbox("Saat", saat_listesi, index=v_saat_index)

    with col_s2:
        dakika_listesi = [f"{m:02d}" for m in range(0, 60)]
        v_dak_str = f"{varsayilan_dakika:02d}"
        v_dak_index = dakika_listesi.index(v_dak_str) if v_dak_str in dakika_listesi else 0
        secilen_saat_dakika = st.selectbox("Dakika", dakika_listesi, index=v_dak_index)

    secilen_saat_metin = f"{secilen_saat_saat}:{secilen_saat_dakika}"
    secilen_saat = int(secilen_saat_saat) + (int(secilen_saat_dakika) / 60)

    durak_listesi = HAT_DURAKLARI[secilen_hat]
    varsayilan_durak_indeksi = durak_listesi.index(rutin_binis) if rutin_binis in durak_listesi else 0
    secilen_durak = st.selectbox("Biniş İstasyonu:", durak_listesi, index=varsayilan_durak_indeksi)

# ==========================================
# 3. İSTASYON ANLIK YOĞUNLUK HESAPLAMA MOTORU
# ==========================================
zaman_penceresi_alt = secilen_saat - 1.0
zaman_penceresi_ust = secilen_saat + 1.0

anlik_istasyon_kayitlari = df_egitim[
    (df_egitim['Gun_Indeksi'] == gun_indeksi) &
    (df_egitim['Binis_Duragi'] == secilen_durak) &
    (df_egitim['Saat_Num'] >= zaman_penceresi_alt) &
    (df_egitim['Saat_Num'] <= zaman_penceresi_ust)
    ]

binis_yolcu_sayisi = int(round(len(anlik_istasyon_kayitlari) / 15))
binis_yogunluk_str, binis_renk = yogunluk_hesapla(binis_yolcu_sayisi)

# ==========================================
# 4. ÜST METRİK KARTLARI (GİRİŞ DURUMU)
# ==========================================
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown(f"""
    <div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border-left: 5px solid {binis_renk};">
        <p style="color:gray; margin:0; font-size:13px;">📍 MEVCUT: {secilen_durak} ANLIK YOĞUNLUK</p>
        <h3 style="color:{binis_renk}; margin:5px 0;">{binis_yogunluk_str} (~{binis_yolcu_sayisi} kişi/saat)</h3>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    hedef_yogunluk_alani = st.empty()

st.caption(
    "ℹ️ **Ölçeklendirme Notu:** Bu simülasyondaki yolcu sayıları, sistemin eğitildiği **100 kişilik** prototip veritabanına göredir. "
    "Gerçek dünyada (yaklaşık 2.5 milyon İBB yolcusu için) bu değerleri **x25.000** katsayısıyla çarparak düşünebilirsiniz.")

st.markdown("---")

# ==========================================
# 5. ANA EKRAN (TAHMİN VE HEDEF PROJEKSİYONU)
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎫 Kart Basma Senaryosu")
    st.markdown(f"""
    Turnikeden anlık iletilen veriler:
    - **Yolcu:** {secilen_yolcu_str}
    - **Zaman:** {secilen_gun}, Saat {secilen_saat_metin}
    - **Konum:** {secilen_durak} ({secilen_hat} Hattı)
    """)

    tahmin_et = st.button("🔮 HEDEFİ VE GELECEK YOĞUNLUĞU TAHMİN ET", use_container_width=True)

if tahmin_et:
    try:
        anlik_id = le_id.transform([secilen_uid])[0]
        anlik_hat = le_hat.transform([secilen_hat])[0]
        anlik_binis = le_durak.transform([secilen_durak])[0]

        test_df = pd.DataFrame({
            'ID_Num': [anlik_id],
            'Saat_Num': [secilen_saat],
            'Gun_Indeksi': [gun_indeksi],
            'Hat_Num': [anlik_hat],
            'Binis_Num': [anlik_binis]
        })

        tahmin_hedef = model.predict(test_df)[0]
        guven_skoru = max(model.predict_proba(test_df)[0]) * 100

        varis_saati = secilen_saat + 0.5
        if varis_saati > 24: varis_saati -= 24
        varis_saat_guzel = format_saat(varis_saati)

        hedef_istasyon_kayitlari = df_egitim[
            (df_egitim['Gun_Indeksi'] == gun_indeksi) &
            (df_egitim['Hedef_Duragi'] == tahmin_hedef) &
            (df_egitim['Saat_Num'] >= varis_saati - 0.75) &
            (df_egitim['Saat_Num'] <= varis_saati + 0.75)
            ]
        hedef_yolcu_sayisi = int(round(len(hedef_istasyon_kayitlari) / 15))
        hedef_yogunluk_str,hedef_renk = yogunluk_hesapla(hedef_yolcu_sayisi)

        hedef_yogunluk_alani.markdown(f"""
        <div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border-left: 5px solid {hedef_renk};">
            <p style="color:gray; margin:0; font-size:13px;">🔮 GELECEK: {tahmin_hedef} TAHMİNİ VARMA YOĞUNLUĞU (Saat {varis_saat_guzel})</p>
            <h3 style="color:{hedef_renk}; margin:5px 0;">{hedef_yogunluk_str} (~{hedef_yolcu_sayisi} kişi/saat)</h3>
        </div>
        """, unsafe_allow_html=True)

        # --- DİNAMİK RENK VE DURUM BAŞLIĞI AYARI ---
        if box_status == "normal" and tahmin_hedef == rutin_hedef:
            analiz_kutusu_rengi = "#2ecc71"  # Canlı Yeşil
            durum_basligi = "🟢 RUTİN YOLCULUK"
        else:
            analiz_kutusu_rengi = "#f1c40f"  # Uyarı Sarısı (Anomali)
            durum_basligi = "⚠️ ANOMALİ / RUTİN DIŞI SEKANSLAR"

        with col2:
            st.markdown("### 🤖 Yapay Zeka Analizi")

            # Kutunun üst çizgisini ve metin renklerini dinamik değişkene bağladık!
            st.markdown(f"""
            <div style="background-color:#262730; padding:25px; border-radius:15px; border-top: 8px solid {analiz_kutusu_rengi};">
                <h6 style="color:{analiz_kutusu_rengi}; margin:0 0 5px 0; font-weight:bold;">{durum_basligi}</h6>
                <h5 style="color:gray; margin:0;">TAHMİN EDİLEN VARIŞ NOKTASI:</h5>
                <h1 style="color:{analiz_kutusu_rengi}; margin:10px 0;">🎯 {tahmin_hedef}</h1>
                <h5 style="color:gray; margin:0 0 15px 0;">YOLCUNUN TAHMİNİ VARMA SAATİ: 🕒 <b>{varis_saat_guzel}</b></h5>
                <h5 style="color:gray; margin:0;">TAHMİN GÜVEN ORANI:</h5>
                <h2 style="color:white; margin:5px 0;">% {guven_skoru:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            if box_status == "normal" and tahmin_hedef == rutin_hedef:
                st.success(
                    f"✅ **BAŞARILI TAHMİN:** Yapay zeka, bu kullanıcının **{secilen_hat}** hattındaki rutini olan **{rutin_hedef}** durağına gideceğini bildi!")
            elif box_status == "normal":
                st.warning(
                    f"ℹ️ **SIRA DIŞI ROTA:** Yolcu şu an bu hatta farklı bir yere gidiyor. (Bu hattaki normal rutini: {rutin_hedef})")
            else:
                st.info(
                    f"💡 **YENİ DENEYİM:** Yolcunun bu hatta geçmiş kaydı yoktu. Yapay zeka genel istatistiklere bakarak bir rota öngördü.")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
else:
    hedef_yogunluk_alani.markdown("""
    <div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border-left: 5px solid #3498db;">
        <p style="color:gray; margin:0; font-size:13px;">🔮 GELECEK VARIŞ YOĞUNLUĞU</p>
        <h3 style="color:#3498db; margin:5px 0;">Analiz Bekleniyor... ⏱️</h3>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.info("👈 Sol taraftan senaryoyu kurup 'Hedefi ve Gelecek Yoğunluğu Tahmin Et' butonuna bas kanka!")

st.markdown("---")
with st.expander("🛠️ MetroMüdavimi Proje Tanıtımı & Veri Metodolojisi (Genişletmek için tıklayın)"):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        ### 📊 Veri Seti Mimarisi
        Bu platformda kullanılan tüm veriler, makine öğrenmesi algoritmalarının örüntü tanıma (pattern recognition) yeteneklerini test etmek amacıyla **yapay zeka (sentetik veri motoru) tarafından üretilmiştir.**
        - **Kullanıcı Havuzu:** Sistemde kayıtlı, her birinin kendine has demografik seyahat alışkanlıkları ve çoklu hat müdavimlikleri olan **100 adet tekil yolcu profili** bulunmaktadır.
        - **Yolculuk Geçmişi:** Bu 100 yolcunun geçmiş 100 gün boyunca turnikelerden gerçekleştirdiği toplam **10.000 adet geçmiş akbil biniş kaydı** modellenmiştir.
        - **Zaman Esnekliği:** Yolcuların biniş saatleri %85 oranında kendi ana rutinlerine sadık kalınarak (Gauss Dağılımı ile) üretilmiş, %15 oranında ise sisteme 'gürültü' ve 'anomali' katması için tamamen rastgele seyahat senaryoları eklenmiştir.
        """)
    with col_t2:
        st.markdown("""
        ### 🧠 Algoritma & Tahmin Modellemesi
        Arka planda çalışan analitik motor, İstanbul akıllı şehir planlama ağının dijital bir ikizini simüle eder.
        - **Makine Öğrenmesi:** Sistem, Scikit-Learn kütüphanesi tabanlı bir **Random Forest Classifier (Rastgele Orman Sınıflandırıcısı)** algoritması kullanır. 
        - **Karar Mekanizması:** Arka planda tam **150 adet bağımsız karar ağacı** eğitilmiştir. Bu ağaçlar yolcu ID'si, haftanın günü, biniş istasyonu ve dakika bazlı hassas zaman verilerini analiz ederek saniyenin binde biri hızda varış noktası projeksiyonu üretir.
        - **Yoğunluk Projeksiyonu:** Turnike yoğunluk ve gelecek yük tahminleri, geçmiş 10.000 satırlık matrisin anlık sorgulanan zaman pencerelerine göre frekans analizinin (mod hesaplaması) yapılmasıyla dinamik olarak hesaplanmaktadır.
        """)