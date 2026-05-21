# 🔮 MetroMeter: Akıllı İstasyon Yoğunluk & Yolcu Rota Analiz Merkezi

**MetroMeter**, makine öğrenmesi algoritmalarını şehir içi mobilite ve toplu taşıma optimizasyonuyla birleştiren proaktif bir akıllı şehir (Smart City) simülasyonudur. Proje, sadece geçmiş verileri raporlamakla kalmaz; bireysel yolcu davranışlarını anlık tahmin ederken, istasyonların gelecek yönlü yoğunluk projeksiyonlarını (**Forward-Looking Load Forecasting**) çıkarır.

---

## 🚀 Öne Çıkan Özellikler

* **Dinamik Çoklu Hat Rutin Analizi:** Yolcu seçildiği an, geçmiş biniş kayıtlarından o yolculuğun hat bazlı (M2/M7) spesifik en yoğun biniş saati, sık kullanılan istasyon ve müdavimlik günleri anlık olarak analiz edilir.
* **Dakika Hassasiyetli Zaman Kontrolü:** Klasik kaba saat blokları yerine, yolcuların turnikeye geliş anlarını dakika hassasiyetinde (`18:03`, `08:42` gibi) yakalayan esnek bir ondalık dönüştürücü modül içerir.
* **Gelecek Yönlü Yoğunluk Motoru:** Turnikeden kart basıldığı an, yolculuk süresi hesaplanarak **varış istasyonuna ulaştığı andaki gelecek yoğunluk durumu** simüle edilir.
* **Canlı Anomali Tespit Sistemi (Real-time Anomaly Detection):** Yapay zeka analizi, yolcu kendi rutini içinde hareket ediyorsa yeşil panel (**Rutin Yolculuk**); normal davranış kalıplarının dışına çıkan farklı bir senaryoda ise sarı panel (**Anomali**) uyarısı vererek sistemi dinamik olarak manipüle eder.

---

## 🧠 Teknik Altyapı & Metodoloji

* **Veri Kümesi:** Sistem, sentetik veri motoru tarafından üretilen **100 tekil yolcu profili** ve bu yolcuların geçmiş 100 gün boyunca gerçekleştirdiği **10.000 turnike geçiş sekansını** içeren büyük bir veri matrisi ile eğitilmiştir.
* **Algoritma:** Scikit-Learn kütüphanesi tabanlı **Random Forest Classifier (Rastgele Orman Sınıflandırıcısı)** algoritması kullanılmıştır. Modelin arkasında tam **150 adet bağımsız karar ağacı (Decision Tree)** yapılandırılmıştır.
* **Arayüz:** Kullanıcı deneyimini interaktif hale getirmek amacıyla **Streamlit** kütüphanesi tercih edilmiştir ve başlık çapa linkleri CSS enjeksiyonu ile optimize edilmiştir.

---
