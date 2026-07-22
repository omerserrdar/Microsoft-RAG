# 🚀 Microsoft Foundry Local RAG Asistanı (Project Antigravity)

**Project Antigravity**, Microsoft Foundry Local SDK ve SQLite altyapısı kullanarak tamamen yerel (offline) çalışan, döküman tabanlı bir **RAG (Retrieval-Augmented Generation)** yapay zekâ sohbet uygulamasıdır.

İnternet bağlantısı gerektirmez, tüm verileriniz ve AI modelleriniz bilgisayarınızda yerel olarak çalışır.

---

## 🌟 Ana Özellikler

- 🔒 **%100 Gizlilik ve İnternetsiz Çalışma:** Tüm veri ve yapay zekâ modelleri yerelde çalışır, hiç bir veri dış sunuculara gönderilmez.
- 🧠 **Çok Turlu Sohbet Hafızası (Multi-Turn Memory):** Konuşma geçmişini hatırlayarak bağlamı korur.
- 🔀 **Gelişmiş Hibrit Arama (Hybrid Search):** Vektör araması (Cosine Similarity UDF) + Anahtar Kelime (LIKE) araması birleştirilerek en doğru döküman parçaları getirilir.
- 📄 **Otomatik Belge Özeti:** Yüklenen PDF/TXT/MD belgelerinin yapay zekâ destekli özetini çıkarır.
- 🏷️ **Belge Bazlı Filtreleme:** İstenen spesifik dökümanlar üzerinde arama yapma imkanı.
- 💬 **Sohbet Geçmişi Kaydı:** Sohbetler SQLite veritabanında saklanır, eski sohbetlere tek tıkla dönülebilir.
- 📊 **Vektör Benzerlik Analiz Grafikleri:** Yanıtların hangi döküman parçalarına dayandığını görselleştirir.
- 👍👎 **Kullanıcı Geri Bildirim Sistemi:** Yanıt kalitesini değerlendirme imkanı.
- 📋 **Sohbet Raporu İndirme:** Sohbet geçmişini Markdown formatında dışa aktarır.

---

## 🛠️ Mimari ve Teknolojiler

- **Kullanıcı Arayüzü (UI):** [Streamlit](https://streamlit.io/) (Özel CSS temalı)
- **Model Yönetimi ve Yerel LLM/Embedding:** [Microsoft Foundry Local SDK](https://github.com/microsoft/foundry)
- **Sohbet Modeli:** `phi-3.5-mini`
- **Embedding Modeli:** `qwen3-embedding-0.6b`
- **Veritabanı:** [aiosqlite](https://github.com/omnilib/aiosqlite) (C tabanlı Cosine Similarity UDF entegrasyonu ile)
- **Döküman İşleme:** `pypdf` & Akıllı Parçalama (Overlapping Chunking)

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/omerserrdar/Microsoft-RAG.git
cd Microsoft-RAG
```

### 2. Gerekli Paketleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın
```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

---

## 📁 Proje Yapısı

```
microsoft-rag/
├── app.py                  # Streamlit ana arayüz uygulaması
├── requirements.txt        # Bağımlılıklar
├── core/
│   ├── __init__.py
│   ├── database.py         # Asenkron SQLite veritabanı & Cosine Similarity UDF
│   ├── ingester.py         # Döküman okuma, parçalama ve vektörleştirme
│   ├── retriever.py        # Semantik & Hibrit arama motoru
│   └── generator.py        # Yerel LLM streaming yanıt ve özet üretimi
└── README.md
```

---

## 📜 Lisans
MIT License
