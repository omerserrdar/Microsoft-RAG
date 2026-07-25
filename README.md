# 🚀 Derkenar — Microsoft Foundry Local RAG Asistanı

**Tamamen yerel çalışan, gizlilik odaklı RAG (Retrieval-Augmented Generation) sistemi.**

Bulut API'lerine bağımlı olmadan, kendi belgeleriniz üzerinde soru-cevap yapabilen bir sistem. Tüm embedding ve dil modeli çıkarımı [Microsoft Foundry Local SDK](https://github.com/microsoft/Foundry-Local) üzerinden **cihazınızda** çalışır — verileriniz hiçbir zaman dışarı çıkmaz.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Teknoloji Yığını](#teknoloji-yığını)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Modül Referansı](#modül-referansı)
- [Yapılandırma](#yapılandırma)
- [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
- [Yol Haritası](#yol-haritası)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

---

## Özellikler

- 🔒 **%100 Yerel Çalışma** — Embedding ve LLM çıkarımı Foundry Local üzerinden cihazda yapılır, internet veya API key gerekmez.
- 📄 **Çoklu Format Desteği** — PDF, TXT ve Markdown dosyalarından döküman yükleme.
- ✂️ **Akıllı Metin Parçalama** — Cümle sınırlarına duyarlı, örtüşmeli (overlapping) chunking algoritması.
- 🔍 **Hibrit Arama** — Semantik (cosine similarity) ve anahtar kelime (keyword) aramasını ağırlıklı olarak birleştiren retrieval motoru.
- ⚡ **SQL İçinde Vektör Arama** — Kosinüs benzerliği, Python döngüsü kullanmadan SQLite'a kayıtlı özel bir UDF (User-Defined Function) ile hesaplanır.
- 🌊 **Streaming Yanıtlar** — Async generator mimarisiyle token-token akan LLM yanıtları.
- 🧭 **Halüsinasyon Koruması** — Sistem promptu, yalnızca sağlanan bağlama dayanarak yanıt üretilmesini zorunlu kılar; bilgi yoksa bunu açıkça belirtir.
- 📚 **Otomatik Kaynak Gösterimi** — Her yanıtın sonunda hangi dosyalardan alındığı belirtilir.
- 💬 **Sohbet Geçmişi ve Geri Bildirim** — Oturum bazlı sohbet kaydı ve thumbs up/down geri bildirim desteği.
- 📝 **Döküman Özetleme** — Yüklenen belgeler için otomatik kısa özet üretimi.

---

## Mimari

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌───────────┐
│  PDF / TXT  │ ─▶ │  Akıllı Parçalama │ ─▶ │ Yerel Embedding│ ─▶ │  SQLite   │
│   Dosya     │    │   (Overlapping)   │    │  (Foundry SDK) │    │  Kaydet   │
└─────────────┘    └──────────────────┘    └────────────────┘    └───────────┘
                          YÜKLEME (yukleyici.py)

┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌──────────────┐
│ Kullanıcı    │ ─▶ │ Soru Vektörü   │ ─▶ │ SQL Cosine   │ ─▶ │ En Alakalı   │
│ Sorusu       │    │  (Embedding)   │    │  (Foundry SDK) │    │  Parçalar    │
└──────────────┘    └────────────────┘    └──────────────┘    └──────────────┘
                          ARAMA (arama_motoru.py)

┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌──────────────┐
│ Soru +       │ ─▶ │ System Prompt  │ ─▶ │ phi-3.5-mini │ ─▶ │ Streaming    │
│ Bağlam       │    │  (Dürüstlük)   │    │  Yerel LLM   │    │ Yanıt + Atıf │
└──────────────┘    └────────────────┘    └──────────────┘    └──────────────┘
                          ÜRETİM (ureteci.py)
```

Tüm katmanlar `async`/`await` üzerine kuruludur; ağır ve bloklayıcı işlemler (PDF okuma, model yükleme) `asyncio.to_thread()` ile arka plan thread'lerine delege edilir.

---

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Dil modeli / Embedding çıkarımı | [Foundry Local SDK](https://github.com/microsoft/Foundry-Local) (`foundry_local_sdk`) |
| Embedding modeli | `qwen3-embedding-0.6b` |
| Sohbet modeli | `phi-3.5-mini` (değiştirilebilir: `phi-4-mini`, `qwen2.5-7b`, `qwen3-4b`) |
| Model API katmanı | OpenAI-uyumlu istemci (`openai` Python paketi, `AsyncOpenAI`) |
| Veritabanı | SQLite (`aiosqlite`) + özel Cosine Similarity UDF (`numpy`) |
| PDF okuma | `pypdf` |
| Runtime | Python 3.10+ (async/await, `list[dict]` tip ipuçları) |

---

## Kurulum

### Gereksinimler

- Python 3.10 veya üzeri
- Foundry Local SDK'nın desteklediği bir işletim sistemi (Windows / macOS)
- Yeterli disk alanı (embedding + sohbet modeli için, modeller ilk çalıştırmada indirilir)

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/omerserrdar/Microsoft-RAG.git
cd Microsoft-RAG

# 2. Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install aiosqlite numpy pypdf openai foundry-local-sdk
```

> **Not:** `requirements.txt` üzerinden de yükleyebilirsiniz: `pip install -r requirements.txt`

İlk çalıştırmada Foundry Local SDK, embedding ve sohbet modellerini otomatik olarak indirip yerel bir web servisi başlatır. Bu işlem internet bağlantısı gerektirir (yalnızca ilk indirmede); sonraki kullanımlar tamamen yerel/offline çalışır.

---

## Kullanım

### Belge Yükleme

```python
import asyncio
from core.veritabani import DokumanVeritabani
from core.yukleyici import foundry_vektor_istemcisi_olustur, dosya_yukle

async def main():
    async with DokumanVeritabani() as db:
        client, model = await foundry_vektor_istemcisi_olustur()
        result = await dosya_yukle("rapor.pdf", db, client, model)
        print(result)
        # {"file_name": "rapor.pdf", "chunks": 12, "status": "success"}

asyncio.run(main())
```

Bir klasördeki tüm dosyaları toplu yüklemek için:

```python
from core.yukleyici import klasor_yukle

sonuclar = await klasor_yukle("belgeler/", db, client, model)
```

### Soru Sorma (Arama + Yanıt Üretimi)

```python
from core.arama_motoru import ilgili_parcalari_getir
from core.ureteci import foundry_sohbet_istemcisi_olustur, akisli_yanit_uret

async def soru_sor(soru: str):
    async with DokumanVeritabani() as db:
        embed_client, embed_model = await foundry_vektor_istemcisi_olustur()
        chat_client, chat_model = await foundry_sohbet_istemcisi_olustur()

        bulunanlar = await ilgili_parcalari_getir(soru, db, embed_client, embed_model)

        async for token in akisli_yanit_uret(soru, bulunanlar, chat_client, chat_model):
            print(token, end="", flush=True)

asyncio.run(soru_sor("Şirketin 2024 hedefleri nelerdir?"))
```

### Hibrit Arama (Semantik + Anahtar Kelime)

```python
from core.arama_motoru import hibrit_getir

sonuclar = await hibrit_getir(
    "bütçe planlaması",
    db, embed_client, embed_model,
    semantic_weight=0.7,  # 0.0-1.0 arası, semantik/keyword ağırlığı
)
```

---

## Proje Yapısı

```
core/
├── __init__.py         # Paket tanımı ve modül özeti
├── veritabani.py       # Async SQLite veritabanı + Cosine Similarity UDF
├── yukleyici.py        # Döküman okuma, chunking, embedding, ingestion pipeline
├── arama_motoru.py     # Semantik + hibrit arama (retrieval katmanı)
└── ureteci.py          # LLM streaming yanıt üretimi + döküman özetleme
```

---

## Modül Referansı

### `veritabani.py` — `DokumanVeritabani`

| Metod | Açıklama |
|---|---|
| `initialize()` / `baslat()` | Veritabanını açar, tabloları ve UDF'i kurar |
| `insert_chunk()` / `parca_ekle()` | Tek bir döküman parçası ekler |
| `insert_chunks_bulk()` / `parcalari_toplu_ekle()` | Toplu (bulk) ekleme |
| `search_similar()` / `benzer_ara()` | SQL içinde cosine similarity ile arama |
| `keyword_search()` / `anahtar_kelime_ara()` | LIKE tabanlı anahtar kelime araması |
| `get_document_stats()` / `dokuman_istatistiklerini_getir()` | Dosya bazlı chunk istatistikleri |
| `save_message()`, `get_chat_sessions()`, `get_session_messages()` | Sohbet geçmişi yönetimi |
| `save_feedback()`, `get_feedback_stats()` | Kullanıcı geri bildirimi (thumbs up/down) |

### `yukleyici.py`

| Fonksiyon | Açıklama |
|---|---|
| `read_file()` / `dosya_oku()` | PDF/TXT/MD dosyasını okur |
| `chunk_text()` / `metni_parcala()` | Cümle bazlı, örtüşmeli chunking |
| `create_foundry_embedding_client()` / `foundry_vektor_istemcisi_olustur()` | Embedding modeli için istemci başlatır |
| `compute_embeddings()` / `embedding_hesapla()` | Metinleri vektörlere çevirir |
| `ingest_file()` / `dosya_yukle()` | Tam pipeline: oku → parçala → embed → kaydet |
| `ingest_directory()` / `klasor_yukle()` | Bir klasördeki tüm dosyaları toplu yükler |

### `arama_motoru.py`

| Fonksiyon | Açıklama |
|---|---|
| `retrieve_relevant_chunks()` / `ilgili_parcalari_getir()` | Soruyu vektöre çevirip en benzer chunk'ları getirir |
| `hybrid_retrieve()` / `hibrit_getir()` | Semantik + anahtar kelime skorlarını ağırlıklı birleştirir |

### `ureteci.py`

| Fonksiyon | Açıklama |
|---|---|
| `create_foundry_chat_client()` / `foundry_sohbet_istemcisi_olustur()` | Sohbet modeli için istemci başlatır |
| `generate_streaming_response()` / `akisli_yanit_uret()` | Bağlam + soru ile streaming yanıt üretir |
| `generate_document_summary()` / `belge_ozeti_uret()` | Bir belgenin kısa özetini üretir |

---

## Yapılandırma

Şu an yapılandırma kod içinde varsayılan parametreler olarak tanımlı. Önemli varsayılanlar:

| Parametre | Varsayılan | Konum |
|---|---|---|
| Embedding modeli | `qwen3-embedding-0.6b` | `yukleyici.py` |
| Sohbet modeli | `phi-3.5-mini` | `ureteci.py` |
| Chunk boyutu | `500` karakter | `yukleyici.py` |
| Chunk örtüşmesi | `50` karakter | `yukleyici.py` |
| Arama sonuç sayısı (`top_k`) | `5` | `arama_motoru.py` |
| LLM `temperature` | `0.2` | `ureteci.py` |
| LLM `max_tokens` | `512` | `ureteci.py` |
| Veritabanı yolu | `<proje_kök>/data/derkenar.db` | `veritabani.py` |

> Bu değerlerin bir `.env` dosyası veya `config.py` üzerinden merkezi olarak yönetilmesi, farklı ortamlarda (geliştirme/prod) esneklik sağlar.

---

## Bilinen Sınırlamalar

- **Ölçeklenebilirlik:** Cosine similarity her sorguda tüm `documents` tablosunu tarar (full table scan). Binlerce chunk'ı aşan veri setlerinde bir ANN indeksine (örn. `sqlite-vec`, FAISS) geçiş gerekebilir.
- **Format desteği:** Şu an yalnızca PDF, TXT ve Markdown destekleniyor; `.docx`, `.pptx` veya OCR gerektiren taranmış PDF'ler desteklenmiyor.
- **Dosya güncelleme tespiti:** Aynı isimli bir dosya içerik olarak değişse bile sistem onu "zaten yüklü" sayıp atlıyor (içerik hash kontrolü yok).
- **Bağlam bütçesi:** Yanıt üretiminde bağlam metninin toplam uzunluğu sınırlanmıyor; çok büyük `top_k` veya uzun chunk'larla modelin context penceresi aşılabilir.
- **Tek bağlantı modeli:** `aiosqlite` tek bir bağlantı üzerinden çalışıyor; yüksek eşzamanlılık gerektiren senaryolarda bağlantı havuzu (connection pool) değerlendirilmeli.

---

## Yol Haritası

- [ ] İçerik hash'i ile değişiklik tespiti ve otomatik yeniden indeksleme
- [ ] Chunk seviyesinde sayfa numarası / sıra bilgisi (daha isabetli kaynak gösterimi)
- [ ] Minimum benzerlik skoru filtresi (alakasız sonuçları elemek için)
- [ ] Cross-encoder reranking adımı
- [ ] Token bazlı bağlam bütçesi yönetimi
- [ ] `.docx` ve OCR destekli PDF okuma
- [ ] `requirements.txt` / `pyproject.toml` ve temel birim testleri
- [ ] Web arayüzü (FastAPI / Gradio / Streamlit)

---

## Katkıda Bulunma

Katkılar memnuniyetle karşılanır. Lütfen değişiklik yapmadan önce bir issue açarak neyi çözmeyi planladığınızı belirtin. Kod stili olarak mevcut Türkçe docstring + İngilizce fonksiyon adı + Türkçe alias yapısının korunması tercih edilir.

## Lisans

MIT
