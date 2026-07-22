"""
core/ingester.py — Project Antigravity: Döküman İşleme ve Vektörleştirme
=========================================================================
Bu modül, RAG (Retrieval-Augmented Generation) sisteminin veri yükleme
işlem hattını (Data Ingestion Pipeline) yönetir.

İşlem Hattı Akışı:
    ┌────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌───────────┐
    │ PDF / TXT  │ ──▸ │ Akıllı Parçalama │ ──▸ │ Yerel Embedding│ ──▸ │  SQLite   │
    │ Dosya Oku  │     │ (Overlapping)    │     │ (Foundry SDK)  │     │  Kaydet   │
    └────────────┘     └──────────────────┘     └────────────────┘     └───────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD MODE ÖZELLİĞİ — Akıllı Metin Parçalama (Semantic Overlapping):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Basit karakter bölmesi yerine, anlam bütünlüğünü koruyan, cümle
sınırlarına duyarlı ve birbirine örtüşen (overlapping) parçalar üretir.

Örtüşme (Overlap) Mantığı:
    Parça 1: [████████████████████░░░░░]  ← son 50 karakter →
    Parça 2:                   [░░░░░████████████████████]
    
    Bu sayede bir cümle iki parçanın sınırına denk gelse bile
    anlam bütünlüğü korunur.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kullanım:
    from core.database import DocumentDB
    from core.ingester import ingest_file, create_foundry_embedding_client

    async with DocumentDB() as db:
        client, model = await create_foundry_embedding_client()
        result = await ingest_file("rapor.pdf", db, client, model)
        print(result)  # {"file_name": "rapor.pdf", "chunks": 12, "status": "success"}
"""

# ============================================================
# İMPORT'LAR
# ============================================================

import asyncio            # Asenkron yapı ve thread delegasyonu
import logging            # Loglama altyapısı
import re                 # Düzenli ifadeler (cümle bölme)
from pathlib import Path  # Platformdan bağımsız dosya yolu yönetimi
from typing import Optional  # Tip ipuçları

# ── Proje İçi İmport ──
from core.database import DocumentDB


# ============================================================
# LOGLAMA AYARLARI
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# BÖLÜM 1: DOSYA OKUMA — PDF ve TXT DESTEĞİ
# ============================================================
# PDF okuma işlemi blokajlı (blocking) bir I/O işlemidir.
# asyncio.to_thread() ile arka plan thread'ine delege edilerek
# ana olay döngüsünün (event loop) bloklanması önlenir.
# ============================================================

async def read_file(file_path: Path) -> tuple[str, int]:
    """
    Dosya uzantısına göre uygun okuyucuyu seçer ve metin içeriğiyle sayfa sayısını döndürür.

    Desteklenen Formatlar:
        .pdf  → pypdf kütüphanesiyle sayfa sayfa metin çıkarımı
        .txt  → Düz metin olarak okuma (UTF-8)
        .md   → Markdown dosyası olarak okuma (UTF-8)

    Parametreler:
        file_path (Path): Okunacak dosyanın yolu

    Döndürür:
        tuple[str, int]: (Dosyanın tüm metin içeriği, sayfa sayısı)

    Hatalar:
        FileNotFoundError : Dosya bulunamazsa
        ValueError        : Desteklenmeyen format ise
    """
    # Dosyanın var olup olmadığını kontrol et
    if not file_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    # Uzantıya göre okuyucu seç
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return await _read_pdf(file_path)
    elif suffix in (".txt", ".md", ".text"):
        return await _read_text_file(file_path)
    else:
        raise ValueError(
            f"Desteklenmeyen dosya formatı: '{suffix}'. "
            f"Desteklenen formatlar: .pdf, .txt, .md"
        )


async def _read_pdf(file_path: Path) -> tuple[str, int]:
    """
    PDF dosyasını pypdf kütüphanesiyle okur.

    pypdf blokajlı (blocking) bir kütüphanedir, bu yüzden
    asyncio.to_thread() ile arka plan thread'inde çalıştırılır.
    Bu sayede ana event loop bloklanmaz.

    Parametreler:
        file_path (Path): PDF dosyasının yolu

    Döndürür:
        tuple[str, int]: (Tüm sayfaların birleştirilmiş metin içeriği, sayfa sayısı)
    """
    def _extract_sync():
        """Thread içinde çalışacak senkron PDF okuma fonksiyonu."""
        # pypdf'i burada import ediyoruz — sadece ihtiyaç olduğunda yüklenir
        from pypdf import PdfReader

        # PDF dosyasını aç
        reader = PdfReader(str(file_path))

        # Her sayfanın metnini bir listeye topla
        extracted_pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                extracted_pages.append(page_text)
                logger.debug(f"  📄 Sayfa {page_number}: {len(page_text)} karakter okundu")

        # Sayfaları çift satır sonu ile birleştir
        return "\n\n".join(extracted_pages), len(reader.pages)

    # Senkron fonksiyonu arka plan thread'inde çalıştır
    logger.info(f"📖 PDF okunuyor: {file_path.name}")
    text, page_count = await asyncio.to_thread(_extract_sync)
    logger.info(f"📖 PDF okuma tamamlandı: {len(text)} karakter, {page_count} sayfa")
    return text, page_count


async def _read_text_file(file_path: Path) -> tuple[str, int]:
    """
    Düz metin dosyasını UTF-8 kodlamasıyla okur.

    Dosya I/O işlemi asyncio.to_thread() ile thread'e delege edilir.

    Parametreler:
        file_path (Path): Metin dosyasının yolu

    Döndürür:
        tuple[str, int]: (Dosyanın metin içeriği, 1)
    """
    def _read_sync():
        """Thread içinde çalışacak senkron okuma fonksiyonu."""
        return file_path.read_text(encoding="utf-8")

    logger.info(f"📝 Metin dosyası okunuyor: {file_path.name}")
    text = await asyncio.to_thread(_read_sync)
    logger.info(f"📝 Metin dosyası okuma tamamlandı: {len(text)} karakter")
    return text, 1


# ============================================================
# BÖLÜM 2: AKILLI METİN PARÇALAMA (SEMANTIC OVERLAPPING CHUNKING)
# ============================================================
# Bu bölüm, RAG sisteminin kalitesini doğrudan etkileyen kritik
# bir bileşendir. Kötü parçalama = kötü arama sonuçları.
#
# Strateji:
#   1. Metni cümle sınırlarından böl (anlam bütünlüğünü koru)
#   2. Cümleleri birleştirerek chunk_size hedefine ulaş
#   3. Son N karakter overlap olarak bir sonraki parçaya taşınır
#   4. Tek başına çok uzun cümleler kelime sınırlarından bölünür
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Metni anlam bütünlüğünü koruyarak, üst üste binen (overlapping)
    parçalara ayırır.

    Parçalama Stratejisi (3 Aşamalı):
    ──────────────────────────────────
    Aşama 1 — Cümle Bölme:
        Metni noktalama işaretlerinden (. ! ? ve satır sonlarından) cümlelere ayır.

    Aşama 2 — Cümle Birleştirme:
        Cümleleri sırayla birleştir, toplam uzunluk chunk_size'ı aşana kadar devam et.
        Aşınca mevcut bloğu kaydet ve yeni bloğa geç.

    Aşama 3 — Örtüşme (Overlap):
        Kaydedilen bloğun son chunk_overlap kadar karakterini al,
        bir sonraki bloğun başına ekle. Bu, sınır bölgelerindeki
        anlam kaybını önler.

    Görsel Örnek (chunk_size=20, overlap=5):
        Metin: "Yapay zekâ gelişiyor. Derin öğrenme güçlü."
        Parça 1: "Yapay zekâ gelişiyor."
        Parça 2: "iyor. Derin öğrenme güçlü."
                  ↑↑↑↑↑ overlap bölgesi

    Parametreler:
        text          : Parçalanacak kaynak metin
        chunk_size    : Her parçanın hedef maksimum uzunluğu (karakter)
        chunk_overlap : Parçalar arası örtüşme miktarı (karakter)

    Döndürür:
        list[str]: Parçalanmış metin blokları listesi
    """
    # ── Boş metin kontrolü ──
    if not text or not text.strip():
        return []

    # ── Metni normalleştir ──
    # Ardışık boşlukları (tab, çoklu space, vb.) tek boşluğa indir
    text = re.sub(r"\s+", " ", text).strip()

    # ── Metin zaten chunk_size'dan kısaysa tek parça döndür ──
    if len(text) <= chunk_size:
        return [text]

    # ── Aşama 1: Cümle bazlı bölme ──
    # Noktalama (. ! ?) sonrasındaki boşlukları veya satır sonlarını ayraç olarak kullan
    # (?<=[.!?]) : "lookbehind" — noktalama işaretinin SONRASINDA böl (işareti koru)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Eğer cümle bölme başarısızsa (hiç ayraç bulunamadıysa), orijinal metni kullan
    if not sentences:
        sentences = [text]

    # ── Aşama 2 & 3: Cümle birleştirme + örtüşme (overlap) ──
    chunks: list[str] = []
    current_chunk = ""  # Şu an oluşturulan parça

    for sentence in sentences:

        # --- Özel Durum: Tek bir cümle chunk_size'dan uzunsa ---
        # Bu cümleyi kelime sınırlarından zorunlu olarak böl
        if len(sentence) > chunk_size:
            # Önce mevcut chunk'ı kaydet (boş değilse)
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Uzun cümleyi kelime kelime böl
            forced_chunks = _force_split_long_text(
                sentence, chunk_size, chunk_overlap
            )
            # Son parça hariç hepsini chunks'a ekle
            # Son parça bir sonraki iterasyona taşınır (overlap için)
            if forced_chunks:
                chunks.extend(forced_chunks[:-1])
                current_chunk = forced_chunks[-1]
            continue

        # --- Normal Durum: Cümleyi mevcut chunk'a eklemeyi dene ---
        candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence

        if len(candidate) <= chunk_size:
            # Sığıyor → mevcut chunk'a ekle
            current_chunk = candidate
        else:
            # Sığmıyor → mevcut chunk'ı kaydet ve yeni chunk başlat
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # ── Overlap hesapla ──
            # Kaydedilen chunk'ın son N karakterini al
            current_chunk = _apply_overlap(current_chunk, sentence, chunk_overlap)

    # ── Son kalan parçayı kaydet ──
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    logger.debug(
        f"✂️ Metin {len(chunks)} parçaya ayrıldı "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks


def _apply_overlap(
    previous_chunk: str,
    next_sentence: str,
    overlap_size: int,
) -> str:
    """
    Önceki parçanın son N karakterini alarak yeni parçanın
    başına ekler (overlap). Kelime sınırına uyar.

    Parametreler:
        previous_chunk : Kaydedilen önceki parça
        next_sentence  : Yeni parçanın ilk cümlesi
        overlap_size   : Örtüşme miktarı (karakter)

    Döndürür:
        str: Overlap + yeni cümle birleşimi
    """
    if overlap_size <= 0 or len(previous_chunk) < overlap_size:
        # Overlap uygulanmayacak veya önceki chunk çok kısa
        return next_sentence

    # Son N karakteri al
    overlap_text = previous_chunk[-overlap_size:]

    # Kelime sınırına hizala — overlap'in başını ilk boşluktan başlat
    # Bu sayede kelime ortasından kesmemiş oluruz
    space_index = overlap_text.find(" ")
    if space_index != -1:
        overlap_text = overlap_text[space_index + 1:]

    # Overlap + yeni cümle birleştir
    return (overlap_text + " " + next_sentence).strip()


def _force_split_long_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    chunk_size'dan uzun bir metni kelime sınırlarından zorla böler.

    Bu fonksiyon yalnızca tek bir cümle chunk_size'ı aştığında çağrılır.
    Cümle bazlı bölme mümkün olmadığında, kelimeleri birleştirerek
    chunk_size'a uyan parçalar üretir.

    Parametreler:
        text          : Bölünecek uzun metin
        chunk_size    : Hedef parça boyutu
        chunk_overlap : Parçalar arası örtüşme

    Döndürür:
        list[str]: Bölünmüş parçalar listesi
    """
    words = text.split()
    chunks: list[str] = []
    current = ""

    for word in words:
        # Kelimeyi eklemeyi dene
        candidate = (current + " " + word).strip() if current else word

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Mevcut chunk'ı kaydet
            if current.strip():
                chunks.append(current.strip())
            # Overlap uygula
            current = _apply_overlap(current, word, chunk_overlap)

    # Son kalan parça
    if current.strip():
        chunks.append(current.strip())

    return chunks


# ============================================================
# BÖLÜM 3: YEREL EMBEDDİNG İSLEMCİSİ (FOUNDRY SDK)
# ============================================================
# Foundry Local SDK, OpenAI-uyumlu bir endpoint sunar.
# Bu sayede standart openai kütüphanesiyle (AsyncOpenAI)
# yerel embedding modeline istek gönderebiliyoruz.
# ============================================================

async def create_foundry_embedding_client(
    model_name: str = "qwen3-embedding-0.6b",
) -> tuple:
    """
    Microsoft Foundry Local SDK'yı başlatır, embedding modelini indirir/yükler
    ve OpenAI-uyumlu asenkron bir istemci döndürür.

    Bu fonksiyon tüm ağır işlemleri (SDK başlatma, model indirme, yükleme)
    asyncio.to_thread() ile arka plan thread'ine delege eder.

    Parametreler:
        model_name: Kullanılacak embedding modelinin adı
                    Varsayılan: "qwen3-embedding-0.6b"

    Döndürür:
        tuple: (AsyncOpenAI_client, model_adı)
            - client: Embedding istekleri göndermek için hazır istemci
            - model_adı: Modelin katalog adı (API çağrılarında kullanılır)

    Örnek:
        client, model = await create_foundry_embedding_client()
        response = await client.embeddings.create(model=model, input=["merhaba"])
    """
    def _initialize_sdk():
        """
        Thread içinde çalışacak senkron SDK başlatma fonksiyonu.

        Foundry SDK'nın tüm yönetim işlemleri (initialize, download, load)
        senkron API sunar, bu yüzden thread'e delege ediyoruz.
        """
        from foundry_local_sdk import Configuration, FoundryLocalManager

        # ── SDK yapılandırmasını oluştur ──
        config = Configuration(app_name="project_antigravity")

        # ── SDK yöneticisini başlat (singleton) ──
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance

        # ── Embedding modelini katalogdan al ──
        model = manager.catalog.get_model(model_name)
        logger.info(f"📦 Embedding modeli bulundu: {model_name}")

        # ── Modeli indir (zaten indirildiyse bu adım atlanır) ──
        logger.info(f"⬇️ Model indiriliyor (veya önbellekten yükleniyor)...")
        model.download(
            lambda progress: logger.debug(f"  ⬇️ İndirme: {progress:.1f}%")
        )

        # ── Modeli belleğe yükle ──
        logger.info(f"🧠 Model belleğe yükleniyor...")
        model.load()
        logger.info(f"✅ Embedding modeli hazır: {model_name}")

        # ── Yerel web servisini başlat (OpenAI-compat endpoint) ──
        manager.start_web_service()
        base_url = f"{manager.urls[0]}/v1"
        logger.info(f"🌐 Yerel endpoint aktif: {base_url}")

        return base_url

    # Senkron SDK başlatmayı arka plan thread'inde çalıştır
    base_url = await asyncio.to_thread(_initialize_sdk)

    # ── OpenAI-uyumlu asenkron istemciyi oluştur ──
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=base_url,  # Yerel Foundry endpoint'ine yönlendir
        api_key="local",    # Yerel çalıştığımız için API key gerekli değil
    )

    return client, model_name


async def compute_embeddings(
    texts: list[str],
    client,
    model_name: str = "qwen3-embedding-0.6b",
) -> list[list[float]]:
    """
    Metin listesini yerel embedding modeline gönderir ve
    her metin için sayısal vektör karşılığını alır.

    Bu fonksiyon OpenAI-uyumlu Foundry endpoint'ini kullanır,
    dolayısıyla standart openai kütüphanesinin embeddings API'si ile çalışır.

    Parametreler:
        texts      : Vektöre dönüştürülecek metin listesi
        client     : AsyncOpenAI istemcisi (create_foundry_embedding_client'tan)
        model_name : Embedding modelinin adı

    Döndürür:
        list[list[float]]: Her metin için bir embedding vektörü listesi.
                           Sıralama, girdi listesiyle birebir eşleşir.

    Örnek:
        vektorler = await compute_embeddings(
            ["Yapay zekâ", "Makine öğrenmesi"],
            client, "qwen3-embedding-0.6b"
        )
        print(len(vektorler))      # 2
        print(len(vektorler[0]))   # Model boyutu (ör: 1024)
    """
    # ── OpenAI Embeddings API çağrısı ──
    # client yerel Foundry endpoint'ine bağlı, ama API formatı aynı
    response = await client.embeddings.create(
        model=model_name,
        input=texts,
    )

    # ── Her bir metin için embedding vektörünü çıkar ──
    # response.data, girdi sırasıyla eşleşen EmbeddingObject listesi döndürür
    embeddings = [item.embedding for item in response.data]

    logger.debug(
        f"🔢 {len(texts)} metin → {len(embeddings)} vektör "
        f"(boyut: {len(embeddings[0]) if embeddings else '?'})"
    )

    return embeddings


# ============================================================
# BÖLÜM 4: ANA VERİ YÜKLEME İŞLEM HATTI (INGESTION PIPELINE)
# ============================================================
# Bu bölüm, yukarıdaki tüm bileşenleri bir araya getirir:
#   Dosya Oku → Parçala → Embedding Al → Veritabanına Yaz
#
# Tüm aşamalar asenkron çalışır ve toplu (batch) işlem yapar.
# ============================================================

async def ingest_file(
    file_path: str | Path,
    db: DocumentDB,
    embedding_client,
    embedding_model: str = "qwen3-embedding-0.6b",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 16,
) -> dict:
    """
    Ana veri yükleme fonksiyonu — dosyadan veritabanına tam pipeline.

    İşlem Adımları:
        1. Dosyayı oku (PDF veya TXT)
        2. Metni akıllı parçalara ayır (overlapping chunking)
        3. Her parçanın embedding vektörünü hesapla (batch halinde)
        4. Veritabanına toplu olarak kaydet (bulk insert)

    Parametreler:
        file_path        : Yüklenecek dosyanın yolu (str veya Path)
        db               : DocumentDB instance'ı (başlatılmış olmalı)
        embedding_client : AsyncOpenAI istemcisi (Foundry endpoint'i)
        embedding_model  : Embedding modelinin adı
        chunk_size       : Metin parçası hedef boyutu (karakter)
        chunk_overlap    : Parçalar arası örtüşme (karakter)
        batch_size       : Embedding hesabı için toplu istek boyutu
                           (Çok büyük olursa bellek sorununa,
                            çok küçük olursa yavaşlığa neden olabilir)

    Döndürür:
        dict: İşlem sonuç raporu
            {
                "file_name": "rapor.pdf",
                "chunks":    12,
                "status":    "success" | "empty" | "already_exists"
            }

    Hatalar:
        FileNotFoundError : Dosya bulunamazsa
        ValueError        : Desteklenmeyen dosya formatı ise

    Örnek:
        async with DocumentDB() as db:
            client, model = await create_foundry_embedding_client()
            result = await ingest_file("rapor.pdf", db, client, model)
    """
    # ── Dosya yolunu Path nesnesine dönüştür ──
    file_path = Path(file_path)
    file_name = file_path.name  # Sadece dosya adı (ör: "rapor.pdf")

    # ── Dosyanın zaten yüklenmiş olup olmadığını kontrol et ──
    if await db.file_exists(file_name):
        logger.warning(f"⚠️ '{file_name}' zaten veritabanında mevcut, atlanıyor.")
        return {"file_name": file_name, "chunks": 0, "status": "already_exists"}

    # ══════════════════════════════════════════════════════════
    # ADIM 1: Dosyayı oku
    # ══════════════════════════════════════════════════════════
    logger.info(f"{'═' * 50}")
    logger.info(f"📄 Veri yükleme başlatılıyor: {file_name}")
    logger.info(f"{'═' * 50}")

    raw_text, page_count = await read_file(file_path)

    # Boş dosya kontrolü
    if not raw_text.strip():
        logger.warning(f"⚠️ Dosya boş veya metin çıkarılamadı: {file_name}")
        return {"file_name": file_name, "chunks": 0, "status": "empty"}

    logger.info(f"📄 Toplam metin uzunluğu: {len(raw_text)} karakter")

    # ══════════════════════════════════════════════════════════
    # ADIM 2: Metni akıllı parçalara ayır (overlapping chunking)
    # ══════════════════════════════════════════════════════════
    chunks = chunk_text(
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        logger.warning(f"⚠️ Parçalama sonrası metin parçası oluşmadı: {file_name}")
        return {"file_name": file_name, "chunks": 0, "status": "empty"}

    logger.info(
        f"✂️ {len(chunks)} parçaya ayrıldı "
        f"(boyut={chunk_size}, örtüşme={chunk_overlap})"
    )

    # ══════════════════════════════════════════════════════════
    # ADIM 3 & 4: Batch halinde embedding hesapla ve DB'ye yaz
    # ══════════════════════════════════════════════════════════
    # Tüm chunk'ları bir kerede göndermek yerine batch_size'lık
    # gruplar halinde işliyoruz. Bu:
    #   - Bellek kullanımını kontrol altında tutar
    #   - Büyük dosyalarda ilerleme takibini mümkün kılar
    #   - Hata durumunda kısmi kayıp riskini azaltır
    # ══════════════════════════════════════════════════════════

    total_inserted = 0  # Toplam eklenen parça sayacı
    total_batches = (len(chunks) + batch_size - 1) // batch_size  # Toplam batch sayısı

    for batch_index in range(0, len(chunks), batch_size):
        # ── Bu batch'teki chunk'ları seç ──
        batch_chunks = chunks[batch_index : batch_index + batch_size]
        current_batch_number = (batch_index // batch_size) + 1

        logger.info(
            f"  🔄 Batch {current_batch_number}/{total_batches}: "
            f"{len(batch_chunks)} parça işleniyor..."
        )

        # ── Bu batch için embedding vektörlerini hesapla ──
        # Foundry endpoint'ine asenkron istek gönder
        embeddings = await compute_embeddings(
            batch_chunks, embedding_client, embedding_model
        )

        # ── (dosya_adı, metin_parçası, vektör) tuple'ları oluştur ──
        records = [
            (file_name, chunk_text_item, vector)
            for chunk_text_item, vector in zip(batch_chunks, embeddings)
        ]

        # ── Toplu veritabanı ekleme (bulk insert) ──
        inserted = await db.insert_chunks_bulk(records)
        total_inserted += inserted

        logger.info(
            f"  ✅ Batch {current_batch_number}/{total_batches}: "
            f"{inserted} parça kaydedildi"
        )

    # ══════════════════════════════════════════════════════════
    # SONUÇ RAPORU
    # ══════════════════════════════════════════════════════════

    result = {
        "file_name": file_name,
        "chunks": total_inserted,
        "status": "success",
    }

    logger.info(f"{'═' * 50}")
    logger.info(
        f"🎉 Veri yükleme tamamlandı: {file_name} → "
        f"{total_inserted} parça veritabanına kaydedildi"
    )
    logger.info(f"{'═' * 50}")

    return result


# ============================================================
# BÖLÜM 5: ÇOK DOSYALI TOPLU YÜKLEME
# ============================================================

async def ingest_directory(
    directory_path: str | Path,
    db: DocumentDB,
    embedding_client,
    embedding_model: str = "qwen3-embedding-0.6b",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 16,
    extensions: Optional[tuple[str, ...]] = None,
) -> list[dict]:
    """
    Bir klasördeki tüm desteklenen dosyaları sırayla yükler.

    Parametreler:
        directory_path : Taranacak klasör yolu
        db, embedding_client, embedding_model, chunk_size,
        chunk_overlap, batch_size : ingest_file ile aynı
        extensions     : Yüklenecek dosya uzantıları
                         Varsayılan: (".pdf", ".txt", ".md")

    Döndürür:
        list[dict]: Her dosya için ingest_file sonuç raporu

    Örnek:
        sonuclar = await ingest_directory("belgeler/", db, client, model)
        for s in sonuclar:
            print(f"{s['file_name']}: {s['chunks']} parça ({s['status']})")
    """
    directory_path = Path(directory_path)

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Klasör bulunamadı: {directory_path}")

    # Varsayılan desteklenen uzantılar
    if extensions is None:
        extensions = (".pdf", ".txt", ".md")

    # Klasördeki uygun dosyaları bul ve sırala
    files = sorted([
        f for f in directory_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])

    if not files:
        logger.warning(f"⚠️ Klasörde desteklenen dosya bulunamadı: {directory_path}")
        return []

    logger.info(f"📂 {len(files)} dosya bulundu, yükleme başlıyor...")

    # Her dosyayı sırayla işle
    results = []
    for file_index, file_path in enumerate(files, start=1):
        logger.info(f"\n📁 [{file_index}/{len(files)}] {file_path.name}")
        result = await ingest_file(
            file_path=file_path,
            db=db,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )
        results.append(result)

    # Özet rapor
    success_count = sum(1 for r in results if r["status"] == "success")
    total_chunks = sum(r["chunks"] for r in results)
    logger.info(
        f"\n📊 Toplu yükleme özeti: {success_count}/{len(files)} dosya başarılı, "
        f"toplam {total_chunks} parça kaydedildi"
    )

    return results


# ────────────────────────────────────────────────────────
# TÜRKÇE ALIAS FONKSİYONLAR
# ────────────────────────────────────────────────────────
foundry_vektor_istemcisi_olustur = create_foundry_embedding_client
dosya_oku = read_file
_pdf_oku = _read_pdf
_metin_dosyasi_oku = _read_text_file
metni_parcala = chunk_text
dosya_yukle = ingest_file
klasor_yukle = ingest_directory
embedding_hesapla = compute_embeddings
