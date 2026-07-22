"""
core/veritabani.py — Project Antigravity: Veritabanı Altyapısı
=============================================================
Bu modül, projenin tüm veri saklama ve vektör arama işlemlerini
asenkron (async) bir SQLite veritabanı üzerinden yönetir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD MODE ÖZELLİĞİ — Cosine Similarity SQL Fonksiyonu (UDF):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kosinüs benzerliği hesaplaması Python döngüleriyle yapılmaz.
Bunun yerine, SQLite motoruna kaydedilen özel bir SQL fonksiyonu
(User-Defined Function) aracılığıyla doğrudan veritabanı
sorgusu içinde gerçekleştirilir:

    SELECT *, cosine_similarity(embedding_vector, ?) AS score
    FROM documents
    ORDER BY score DESC
    LIMIT 3

Bu sayede arama işlemi tek bir SQL sorgusuyla tamamlanır.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kullanım:
    async with DocumentDB() as db:
        await db.insert_chunk("rapor.pdf", "metin parçası", [0.1, 0.2, ...])
        sonuclar = await db.search_similar([0.15, 0.22, ...], top_k=3)
"""

# ============================================================
# İMPORT'LAR
# ============================================================

import json                     # Vektörleri JSON string'e dönüştürmek için
import logging                  # Loglama altyapısı
from pathlib import Path        # Dosya yolu yönetimi
from typing import Optional     # Tip ipuçları (type hints)

import aiosqlite                # Asenkron SQLite bağlantısı
import numpy as np              # Vektör hesaplamaları (UDF'de kullanılır)


# ============================================================
# LOGLAMA AYARLARI
# ============================================================

# Bu modüle özel bir logger oluştur
logger = logging.getLogger(__name__)


# ============================================================
# SABİTLER
# ============================================================

# Veritabanı dosyasının varsayılan konumu:
# Proje kökü / data / antigravity.db
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "antigravity.db"


# ============================================================
# BÖLÜM 1: KOSİNÜS BENZERLİĞİ — SQL KULLANICI TANIMLI FONKSİYONU
# ============================================================
# Bu fonksiyon Python tarafında tanımlanır ama SQLite motoruna
# kaydedildikten sonra tamamen SQL sorguları içinden çağrılır.
# Python'da satır satır döngü yapmaya gerek kalmaz.
# ============================================================

def _cosine_similarity_udf(vec_json_1: str, vec_json_2: str) -> float:
    """
    İki JSON-serialized vektör stringi arasındaki Kosinüs Benzerliğini hesaplar.

    Bu fonksiyon doğrudan çağrılmak için değil, SQLite motoruna
    kaydedilmek (register) içindir. SQL sorgularında şu şekilde kullanılır:

        cosine_similarity(embedding_vector, ?)

    Matematiksel Formül:
        cos(θ) = (A · B) / (‖A‖ × ‖B‖)

    Burada:
        A · B    = İki vektörün nokta çarpımı (dot product)
        ‖A‖      = A vektörünün büyüklüğü (Euclidean norm)
        ‖B‖      = B vektörünün büyüklüğü (Euclidean norm)

    Parametreler:
        vec_json_1 (str): Veritabanında saklanan vektör (JSON string)
        vec_json_2 (str): Sorgu vektörü (JSON string)

    Döndürür:
        float: -1.0 ile 1.0 arasında benzerlik skoru
               (1.0 = tamamen aynı yön, 0.0 = ilişkisiz)
    """
    try:
        # ── Adım 1: JSON string'leri numpy dizilerine dönüştür ──
        # float32 kullanarak bellek ve hız optimizasyonu sağlıyoruz
        vec1 = np.array(json.loads(vec_json_1), dtype=np.float32)
        vec2 = np.array(json.loads(vec_json_2), dtype=np.float32)

        # ── Adım 2: Nokta çarpımını (dot product) hesapla ──
        # A · B = Σ(a_i × b_i) — iki vektörün eleman eleman çarpımlarının toplamı
        dot_product = np.dot(vec1, vec2)

        # ── Adım 3: Her iki vektörün büyüklüğünü (norm) hesapla ──
        # ‖A‖ = √(Σ(a_i²)) — vektörün Öklid normu
        norm_1 = np.linalg.norm(vec1)
        norm_2 = np.linalg.norm(vec2)

        # ── Adım 4: Sıfır vektör kontrolü ──
        # Normlardan biri sıfırsa bölme hatası oluşur, 0.0 döndür
        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0

        # ── Adım 5: Kosinüs benzerliğini hesapla ve döndür ──
        similarity = dot_product / (norm_1 * norm_2)
        return float(similarity)

    except (json.JSONDecodeError, ValueError, TypeError) as hata:
        # Geçersiz veri durumunda sessizce 0.0 döndür
        # (Hatalı bir satır tüm sorguyu çökertmesin)
        logger.debug(f"UDF hatası: {hata}")
        return 0.0


# ============================================================
# BÖLÜM 2: DocumentDB SINIFI — ANA VERİTABANI YÖNETİCİSİ
# ============================================================

class DokumanVeritabani:
    """
    Projenin asenkron SQLite veritabanı yöneticisi.

    Bu sınıf şu görevleri üstlenir:
        1. Veritabanı bağlantısını açmak ve kapatmak
        2. 'documents' tablosunu oluşturmak
        3. Cosine Similarity UDF'ini SQLite motoruna kaydetmek
        4. Döküman parçalarını tekli veya toplu (bulk) eklemek
        5. Vektör benzerlik araması yapmak (SQL içinden)

    Kullanım Örneği:
        # Context manager ile (önerilen):
        async with DokumanVeritabani() as db:
            await db.parca_ekle("dosya.pdf", "metin", [0.1, ...])

        # Manuel kullanım:
        db = DokumanVeritabani(db_path=Path("ozel_yol.db"))
        await db.baslat()
        ...
        await db.kapat()
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        DokumanVeritabani nesnesini oluşturur.

        Parametreler:
            db_path (Path, opsiyonel): Veritabanı dosyasının yolu.
                Belirtilmezse varsayılan yol kullanılır:
                proje_kök/data/antigravity.db
        """
        # Veritabanı dosya yolunu ayarla (varsayılan veya özel)
        self._db_path: Path = db_path or _DEFAULT_DB_PATH

        # Asenkron bağlantı nesnesi — başlatılana kadar None
        self._db: Optional[aiosqlite.Connection] = None

    # ────────────────────────────────────────────────────────
    # 2a. BAĞLANTI YÖNETİMİ
    # ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Veritabanını başlatır. Sırasıyla şu adımları uygular:

        1. Veritabanı dosyasının bulunacağı klasörü oluşturur
        2. Asenkron SQLite bağlantısını açar
        3. Sorgu sonuçlarının sözlük (dict) gibi erişilebilir olmasını sağlar
        4. Cosine Similarity UDF'ini SQLite motoruna kaydeder
        5. 'documents' tablosunu oluşturur (henüz yoksa)
        """
        # ── Klasör oluştur ──
        # data/ klasörü yoksa otomatik oluştur (parents=True: ara klasörler de)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Asenkron bağlantıyı aç ──
        # aiosqlite, tüm sqlite3 işlemlerini arka plan thread'inde çalıştırır
        # check_same_thread=False: aiosqlite, SQLite bağlantısını kendi
        # arka plan thread'inde oluşturur. UDF kaydı (create_function) ise
        # farklı bir thread'den yapılır. Bu parametre, cross-thread erişime
        # izin vererek SQLite thread güvenliği hatasını önler.
        self._db = await aiosqlite.connect(
            str(self._db_path), check_same_thread=False
        )

        # ── Sorgu sonuçlarını sözlük olarak döndür ──
        # row["file_name"] gibi isimle erişebilmek için Row factory kullanıyoruz
        self._db.row_factory = aiosqlite.Row

        # ════════════════════════════════════════════════════
        # HARD MODE: Cosine Similarity UDF'ini kaydet
        # ════════════════════════════════════════════════════
        # aiosqlite, create_function() metodu sunmaz.
        # Çözüm: İç yapıdaki ham sqlite3.Connection nesnesine erişip
        # UDF'i doğrudan ona kaydediyoruz.
        #
        # Uyumluluk notu:
        #   - aiosqlite v0.17+: _conn özelliği kullanılır
        #   - Eski sürümler: _connection kullanılabilir
        #   - Hiçbiri yoksa: açıklayıcı hata fırlatılır
        # ════════════════════════════════════════════════════

        # Alttaki ham sqlite3.Connection nesnesini bul
        raw_connection = getattr(self._db, "_conn", None) or \
                         getattr(self._db, "_connection", None)

        if raw_connection is None:
            raise RuntimeError(
                "aiosqlite bağlantısının dahili sqlite3 nesnesine erişilemedi. "
                "Lütfen aiosqlite sürümünüzü kontrol edin (>=0.17 önerilir)."
            )

        # UDF'i SQLite motoruna kaydet
        # İlk parametre: SQL'de kullanılacak fonksiyon adı
        # İkinci parametre: Fonksiyonun kaç argüman alacağı (2 vektör)
        # Üçüncü parametre: Çağrılacak Python fonksiyonu
        raw_connection.create_function(
            "cosine_similarity",    # SQL'deki fonksiyon adı
            2,                      # Parametre sayısı
            _cosine_similarity_udf  # Yukarıda tanımlanan Python fonksiyonu
        )
        logger.info("✅ Cosine Similarity UDF, SQLite motoruna başarıyla kaydedildi.")

        # ── Tabloları oluştur ──
        await self._create_tables()
        logger.info(f"✅ Veritabanı hazır: {self._db_path}")

    async def _create_tables(self) -> None:
        """
        'documents' tablosunu oluşturur (IF NOT EXISTS — zaten varsa atlar).

        Tablo Şeması:
        ┌──────────────────┬─────────┬──────────────────────────────────┐
        │ Sütun            │ Tip     │ Açıklama                        │
        ├──────────────────┼─────────┼──────────────────────────────────┤
        │ id               │ INTEGER │ Otomatik artan birincil anahtar  │
        │ file_name        │ TEXT    │ Kaynak dosyanın adı (ör: a.pdf) │
        │ chunk_content    │ TEXT    │ Metin parçasının kendisi         │
        │ embedding_vector │ TEXT    │ JSON-serialized float listesi    │
        └──────────────────┴─────────┴──────────────────────────────────┘
        """
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name        TEXT    NOT NULL,
                chunk_content    TEXT    NOT NULL,
                embedding_vector TEXT    NOT NULL
            )
        """)
        
        # Geri bildirim tablosu (kullanıcı thumbs up/down)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sohbet geçmişi tablosu
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                chunks_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Değişiklikleri kalıcı hale getir
        await self._db.commit()
        logger.debug("📋 Tablolar hazır (documents, feedback, chat_history).")

    async def close(self) -> None:
        """
        Veritabanı bağlantısını güvenli şekilde kapatır.
        Kapatma işlemi idempotent'tir (birden fazla kez çağrılabilir).
        """
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("🔒 Veritabanı bağlantısı kapatıldı.")

    # ────────────────────────────────────────────────────────
    # 2b. ASYNC CONTEXT MANAGER (with bloğu desteği)
    # ────────────────────────────────────────────────────────

    async def __aenter__(self) -> "DocumentDB":
        """
        'async with' bloğuna girerken veritabanını başlatır.

        Örnek:
            async with DocumentDB() as db:
                # db kullanıma hazır
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        'async with' bloğundan çıkarken bağlantıyı kapatır.
        Hata olsa bile bağlantı güvenle kapatılır.
        """
        await self.close()

    # ────────────────────────────────────────────────────────
    # 2c. VERİ EKLEME İŞLEMLERİ
    # ────────────────────────────────────────────────────────

    async def insert_chunk(
        self,
        file_name: str,
        chunk_content: str,
        embedding_vector: list[float],
    ) -> int:
        """
        Tek bir döküman parçasını veritabanına ekler.

        Parametreler:
            file_name        : Kaynak dosyanın adı (ör: "rapor.pdf")
            chunk_content    : Metin parçası (chunk)
            embedding_vector : Embedding vektörü — float listesi

        Döndürür:
            int: Eklenen satırın otomatik üretilen ID'si

        Örnek:
            row_id = await db.insert_chunk(
                "rapor.pdf",
                "Yapay zekâ günümüzde...",
                [0.12, -0.34, 0.56, ...]
            )
        """
        # Vektörü JSON string'e dönüştür (TEXT sütununda saklanacak)
        vector_json = json.dumps(embedding_vector)

        # SQL INSERT sorgusu — parametreli (?) güvenli format
        cursor = await self._db.execute(
            """
            INSERT INTO documents (file_name, chunk_content, embedding_vector)
            VALUES (?, ?, ?)
            """,
            (file_name, chunk_content, vector_json),
        )

        # Değişikliği kalıcı yap
        await self._db.commit()

        # Eklenen satırın ID'sini döndür
        return cursor.lastrowid

    async def insert_chunks_bulk(
        self,
        records: list[tuple[str, str, list[float]]],
    ) -> int:
        """
        Birden fazla döküman parçasını tek seferde toplu olarak ekler (bulk insert).

        Bu yöntem, her parça için ayrı ayrı INSERT çalıştırmak yerine
        executemany() kullanarak çok daha hızlı çalışır.

        Parametreler:
            records: Her biri (dosya_adı, metin_parçası, vektör) olan
                     tuple'ların listesi.

                     Örnek:
                     [
                         ("rapor.pdf", "Birinci paragraf...", [0.1, 0.2, ...]),
                         ("rapor.pdf", "İkinci paragraf...",  [0.3, 0.4, ...]),
                     ]

        Döndürür:
            int: Eklenen toplam kayıt sayısı
        """
        if not records:
            logger.warning("⚠️ Boş kayıt listesi — ekleme yapılmadı.")
            return 0

        # ── Her kaydın vektörünü JSON string'e çevir ──
        # Veritabanına yazılacak nihai formata dönüştürüyoruz
        prepared_records = [
            (file_name, content, json.dumps(vector))
            for file_name, content, vector in records
        ]

        # ── Toplu ekleme (executemany) ──
        # Tek bir transaction içinde tüm satırları ekler — çok daha hızlı
        await self._db.executemany(
            """
            INSERT INTO documents (file_name, chunk_content, embedding_vector)
            VALUES (?, ?, ?)
            """,
            prepared_records,
        )

        # Tüm değişiklikleri tek seferde kalıcı yap
        await self._db.commit()

        inserted_count = len(prepared_records)
        logger.info(f"📦 {inserted_count} döküman parçası toplu olarak eklendi.")
        return inserted_count

    # ────────────────────────────────────────────────────────
    # 2d. VEKTÖR BENZERLİK ARAMASI (HARD MODE)
    # ────────────────────────────────────────────────────────

    async def search_similar(
        self,
        query_vector: list[float],
        file_filter: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Sorgu vektörüne en benzer döküman parçalarını bulur.

        ╔══════════════════════════════════════════════════════════╗
        ║  HARD MODE: Bu fonksiyon Python tarafında hiçbir        ║
        ║  döngü çalıştırmaz. Tüm kosinüs benzerliği hesabı     ║
        ║  SQL içinde, kaydedilmiş UDF aracılığıyla yapılır.     ║
        ║                                                         ║
        ║  SQL Sorgusu:                                           ║
        ║    SELECT *, cosine_similarity(embedding_vector, ?)     ║
        ║    FROM documents                                       ║
        ║    ORDER BY score DESC                                  ║
        ║    LIMIT ?                                              ║
        ╚══════════════════════════════════════════════════════════╝

        Parametreler:
            query_vector : Sorgunun embedding vektörü (float listesi)
            file_filter  : Opsiyonel olarak aranacak dosya adları listesi
            top_k        : Döndürülecek en benzer sonuç sayısı (varsayılan: 3)

        Döndürür:
            Liste[dict]: Her biri şu anahtarları içeren sözlükler:
                - "id"            : Satır ID'si
                - "file_name"     : Kaynak dosya adı
                - "chunk_content" : Metin parçası
                - "score"         : Kosinüs benzerlik skoru (0.0 — 1.0)

        Örnek:
            sonuclar = await db.search_similar([0.15, 0.22, ...], top_k=5)
            for s in sonuclar:
                print(f"{s['score']:.3f}  {s['file_name']}  {s['chunk_content'][:80]}")
        """
        # Sorgu vektörünü JSON string'e çevir — UDF bu formatı bekliyor
        query_json = json.dumps(query_vector)

        # ════════════════════════════════════════════════════
        # SQL İÇİNDE COSINE SİMİLARİTY HESABI
        # ════════════════════════════════════════════════════
        # cosine_similarity() fonksiyonu, initialize() aşamasında
        # SQLite motoruna kaydedilen UDF'dir.
        # Her satırın embedding_vector sütununu sorgu vektörüyle
        # karşılaştırır ve benzerlik skorunu hesaplar.
        # Sonuçlar skora göre azalan sırada sıralanır.
        # ════════════════════════════════════════════════════
        
        file_clause = ""
        params = [query_json]
        
        if file_filter:
            placeholders = ",".join("?" for _ in file_filter)
            file_clause = f"WHERE file_name IN ({placeholders})"
            params.extend(file_filter)
            
        params.append(top_k)

        cursor = await self._db.execute(
            f"""
            SELECT
                id,
                file_name,
                chunk_content,
                cosine_similarity(embedding_vector, ?) AS score
            FROM documents
            {file_clause}
            ORDER BY score DESC
            LIMIT ?
            """,
            params,
        )

        # Tüm sonuç satırlarını çek
        rows = await cursor.fetchall()

        # Her satırı erişimi kolay bir sözlüğe dönüştür
        results = [
            {
                "id": row["id"],
                "file_name": row["file_name"],
                "chunk_content": row["chunk_content"],
                "score": row["score"],
            }
            for row in rows
        ]

        logger.info(
            f"🔍 Arama tamamlandı: {len(results)} sonuç bulundu "
            f"(en yüksek skor: {results[0]['score']:.4f})" if results else
            "🔍 Arama tamamlandı: sonuç bulunamadı."
        )

        return results

    # ────────────────────────────────────────────────────────
    # 2e. YARDIMCI FONKSİYONLAR
    # ────────────────────────────────────────────────────────

    async def get_document_count(self) -> int:
        """Veritabanındaki toplam döküman parçası sayısını döndürür."""
        cursor = await self._db.execute("SELECT COUNT(*) AS cnt FROM documents")
        row = await cursor.fetchone()
        return row["cnt"]

    async def get_file_names(self) -> list[str]:
        """
        Veritabanında kayıtlı tüm benzersiz dosya adlarını döndürür.

        Döndürür:
            list[str]: Alfabetik sırada dosya adları listesi
                       Örnek: ["makale.pdf", "rapor.pdf", "notlar.txt"]
        """
        cursor = await self._db.execute(
            "SELECT DISTINCT file_name FROM documents ORDER BY file_name"
        )
        rows = await cursor.fetchall()
        return [row["file_name"] for row in rows]

    async def delete_by_file(self, file_name: str) -> int:
        """
        Belirtilen dosyaya ait tüm döküman parçalarını siler.

        Parametreler:
            file_name: Silinecek dosyanın adı (ör: "eski_rapor.pdf")

        Döndürür:
            int: Silinen satır sayısı
        """
        cursor = await self._db.execute(
            "DELETE FROM documents WHERE file_name = ?",
            (file_name,),
        )
        await self._db.commit()

        deleted_count = cursor.rowcount
        logger.info(f"🗑️ '{file_name}' dosyasına ait {deleted_count} parça silindi.")
        return deleted_count

    async def file_exists(self, file_name: str) -> bool:
        """Belirtilen dosyanın veritabanında kayıtlı olup olmadığını kontrol eder."""
        cursor = await self._db.execute(
            "SELECT 1 FROM documents WHERE file_name = ? LIMIT 1",
            (file_name,),
        )
        row = await cursor.fetchone()
        return row is not None

    async def get_document_stats(self) -> list[dict]:
        """
        Veritabanındaki dökümanların parça (chunk) sayılarını döndürür.
        
        Döndürür:
            list[dict]: Her bir dosya için dosya adı ve parça sayısı.
        """
        cursor = await self._db.execute(
            "SELECT file_name, COUNT(*) AS chunk_count FROM documents GROUP BY file_name ORDER BY chunk_count DESC"
        )
        rows = await cursor.fetchall()
        return [
            {"file_name": row["file_name"], "chunk_count": row["chunk_count"]}
            for row in rows
        ]

    async def keyword_search(self, query: str, file_filter: list[str] | None = None, top_k: int = 5) -> list[dict]:
        """
        Anahtar kelime tabanlı metin araması yapar.
        Sorguyu kelimelere böler ve LIKE ile her kelimeyi arar.
        Eşleşen kelime sayısına göre skor hesaplar.
        """
        # Split query into words
        words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
        if not words:
            return []
        
        # Build SQL: count matching words per chunk
        conditions = " + ".join([
            f"(CASE WHEN LOWER(chunk_content) LIKE '%' || ? || '%' THEN 1 ELSE 0 END)"
            for _ in words
        ])
        
        file_clause = ""
        params = list(words)
        if file_filter:
            placeholders = ",".join("?" for _ in file_filter)
            file_clause = f" AND file_name IN ({placeholders})"
            params.extend(file_filter)
        
        sql = f"""
            SELECT id, file_name, chunk_content,
                   ({conditions}) * 1.0 / {len(words)} AS score
            FROM documents
            WHERE ({conditions}) > 0 {file_clause}
            ORDER BY score DESC
            LIMIT ?
        """
        # params need to be doubled because conditions appear twice
        all_params = list(words) + params + [top_k]
        
        cursor = await self._db.execute(sql, all_params)
        rows = await cursor.fetchall()
        return [{"id": r["id"], "file_name": r["file_name"], "chunk_content": r["chunk_content"], "score": r["score"]} for r in rows]

    async def save_feedback(self, session_id: str, message_index: int, rating: int) -> None:
        """Kullanıcı geri bildirimini (thumbs up/down) kaydeder."""
        await self._db.execute(
            "INSERT INTO feedback (session_id, message_index, rating) VALUES (?, ?, ?)",
            (session_id, message_index, rating)
        )
        await self._db.commit()

    async def get_feedback_stats(self) -> dict:
        """Toplam geri bildirim istatistiklerini döndürür."""
        cursor = await self._db.execute("SELECT rating, COUNT(*) as cnt FROM feedback GROUP BY rating")
        rows = await cursor.fetchall()
        stats = {"positive": 0, "negative": 0}
        for row in rows:
            if row["rating"] == 1:
                stats["positive"] = row["cnt"]
            elif row["rating"] == -1:
                stats["negative"] = row["cnt"]
        return stats

    async def save_message(self, session_id: str, role: str, content: str, chunks_json: str | None = None) -> None:
        """Sohbet mesajını veritabanına kaydeder."""
        await self._db.execute(
            "INSERT INTO chat_history (session_id, role, content, chunks_json) VALUES (?, ?, ?, ?)",
            (session_id, role, content, chunks_json)
        )
        await self._db.commit()

    async def get_chat_sessions(self) -> list[dict]:
        """Kayıtlı sohbet oturumlarını döndürür."""
        cursor = await self._db.execute(
            "SELECT session_id, MIN(created_at) as started, COUNT(*) as msg_count "
            "FROM chat_history GROUP BY session_id ORDER BY started DESC LIMIT 20"
        )
        rows = await cursor.fetchall()
        return [{"session_id": r["session_id"], "started": r["started"], "msg_count": r["msg_count"]} for r in rows]

    async def get_session_messages(self, session_id: str) -> list[dict]:
        """Belirli bir oturumun mesajlarını döndürür."""
        cursor = await self._db.execute(
            "SELECT role, content, chunks_json, created_at FROM chat_history WHERE session_id = ? ORDER BY id",
            (session_id,)
        )
        rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"], "chunks_json": r["chunks_json"], "created_at": r["created_at"]} for r in rows]

    async def delete_session(self, session_id: str) -> None:
        """Belirli bir sohbet oturumunu siler."""
        await self._db.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        await self._db.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
        await self._db.commit()

    # ────────────────────────────────────────────────────────
    # TÜRKÇE ALIAS METOTLAR (Geriye Dönük Uyumluluk)
    # ────────────────────────────────────────────────────────
    baslat = initialize
    kapat = close
    parca_ekle = insert_chunk
    parcalari_toplu_ekle = insert_chunks_bulk
    benzer_ara = search_similar
    dokuman_sayisini_getir = get_document_count
    dosya_adlarini_getir = get_file_names
    dosyaya_gore_sil = delete_by_file
    dosya_var_mi = file_exists
    dokuman_istatistiklerini_getir = get_document_stats
    anahtar_kelime_ara = keyword_search
    geri_bildirim_kaydet = save_feedback
    geri_bildirim_istatistiklerini_getir = get_feedback_stats
    mesaj_kaydet = save_message
    sohbet_oturumlarini_getir = get_chat_sessions
    oturum_mesajlarini_getir = get_session_messages
    oturumu_sil = delete_session


# Geriye dönük sınıf alias'ı
DocumentDB = DokumanVeritabani

