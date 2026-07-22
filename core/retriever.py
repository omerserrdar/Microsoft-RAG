"""
core/retriever.py — Project Antigravity: Arama ve Getirme Motoru
================================================================
Bu modül, RAG (Retrieval-Augmented Generation) sisteminin
"Retrieval" (Arama) katmanını yönetir.

İşlem Akışı:
    ┌──────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────┐
    │ Kullanıcı    │ ──▸ │ Soru Vektörü   │ ──▸ │ SQL Cosine   │ ──▸ │ En Alakalı   │
    │ Sorusu       │     │ (Embedding)    │     │ Similarity   │     │ Parçalar     │
    └──────────────┘     └────────────────┘     └──────────────┘     └──────────────┘

    Adım 1: Kullanıcının doğal dildeki sorusunu, Foundry SDK
            üzerindeki 'qwen3-embedding-0.6b' modeline gönderip
            sayısal bir vektöre (embedding) dönüştürüyoruz.

    Adım 2: Bu vektörü, SQLite veritabanındaki 'cosine_similarity'
            UDF fonksiyonuna parametre olarak vererek, veritabanındaki
            tüm döküman parçalarıyla karşılaştırıyoruz.

    Adım 3: En yüksek benzerlik skoruna sahip ilk 3 (top_k) döküman
            parçasını (chunk_content, file_name, score) döndürüyoruz.

Kullanım:
    from core.database import DocumentDB
    from core.retriever import retrieve_relevant_chunks
    from core.ingester import create_foundry_embedding_client

    async with DocumentDB() as db:
        client, model = await create_foundry_embedding_client()
        results = await retrieve_relevant_chunks(
            "Kızıl gezegen hangisidir?", db, client, model
        )
        for r in results:
            print(f"{r['score']:.4f} | {r['file_name']} | {r['chunk_content'][:80]}")
"""

# ============================================================
# İMPORT'LAR
# ============================================================

import logging              # Loglama altyapısı

# ── Proje İçi İmportlar ──
from core.database import DocumentDB          # Veritabanı yöneticisi
from core.ingester import compute_embeddings  # Vektör hesaplama fonksiyonu


# ============================================================
# LOGLAMA AYARLARI
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# ANA FONKSİYON: BAĞLAM GETİRME (RETRIEVAL)
# ============================================================

async def retrieve_relevant_chunks(
    question: str,
    db: DocumentDB,
    embedding_client,
    embedding_model: str = "qwen3-embedding-0.6b",
    file_filter: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Kullanıcının sorusuna en alakalı döküman parçalarını bulur ve döndürür.

    Bu fonksiyon, RAG pipeline'ının kalbidir. İki temel adımı
    tek bir çağrıda birleştirir:
        1. Soruyu vektöre çevir (Embedding)
        2. Veritabanında en benzer parçaları bul (Cosine Similarity)

    Parametreler:
        question         : Kullanıcının doğal dildeki sorusu
                           Örnek: "Mars'ın rengi neden kırmızıdır?"
        db               : Başlatılmış DocumentDB instance'ı
                           (cosine_similarity UDF'i kayıtlı olmalı)
        embedding_client : AsyncOpenAI embedding istemcisi
                           (create_foundry_embedding_client'tan)
        embedding_model  : Embedding modelinin katalog adı
                           Varsayılan: "qwen3-embedding-0.6b"
        file_filter      : Opsiyonel dosya filtresi (sadece bu dosyalarda ara)
        top_k            : Döndürülecek en benzer sonuç sayısı
                           Varsayılan: 3

    Döndürür:
        list[dict]: Her biri şu anahtarları içeren sözlükler listesi:
            - "id"            : Veritabanı satır ID'si (integer)
            - "file_name"     : Kaynak dosyanın adı (ör: "rapor.pdf")
            - "chunk_content" : Metin parçasının kendisi (string)
            - "score"         : Kosinüs benzerlik skoru (0.0 — 1.0)

        Sonuçlar benzerlik skoruna göre azalan sırada döner.
        Boş veritabanı durumunda boş liste döner.

    Hatalar:
        Veritabanı bağlantısı kesilmişse veya embedding modeli
        yanıt vermezse ilgili kütüphanenin hatası fırlatılır.

    Örnek:
        sonuclar = await retrieve_relevant_chunks(
            question="Güneş sistemi kaç gezegenden oluşur?",
            db=db,
            embedding_client=client,
            embedding_model="qwen3-embedding-0.6b",
            file_filter=None,
            top_k=3,
        )
        for s in sonuclar:
            print(f"Skor: {s['score']:.4f} | Dosya: {s['file_name']}")
            print(f"Metin: {s['chunk_content'][:100]}...")
    """
    # ── Giriş logla ──
    logger.info(f"🔎 Arama başlatılıyor: \"{question[:80]}\"")

    # ══════════════════════════════════════════════════════════
    # ADIM 1: SORUYU VEKTÖRE ÇEVİR
    # ══════════════════════════════════════════════════════════
    # Kullanıcının doğal dildeki sorusunu, embedding modeline
    # göndererek sayısal bir temsiline (vektörüne) dönüştürüyoruz.
    #
    # compute_embeddings() birden fazla metin kabul eder ancak
    # burada tek bir soru gönderiyoruz, bu yüzden sonucun
    # ilk elemanını ([0]) alıyoruz.
    # ══════════════════════════════════════════════════════════

    query_vectors = await compute_embeddings(
        texts=[question],
        client=embedding_client,
        model_name=embedding_model,
    )
    # İlk (ve tek) metin için vektörü al
    query_vector = query_vectors[0]

    logger.debug(
        f"🔢 Sorgu vektörü hesaplandı "
        f"(boyut: {len(query_vector)})"
    )

    # ══════════════════════════════════════════════════════════
    # ADIM 2: VERİTABANINDA BENZERLİK ARAMASI YAP
    # ══════════════════════════════════════════════════════════
    # DocumentDB.search_similar() metodu, SQL içindeki
    # cosine_similarity UDF'ini kullanarak tüm döküman
    # parçalarını sorgu vektörüyle karşılaştırır.
    #
    # Bu işlem Python döngüsü kullanmaz — tamamı SQL
    # içinde, tek bir sorguyla gerçekleşir (HARD MODE).
    #
    # SQL sorgusu (dahili):
    #   SELECT *, cosine_similarity(embedding_vector, ?) AS score
    #   FROM documents
    #   ORDER BY score DESC
    #   LIMIT ?
    # ══════════════════════════════════════════════════════════

    results = await db.search_similar(
        query_vector=query_vector,
        file_filter=file_filter,
        top_k=top_k,
    )

    # ── Sonuç özetini logla ──
    if results:
        logger.info(
            f"🔎 {len(results)} sonuç bulundu "
            f"(en yüksek skor: {results[0]['score']:.4f}, "
            f"en düşük skor: {results[-1]['score']:.4f})"
        )
        # Detaylı sonuç logu (debug seviyesi)
        for idx, result in enumerate(results, start=1):
            logger.debug(
                f"  📄 #{idx}: {result['file_name']} | "
                f"Skor: {result['score']:.4f} | "
                f"Metin: \"{result['chunk_content'][:60]}...\""
            )
    else:
        logger.warning(
            "🔎 Arama sonucu bulunamadı. "
            "Veritabanında döküman olmayabilir."
        )

    return results


# ============================================================
# HİBRİT ARAMA (SEMANTİK + ANAHTAR KELİME)
# ============================================================

async def hybrid_retrieve(
    question: str,
    db: DocumentDB,
    embedding_client,
    embedding_model: str = "qwen3-embedding-0.6b",
    file_filter: list[str] | None = None,
    top_k: int = 5,
    semantic_weight: float = 0.7,
) -> list[dict]:
    """
    Hibrit arama: Vektör benzerliği (semantic) ve anahtar kelime (keyword)
    aramalarını birleştirerek daha isabetli sonuçlar üretir.

    Algoritma:
        1. Semantik arama (cosine similarity) ile top_k*2 sonuç getir
        2. Anahtar kelime araması (LIKE) ile top_k*2 sonuç getir  
        3. İki sonuç kümesini birleştir:
           - Her chunk için: final_score = semantic_weight * semantic_score + (1-semantic_weight) * keyword_score
           - Her iki listede de bulunan chunk'lar doğal olarak daha yüksek skor alır
        4. En yüksek final skorlu top_k sonucu döndür

    Parametreler:
        question        : Kullanıcının sorusu
        db              : DocumentDB instance'ı
        embedding_client: Embedding istemcisi
        embedding_model : Embedding model adı
        file_filter     : Opsiyonel dosya filtresi
        top_k           : Döndürülecek sonuç sayısı
        semantic_weight : Semantik aramanın ağırlığı (0.0-1.0)

    Döndürür:
        list[dict]: Birleştirilmiş ve skorlanmış sonuçlar
    """
    logger.info(f"🔄 Hibrit arama başlatılıyor: \"{question[:80]}\"")
    
    # 1. Semantik sonuçları getir (top_k * 2)
    semantic_results = await retrieve_relevant_chunks(
        question=question,
        db=db,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        file_filter=file_filter,
        top_k=top_k * 2,
    )
    
    # 2. Anahtar kelime sonuçlarını getir (top_k * 2)
    keyword_results = await db.keyword_search(
        query=question,
        file_filter=file_filter,
        top_k=top_k * 2,
    )
    
    # 3. İki sonuç kümesini birleştir
    merged_results = {}
    
    # Semantik sonuçları ekle
    for res in semantic_results:
        chunk_id = res["id"]
        merged_results[chunk_id] = {
            "id": chunk_id,
            "file_name": res["file_name"],
            "chunk_content": res["chunk_content"],
            "semantic_score": res["score"],
            "keyword_score": 0.0,
        }
        
    # Anahtar kelime sonuçlarını ekle
    for res in keyword_results:
        chunk_id = res["id"]
        if chunk_id in merged_results:
            merged_results[chunk_id]["keyword_score"] = res["score"]
        else:
            merged_results[chunk_id] = {
                "id": chunk_id,
                "file_name": res["file_name"],
                "chunk_content": res["chunk_content"],
                "semantic_score": 0.0,
                "keyword_score": res["score"],
            }
            
    # Final skorları hesapla
    final_results = []
    for chunk_id, data in merged_results.items():
        s_score = data["semantic_score"]
        k_score = data["keyword_score"]
        final_score = (semantic_weight * s_score) + ((1.0 - semantic_weight) * k_score)
        
        data["score"] = final_score
        final_results.append(data)
        
    # 4. Sonuçları final skoruna göre sırala ve top_k döndür
    final_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = final_results[:top_k]
    
    logger.info(f"🔄 Hibrit arama tamamlandı. {len(top_results)} sonuç döndürülüyor.")
    return top_results
