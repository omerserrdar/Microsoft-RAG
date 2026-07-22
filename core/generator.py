"""
core/generator.py — Project Antigravity: Yerel LLM ve Yanıt Üretim Motoru
==========================================================================
Bu modül, RAG (Retrieval-Augmented Generation) sisteminin
"Generation" (Üretim) katmanını yönetir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZORLUK ÖZELLİĞİ — Async Streaming (Akışkan Yanıt Üretimi):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Yanıt üretilirken kullanıcının tüm cevabı beklemesini önlemek için,
yerel LLM'den gelen çıktılar token token (kelime parçası) olarak
akışkan bir şekilde iletilir.

Bu, Python'un "async generator" (asenkron üreteç) mekanizması ile
gerçekleştirilir: fonksiyon her token hazır olduğunda `yield` ile
fırlatır, çağıran taraf `async for` ile okur.

İşlem Akışı:
    ┌──────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────┐
    │ Soru +       │ ──▸ │ System Prompt  │ ──▸ │ phi-3.5-mini │ ──▸ │ Streaming    │
    │ Bağlam       │     │ (Dürüstlük)   │     │ Yerel LLM    │     │ Yanıt + Atıf │
    └──────────────┘     └────────────────┘     └──────────────┘     └──────────────┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kullanım:
    from core.generator import create_foundry_chat_client, generate_streaming_response

    client, model = await create_foundry_chat_client()
    async for token in generate_streaming_response(
        "Mars nedir?", context_chunks, client, model
    ):
        print(token, end="", flush=True)
"""

# ============================================================
# İMPORT'LAR
# ============================================================

import asyncio                       # Asenkron yapı ve thread delegasyonu
import logging                       # Loglama altyapısı
from typing import AsyncGenerator    # Tip ipuçları (async generator)


# ============================================================
# LOGLAMA AYARLARI
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# BÖLÜM 1: SİSTEM İSTEMİ (SYSTEM PROMPT) — DÜRÜSTLÜK KURALI
# ============================================================
# Bu istem, modelin davranışını belirler. Model YALNIZCA
# getirilen döküman parçalarına bakarak cevap verir.
# Kaynakta yanıt yoksa uydurmaz, bunu açıkça belirtir.
#
# Bu yaklaşım "grounding" olarak adlandırılır ve RAG
# sistemlerinde halüsinasyonu önlemenin temel yöntemidir.
# ============================================================

_SYSTEM_PROMPT = (
    "Sen belgelere dayalı soruları yanıtlayan dürüst ve yardımsever bir yapay zekâ asistanısın.\n"
    "Sana sunulan belge metnini okuyarak kullanıcının sorusuna doğrudan, net ve Türkçe yanıt ver.\n"
    "Üçüncü şahıs analizleri ('Kullanıcı şunu istiyor' vb.), düşünce adımları veya şablon notları yazma; sadece cevabın kendisini ver.\n"
    "Eğer aranan soru belge metninde hiç yoksa: 'Bu bilgi yüklenen belgelerde bulunmamaktadır.' yaz."
)


# ============================================================
# BÖLÜM 2: FOUNDRY SOHBET İSTEMCİSİ OLUŞTURMA
# ============================================================
# Foundry SDK singleton yapıdadır. Eğer embedding modeli için
# zaten başlatıldıysa, mevcut instance yeniden kullanılır.
# Yeni bir model (phi-3.5-mini) eklenerek web servisine kaydedilir.
# ============================================================

async def create_foundry_chat_client(
    model_name: str = "phi-3.5-mini",
) -> tuple:
    """
    Microsoft Foundry Local SDK'yı başlatır, sohbet modelini
    indirir/yükler ve OpenAI-uyumlu asenkron bir istemci döndürür.

    Bu fonksiyon, Foundry SDK'nın singleton yapısını kullanır:
        - SDK zaten başlatıldıysa → mevcut instance kullanılır
        - Model önbellekteyse → indirme adımı atlanır
        - Model yüklüyse → yükleme adımı atlanır
        - Web servisi çalışıyorsa → yeniden başlatılmaz

    Parametreler:
        model_name: Kullanılacak sohbet modelinin katalog adı.
                    Varsayılan: "phi-3.5-mini"
                    Foundry katalogundaki diğer seçenekler:
                    "phi-4-mini", "qwen2.5-7b", "qwen3-4b" vb.

    Döndürür:
        tuple: (AsyncOpenAI_client, model_adı)
            - client   : Sohbet istekleri göndermek için hazır istemci
            - model_adı: Modelin katalog adı (API çağrılarında kullanılır)

    Örnek:
        client, model = await create_foundry_chat_client()
        response = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Merhaba"}]
        )
    """
    def _initialize_chat_sdk():
        """
        Thread içinde çalışacak senkron SDK başlatma fonksiyonu.

        Foundry SDK'nın tüm yönetim işlemleri (initialize, download, load)
        senkron API sunar, bu yüzden asyncio.to_thread() ile
        arka plan thread'ine delege ediyoruz.
        """
        from foundry_local_sdk import Configuration, FoundryLocalManager

        # ── SDK'yı başlat veya mevcut instance'ı kullan ──
        # Foundry SDK singleton'dır: ilk başlatmadan sonra
        # .instance özelliği üzerinden erişilir.
        # Eğer embedding modülü SDK'yı zaten başlattıysa,
        # burada tekrar başlatmaya gerek yok.
        try:
            # Mevcut instance'ı almayı dene
            manager = FoundryLocalManager.instance
            logger.info("♻️  Mevcut Foundry SDK instance'ı kullanılıyor.")
        except Exception:
            # Henüz başlatılmadıysa, yeni başlat
            config = Configuration(app_name="project_antigravity")
            FoundryLocalManager.initialize(config)
            manager = FoundryLocalManager.instance
            logger.info("🆕 Foundry SDK ilk kez başlatıldı.")

        # ── Sohbet modelini katalogdan al ──
        model = manager.catalog.get_model(model_name)
        logger.info(f"📦 Sohbet modeli bulundu: {model_name}")

        # ── Modeli indir (önbellekteyse bu adım atlanır) ──
        if not model.is_cached:
            logger.info("⬇️  Model indiriliyor (bu ilk seferde biraz sürebilir)...")
            model.download(
                lambda progress: logger.debug(
                    f"  ⬇️  İndirme ilerlemesi: {progress:.1f}%"
                )
            )
            logger.info("⬇️  Model indirme tamamlandı.")
        else:
            logger.info("✅ Model önbellekte mevcut, indirme atlanıyor.")

        # ── Modeli belleğe yükle (zaten yüklüyse atlanır) ──
        if not model.is_loaded:
            logger.info("🧠 Model belleğe yükleniyor...")
            model.load()
            logger.info(f"✅ Sohbet modeli belleğe yüklendi: {model_name}")
        else:
            logger.info("✅ Model zaten bellekte.")

        # ── Yerel web servisini başlat ──
        # Web servisi, OpenAI-uyumlu bir REST API sunar.
        # Zaten çalışıyorsa bu çağrı güvenle atlanır.
        try:
            manager.start_web_service()
        except Exception:
            # Web servisi zaten çalışıyor olabilir — sorun değil
            pass

        # Yerel endpoint URL'sini al
        base_url = f"{manager.urls[0]}/v1"
        logger.info(f"🌐 Yerel sohbet endpoint'i aktif: {base_url}")

        return base_url

    # ── Senkron SDK başlatmayı arka plan thread'inde çalıştır ──
    # Bu sayede ana event loop bloklanmaz
    base_url = await asyncio.to_thread(_initialize_chat_sdk)

    # ── OpenAI-uyumlu asenkron istemciyi oluştur ──
    # Foundry'nin yerel endpoint'i OpenAI API formatını destekler,
    # bu yüzden standart openai kütüphanesini kullanabiliyoruz.
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=base_url,    # Yerel Foundry endpoint'ine yönlendir
        api_key="local",      # Yerel çalıştığımız için API key gerekli değil
    )

    return client, model_name


# ============================================================
# BÖLÜM 3: STREAMING YANIT ÜRETİMİ (ASYNC GENERATOR)
# ============================================================
# Bu bölüm, RAG sisteminin en kritik çıktı aşamasıdır.
# Retriever'dan gelen bağlam parçalarını, kullanıcının sorusuyla
# birleştirip yerel LLM'e gönderir ve yanıtı token token döndürür.
#
# NEDEN STREAMING?
#   Büyük dil modelleri yanıt üretmek için saniyeler sürebilir.
#   Tüm yanıtı beklemek yerine, her token hazır olduğunda
#   anında kullanıcıya iletmek çok daha iyi bir deneyim sunar.
#
# TEKNİK YAPI:
#   - Python async generator (async def + yield)
#   - OpenAI streaming API (stream=True)
#   - Her chunk'tan delta.content çıkarılarak yield edilir
# ============================================================

async def generate_streaming_response(
    question: str,
    context_chunks: list[dict],
    client,
    model_name: str = "phi-3.5-mini",
    chat_history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Soru ve bağlam parçalarını kullanarak yerel LLM'den
    streaming (akışkan) yanıt üretir.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ASYNC GENERATOR: Bu fonksiyon bir async generator'dır.
    Her çağrıda bir token (kelime veya kelime parçası) yield eder.
    Çağıran taraf `async for token in ...` ile tüketir.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    İşlem Adımları:
        1. Bağlam metnini oluştur (retriever'dan gelen parçalar)
        2. System prompt + kullanıcı mesajını birleştir
        3. OpenAI streaming API ile yerel modelden yanıt al
        4. Her token'ı yield et (async generator)
        5. Yanıt sonuna kaynak dosya dipnotlarını (citations) ekle

    Parametreler:
        question       : Kullanıcının doğal dildeki sorusu
        context_chunks : Retriever'dan gelen sonuç listesi.
                         Her eleman şu anahtarları içerir:
                         - "file_name"     : Kaynak dosya adı
                         - "chunk_content" : Metin parçası
                         - "score"         : Benzerlik skoru
        client         : AsyncOpenAI istemcisi
                         (create_foundry_chat_client'tan)
        model_name     : Sohbet modelinin katalog adı

    Yields:
        str: Token token üretilen yanıt parçaları.
             Son yield, kaynak dipnotlarını (citations) içerir.

    Örnek:
        async for token in generate_streaming_response(
            "Mars nedir?", context_chunks, client, model
        ):
            print(token, end="", flush=True)
        # Çıktı: "Mars, Güneş sisteminin... \n\n📚 Kaynaklar: rapor.pdf"
    """
    logger.info(f"💬 Yanıt üretimi başlatılıyor: \"{question[:60]}\"")

    # ══════════════════════════════════════════════════════════
    # ADIM 1: BAĞLAM METNİNİ OLUŞTUR
    # ══════════════════════════════════════════════════════════
    # Retriever'dan gelen döküman parçalarını, LLM'in
    # anlayabileceği yapılandırılmış bir metin formatına
    # dönüştürüyoruz. Her parçanın kaynağı ve benzerlik
    # skoru da ekleniyor.
    # ══════════════════════════════════════════════════════════

    if context_chunks:
        # Metin parçalarını temiz bir şekilde birleştir (Parça No ekleme)
        context_parts = [chunk["chunk_content"] for chunk in context_chunks]
        context_text = "\n\n---\n\n".join(context_parts)
    else:
        context_text = "(Veritabanında döküman bulunmuyor.)"

    user_message = (
        f"Belge Metni:\n"
        f"------------\n"
        f"{context_text}\n"
        f"------------\n\n"
        f"Soru: {question}\n\n"
        f"Cevap:"
    )

    logger.debug(
        f"📝 Prompt hazırlandı — "
        f"Bağlam: {len(context_chunks)} parça, "
        f"Mesaj uzunluğu: {len(user_message)} karakter"
    )

    # ══════════════════════════════════════════════════════════
    # ADIM 3: STREAMING YANIT AL (ASYNC GENERATOR)
    # ══════════════════════════════════════════════════════════
    # OpenAI streaming API kullanarak yerel modelden
    # token token yanıt alıyoruz.
    #
    # stream=True parametresi, modelin her token'ı hazır
    # olduğunda göndermesini sağlar. Bu sayede kullanıcı
    # tüm yanıtı beklemek zorunda kalmaz.
    #
    # temperature=0.3: Düşük sıcaklık, daha tutarlı ve
    # odaklı yanıtlar üretir (RAG için ideal).
    # ══════════════════════════════════════════════════════════

    # Mesaj listesini oluştur
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Eğer sohbet geçmişi varsa, sadece son 2 mesajı al (hız için az token)
    if chat_history:
        recent_history = chat_history[-2:]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Mevcut kullanıcı mesajını ekle
    messages.append({"role": "user", "content": user_message})

    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,         # Streaming modu açık
        temperature=0.2,     # Düşük sıcaklık = daha hızlı ve tutarlı üretim
        max_tokens=512,      # Hızlı ve özlü yanıtlar için makul limit
    )

    # ── Her token'ı yield et ──
    # Streaming yanıtta her chunk, bir "delta" (değişiklik) içerir.
    # delta.content, o adımdaki yeni token'ı (kelime parçasını) barındırır.
    # Boş delta'lar atlanır (ör: fonksiyon çağrısı başlangıcı).
    token_count = 0
    async for chunk in stream:
        # Bazı durumlarda (akışın sonu veya kullanım istatistikleri) choices listesi boş gelebilir.
        # Bu durumda IndexError: list index out of range hatasını önlemek için kontrol ediyoruz.
        if not chunk.choices:
            continue
        
        # chunk.choices[0].delta: Bu adımdaki değişiklik nesnesi
        delta = chunk.choices[0].delta

        # delta.content: Yeni üretilen token (string veya None)
        if delta.content:
            token_count += 1
            yield delta.content

    logger.debug(f"🔤 Toplam {token_count} token üretildi.")

    # ══════════════════════════════════════════════════════════
    # ADIM 4: DİPNOT — KAYNAK DOSYA ADLARI (CITATIONS)
    # ══════════════════════════════════════════════════════════
    # Yanıtın sonuna, bilginin hangi dosyalardan çekildiğini
    # gösteren dipnot ekliyoruz. Bu, kullanıcının yanıtı
    # doğrulayabilmesi için kritik önem taşır.
    #
    # dict.fromkeys() ile benzersiz dosya adlarını sıra
    # koruyarak (set'ten farklı olarak) elde ediyoruz.
    # ══════════════════════════════════════════════════════════

    if context_chunks:
        # Benzersiz dosya adlarını sıra koruyarak al
        unique_files = list(dict.fromkeys(
            chunk["file_name"] for chunk in context_chunks
        ))

        # Dipnot metnini oluştur
        citations = (
            "\n\n---\n"
            "📚 **Kaynaklar:** " +
            ", ".join(f"`{fname}`" for fname in unique_files)
        )

        # Dipnotu son token olarak yield et
        yield citations

    logger.info("💬 Yanıt üretimi tamamlandı.")


async def generate_document_summary(
    document_text: str,
    file_name: str,
    client,
    model_name: str = "phi-3.5-mini",
    max_chars: int = 3000,
) -> str:
    """
    Yüklenen bir belgenin yapay zekâ destekli özetini üretir.
    
    Belge metni çok uzunsa, ilk max_chars karakteri alınarak
    özetleme yapılır. Bu fonksiyon streaming kullanmaz,
    tek seferde yanıt döndürür.

    Parametreler:
        document_text : Belgenin ham metin içeriği
        file_name     : Belge dosya adı (özette referans için)
        client        : AsyncOpenAI istemcisi
        model_name    : Sohbet modelinin adı
        max_chars     : İşlenecek maksimum karakter sayısı

    Döndürür:
        str: Belgenin Türkçe özeti (2-4 cümle)
    """
    logger.info(f"📄 Özet üretimi başlatılıyor: {file_name}")
    
    try:
        # 1. Metni kırp (gerekirse)
        truncated_text = document_text[:max_chars]
        
        # 2. Özet prompt'unu oluştur
        system_prompt = "Sen bir belge özetleme asistanısın. Verilen metnin kısa ve öz bir Türkçe özetini yaz. Özet 2-4 cümle olmalıdır."
        user_prompt = f"Aşağıdaki belgenin ({file_name}) özetini çıkar:\n\n{truncated_text}"
        
        # 3. Modelden yanıt al (streaming olmadan)
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            temperature=0.3,
            max_tokens=256
        )
        
        # 4. İçeriği döndür
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"❌ Özet oluşturulamadı ({file_name}): {e}")
        return f"'{file_name}' dosyasının özeti oluşturulamadı."


# ────────────────────────────────────────────────────────
# TÜRKÇE ALIAS FONKSİYONLAR
# ────────────────────────────────────────────────────────
foundry_sohbet_istemcisi_olustur = create_foundry_chat_client
akisli_yanit_uret = generate_streaming_response
belge_ozeti_uret = generate_document_summary
