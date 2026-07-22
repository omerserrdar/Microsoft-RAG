"""
app.py — Project Antigravity: Kapsamlı Streamlit Sohbet Arayüzü
================================================================
Özellikler:
    1. 🧠 Çok turlu sohbet hafızası (Multi-turn Memory)
    2. 📄 Otomatik belge özeti (Auto Document Summary)
    3. 👍👎 Thumbs up/down geri bildirim (Feedback)
    4. 🏷️ Belge bazlı filtreleme (Document Filter)
    5. 🌙 Profesyonel tema ve CSS (Premium Theme)
    6. 🔀 Hibrit arama (Hybrid Search)
    7. 💾 Sohbet geçmişi kaydetme/yükleme (Chat Persistence)
    8. 📋 PDF rapor çıktısı (PDF Export)

Çalıştırma:
    python -m streamlit run app.py
"""

import streamlit as st
import asyncio
import queue
import os
import json
import uuid
import tempfile
import time
from threading import Thread
from pathlib import Path
from datetime import datetime

# ── Proje İçi İmportlar ──
from core.database import DocumentDB
from core.ingester import create_foundry_embedding_client, ingest_file, read_file
from core.retriever import retrieve_relevant_chunks, hybrid_retrieve
from core.generator import (
    create_foundry_chat_client,
    generate_streaming_response,
    generate_document_summary,
)

# ============================================================
# BÖLÜM 1: ASYNC-SYNC KÖPRÜSÜ (AsyncRunner)
# ============================================================
# Streamlit senkron çalışır, core modüllerimiz asenkron.
# Arka planda kalıcı bir event loop çalıştırarak köprü kuruyoruz.
# ============================================================

class AsyncRunner:
    """Asenkron coroutine ve generator'ları Streamlit'in senkron yapısına bağlar."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro):
        """Async coroutine'i senkron çağırır."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stream(self, async_generator):
        """Async generator'ı Streamlit'in okuyabileceği sync generator'a çevirir."""
        q = queue.Queue()

        async def _consume():
            try:
                async for item in async_generator:
                    q.put(item)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        asyncio.run_coroutine_threadsafe(_consume(), self.loop)

        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item


# ============================================================
# BÖLÜM 2: SAYFA YAPILANDIRMASI VE PREMİUM TEMA
# ============================================================

st.set_page_config(
    page_title="Project Antigravity — Yerel RAG Asistan",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS Tema ──
st.markdown("""
<style>
    /* ===== GENEL TEMA ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ===== BAŞLIK BANNER ===== */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(100, 100, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero-banner h1 {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6ec7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #a0a0c0;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* ===== SIDEBAR STİLLERİ ===== */
    .sidebar-section-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #7b2ff7;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(123, 47, 247, 0.3);
    }

    /* ===== İSTATİSTİK KARTLARI ===== */
    .stat-row {
        display: flex;
        gap: 0.8rem;
        margin: 0.8rem 0;
    }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(123,47,247,0.15), rgba(0,210,255,0.1));
        border: 1px solid rgba(123, 47, 247, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.2rem;
    }

    /* ===== BELGE ÖZETİ KUTUSU (ANA PANEL) ===== */
    .doc-info-card {
        background: linear-gradient(135deg, rgba(123,47,247,0.1), rgba(0,210,255,0.05));
        border: 1px solid rgba(123, 47, 247, 0.25);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin: 0.8rem 0 1.2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .doc-info-card .doc-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e0ff;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .doc-info-card .doc-meta {
        display: flex;
        gap: 1.2rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }
    .doc-info-card .doc-meta-item {
        background: rgba(123, 47, 247, 0.15);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #b0b0d0;
    }
    .doc-info-card .doc-summary {
        background: rgba(0, 0, 0, 0.2);
        border-left: 3px solid #7b2ff7;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #c8c8e0;
        line-height: 1.5;
    }

    /* ===== GERİ BİLDİRİM BUTONLARI ===== */
    .feedback-container {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
        opacity: 0.7;
        transition: opacity 0.2s;
    }
    .feedback-container:hover {
        opacity: 1;
    }

    /* ===== FOOTER ===== */
    .footer-text {
        text-align: center;
        color: #555;
        font-size: 0.7rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# BÖLÜM 3: MODEL SEÇİMİ VE SİSTEM BAŞLATMA
# ============================================================

# ── Yan panelin en üstünde model seçimi ──
st.sidebar.markdown('<p class="sidebar-section-title">🤖 Model Ayarları</p>', unsafe_allow_html=True)
selected_model = st.sidebar.selectbox(
    "Sohbet LLM Modeli",
    options=["phi-3.5-mini", "qwen2.5-1.5b", "qwen2.5-0.5b"],
    index=0,
    help="Küçük modeller CPU'da daha hızlı çalışır."
)

# ── Arama modu seçimi ──
search_mode = st.sidebar.radio(
    "Arama Yöntemi",
    options=["Semantik", "Hibrit (Semantik + Anahtar Kelime)"],
    index=0,
    help="Hibrit mod, isim/tarih gibi anahtar kelime aramalarında daha başarılıdır."
)

# ── Sistem bileşenlerini cache'le ──
@st.cache_resource
def get_system_components(chat_model_name: str):
    """Tüm asenkron bileşenleri bir kez başlatıp önbelleğe alır."""
    runner = AsyncRunner()
    db = DocumentDB()
    runner.run(db.initialize())
    emb_client, emb_model = runner.run(create_foundry_embedding_client("qwen3-embedding-0.6b"))
    chat_client, chat_model = runner.run(create_foundry_chat_client(chat_model_name))
    return runner, db, emb_client, emb_model, chat_client, chat_model

with st.spinner(f"🔌 Yerel yapay zekâ modeli yükleniyor ({selected_model})..."):
    runner, db, emb_client, emb_model, chat_client, chat_model = get_system_components(selected_model)

# ============================================================
# BÖLÜM 4: HERO BANNER
# ============================================================

st.markdown("""
<div class="hero-banner">
    <h1>🚀 Project Antigravity</h1>
    <p>Foundry Local SDK & SQLite tabanlı · Tamamen yerel · Gizlilik odaklı · İnternet gerektirmez</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# BÖLÜM 5: OTURUM (SESSION) YÖNETİMİ
# ============================================================

# Her tarayıcı sekmesi için benzersiz bir session ID oluştur
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Yüklenen belge özetleri (ana panelde gösterilecek)
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = {}

# ============================================================
# BÖLÜM 6: SIDEBAR — BELGE YÖNETİMİ
# ============================================================

st.sidebar.markdown('<p class="sidebar-section-title">📂 Belge Yönetimi</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "PDF, TXT veya MD belgesi yükleyin",
    type=["pdf", "txt", "md"],
    help="Dosyalarınız yerel olarak işlenir, hiçbir veri dışarı gönderilmez.",
)

if uploaded_file is not None:
    exists = runner.run(db.file_exists(uploaded_file.name))
    if exists:
        st.sidebar.warning(f"⚠️ `{uploaded_file.name}` zaten yüklü.")
    else:
        with st.sidebar.spinner("📄 Belge işleniyor..."):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                # Geçici dosyayı gerçek isimle yeniden adlandır
                real_temp_path = tmp_path.parent / uploaded_file.name
                if tmp_path.exists():
                    os.replace(tmp_path, real_temp_path)

                # Ingestion pipeline'ını çalıştır
                result = runner.run(ingest_file(
                    file_path=real_temp_path, db=db,
                    embedding_client=emb_client, embedding_model=emb_model,
                    chunk_size=1000, chunk_overlap=100,
                ))
                st.sidebar.success(f"✅ `{uploaded_file.name}` — {result['chunks']} parça")

                # ── Belge Özetini Üret ve Session State'e Kaydet ──
                # Özet artık ana panelde gösterilecek (sidebar'da değil)
                raw_text, _ = runner.run(read_file(real_temp_path))
                summary = runner.run(generate_document_summary(
                    raw_text, uploaded_file.name, chat_client, chat_model
                ))
                st.session_state.doc_summaries[uploaded_file.name] = {
                    "summary": summary,
                    "chunks": result['chunks'],
                    "size_kb": round(len(uploaded_file.getvalue()) / 1024, 1),
                    "type": suffix.upper().replace(".", ""),
                    "time": datetime.now().strftime("%H:%M"),
                }

                # Temizlik
                if real_temp_path.exists():
                    os.unlink(real_temp_path)
            except Exception as e:
                st.sidebar.error(f"❌ Hata: {e}")
                for p in [tmp_path, real_temp_path]:
                    if p.exists():
                        os.unlink(p)

# ============================================================
# BÖLÜM 7: SIDEBAR — VERİTABANI İSTATİSTİKLERİ VE GRAFİKLER
# ============================================================

st.sidebar.markdown('<p class="sidebar-section-title">📊 Veritabanı Durumu</p>', unsafe_allow_html=True)

doc_count = runner.run(db.get_document_count())
stats = runner.run(db.get_document_stats())
feedback_stats = runner.run(db.get_feedback_stats())

# İstatistik kartları
st.sidebar.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-value">{doc_count}</div>
        <div class="stat-label">Toplam Parça</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{len(stats)}</div>
        <div class="stat-label">Belge Sayısı</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">👍{feedback_stats['positive']} 👎{feedback_stats['negative']}</div>
        <div class="stat-label">Geri Bildirim</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Belge dağılım grafiği
if stats:
    import pandas as pd
    df_stats = pd.DataFrame(stats).rename(columns={"file_name": "Belge", "chunk_count": "Parça"})
    st.sidebar.bar_chart(df_stats.set_index("Belge"))

    for idx, item in enumerate(stats, 1):
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"{idx}. `{item['file_name']}` ({item['chunk_count']})")
        if col2.button("🗑️", key=f"del_{item['file_name']}"):
            runner.run(db.delete_by_file(item['file_name']))
            st.rerun()
else:
    st.sidebar.info("Henüz belge yüklenmemiş.")

# ============================================================
# BÖLÜM 8: SIDEBAR — BELGE FİLTRESİ
# ============================================================

st.sidebar.markdown('<p class="sidebar-section-title">🏷️ Arama Filtresi</p>', unsafe_allow_html=True)

file_names = runner.run(db.get_file_names())
selected_files = st.sidebar.multiselect(
    "Sadece seçili belgelerde ara",
    options=file_names,
    default=[],
    help="Boş bırakırsanız tüm belgelerde aranır."
)
file_filter = selected_files if selected_files else None

# ============================================================
# BÖLÜM 9: SIDEBAR — SOHBET GEÇMİŞİ YÖNETİMİ
# ============================================================

st.sidebar.markdown('<p class="sidebar-section-title">💾 Sohbet Geçmişi</p>', unsafe_allow_html=True)

sessions = runner.run(db.get_chat_sessions())

if st.sidebar.button("🆕 Yeni Sohbet Başlat", use_container_width=True):
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.rerun()

if sessions:
    for sess in sessions[:5]:
        col1, col2 = st.sidebar.columns([4, 1])
        label = f"💬 {sess['session_id']} ({sess['msg_count']} msj)"
        if col1.button(label, key=f"load_{sess['session_id']}", use_container_width=True):
            # Eski oturumun mesajlarını yükle
            old_messages = runner.run(db.get_session_messages(sess['session_id']))
            st.session_state.session_id = sess['session_id']
            st.session_state.messages = []
            for msg in old_messages:
                entry = {"role": msg["role"], "content": msg["content"]}
                if msg.get("chunks_json"):
                    try:
                        entry["chunks"] = json.loads(msg["chunks_json"])
                    except Exception:
                        pass
                st.session_state.messages.append(entry)
            st.rerun()
        if col2.button("🗑️", key=f"delsess_{sess['session_id']}"):
            runner.run(db.delete_session(sess['session_id']))
            st.rerun()

# ============================================================
# BÖLÜM 10: SIDEBAR — PDF RAPOR ÇIKTISI
# ============================================================

st.sidebar.markdown('<p class="sidebar-section-title">📋 Rapor</p>', unsafe_allow_html=True)

if st.session_state.messages:
    # Sohbet geçmişini Markdown formatında dışa aktar
    report_lines = [
        "# 🚀 Project Antigravity — Sohbet Raporu\n",
        f"**Oturum:** {st.session_state.session_id}\n",
        f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"**Model:** {chat_model}\n",
        "---\n",
    ]
    for msg in st.session_state.messages:
        role = "👤 Kullanıcı" if msg["role"] == "user" else "🤖 Asistan"
        report_lines.append(f"\n### {role}\n\n{msg['content']}\n")

    report_text = "\n".join(report_lines)

    st.sidebar.download_button(
        label="📥 Sohbet Raporunu İndir (.md)",
        data=report_text,
        file_name=f"antigravity_rapor_{st.session_state.session_id}.md",
        mime="text/markdown",
        use_container_width=True,
    )
else:
    st.sidebar.caption("Sohbet başladığında rapor indirme aktifleşir.")

# ── Footer ──
st.sidebar.markdown(
    '<p class="footer-text">🚀 Project Antigravity v2.0<br>'
    'Tamamen yerel · Gizlilik odaklı · İnternet gerektirmez</p>',
    unsafe_allow_html=True,
)

# ============================================================
# BÖLÜM 11: ANA PANEL — BELGE BİLGİ KARTLARI
# ============================================================
# Yüklenen belgelerin AI özetleri ana panelde gösterilir
# Sidebar'da alan dar olduğu için zengin bilgi kartları burada sunulur

if st.session_state.doc_summaries:
    with st.expander(f"📚 Yüklenen Belgeler ({len(st.session_state.doc_summaries)} belge)", expanded=True):
        for fname, info in st.session_state.doc_summaries.items():
            st.markdown(f"""
<div class="doc-info-card">
    <div class="doc-title">📄 {fname}</div>
    <div class="doc-meta">
        <span class="doc-meta-item">📦 {info['chunks']} parça</span>
        <span class="doc-meta-item">💾 {info['size_kb']} KB</span>
        <span class="doc-meta-item">📋 {info['type']}</span>
        <span class="doc-meta-item">🕐 {info['time']}</span>
    </div>
    <div class="doc-summary">🧠 <b>AI Özet:</b> {info['summary']}</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# BÖLÜM 12: ANA SOHBET EKRANI — MESAJ GEÇMİŞİ
# ============================================================

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # ── Asistan mesajlarının altı: Analiz grafiği + Geri bildirim ──
        if msg["role"] == "assistant":
            # Geri bildirim butonları (thumbs up / thumbs down)
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 18])
            with fb_col1:
                if st.button("👍", key=f"fb_up_{idx}"):
                    runner.run(db.save_feedback(st.session_state.session_id, idx, 1))
                    st.toast("✅ Olumlu geri bildirim kaydedildi!", icon="👍")
            with fb_col2:
                if st.button("👎", key=f"fb_down_{idx}"):
                    runner.run(db.save_feedback(st.session_state.session_id, idx, -1))
                    st.toast("📝 Olumsuz geri bildirim kaydedildi.", icon="👎")

            # Vektör benzerlik analiz grafiği (varsa)
            if "chunks" in msg and msg["chunks"]:
                with st.expander("🔍 Vektör Arama Analizi & Kaynak Metinler"):
                    import pandas as pd
                    chart_data = {
                        "Parça": [f"#{i}" for i in range(1, len(msg["chunks"]) + 1)],
                        "Benzerlik": [c["score"] for c in msg["chunks"]],
                    }
                    df_chart = pd.DataFrame(chart_data)
                    st.bar_chart(df_chart.set_index("Parça"))

                    for i, c in enumerate(msg["chunks"], 1):
                        st.markdown(f"**📄 #{i} | `{c['file_name']}` | Skor: `{c['score']:.4f}`**")
                        st.info(c["chunk_content"][:500])

# ============================================================
# BÖLÜM 12: SORU GİRİŞİ VE YANITLAMA
# ============================================================

if prompt := st.chat_input("Yüklediğiniz belgeler hakkında soru sorun..."):
    # 1. Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Kullanıcı mesajını veritabanına kaydet (sohbet geçmişi)
    runner.run(db.save_message(st.session_state.session_id, "user", prompt))

    # 2. Asistan yanıtı
    with st.chat_message("assistant"):
        # ── Arama Aşaması ──
        with st.spinner("🔎 Belgeler aranıyor..."):
            start_time = time.time()

            if search_mode == "Hibrit (Semantik + Anahtar Kelime)":
                # Hibrit arama (semantic + keyword)
                chunks = runner.run(hybrid_retrieve(
                    question=prompt, db=db,
                    embedding_client=emb_client, embedding_model=emb_model,
                    file_filter=file_filter, top_k=3,
                ))
            else:
                # Sadece semantik arama
                chunks = runner.run(retrieve_relevant_chunks(
                    question=prompt, db=db,
                    embedding_client=emb_client, embedding_model=emb_model,
                    file_filter=file_filter, top_k=3,
                ))

            search_time = time.time() - start_time

        # ── Yanıt Üretimi (Streaming) ──
        # Çok turlu hafıza: önceki mesajları chat_history olarak gönder
        chat_history_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.spinner("🧠 Yapay zekâ yanıt üretiyor..."):
            async_gen = generate_streaming_response(
                question=prompt,
                context_chunks=chunks,
                client=chat_client,
                model_name=chat_model,
                chat_history=chat_history_for_llm,
            )

            # İlk token gelene kadar spinner dönsün (UX iyileştirmesi)
            sync_gen = runner.stream(async_gen)
            first_token = ""
            try:
                first_token = next(sync_gen)
            except StopIteration:
                pass

        # İlk token ve kalanları birleştirerek akıt
        def token_generator():
            if first_token:
                yield first_token
            for token in sync_gen:
                yield token

        response = st.write_stream(token_generator())

        # Yanıtı session state'e ve veritabanına kaydet
        chunks_json = json.dumps(chunks, ensure_ascii=False) if chunks else None
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "chunks": chunks,
        })
        runner.run(db.save_message(
            st.session_state.session_id, "assistant", response, chunks_json
        ))

        # Sayfayı yenile → geri bildirim butonları ve grafikler görünsün
        st.rerun()
