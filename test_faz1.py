import asyncio
from pathlib import Path
from core.database import DokumanVeritabani
from core.ingester import foundry_vektor_istemcisi_olustur, dosya_yukle, embedding_hesapla

async def main():
    # 1. Test için örnek bir metin dosyası oluşturalım
    test_file_path = Path("test_belgesi.txt")
    test_file_path.write_text(
        "Güneş sistemi, merkezinde Güneş ve onun çekim etkisi altında kalan sekiz gezegen, "
        "bu gezegenlerin uyduları ve cüce gezegenlerden oluşur. "
        "Dünya, Güneş'e en yakın üçüncü gezegendir ve üzerinde yaşam olduğu bilinen tek gök cismidir. "
        "Mars ise kızıl gezegen olarak bilinir ve yüzeyinde demir oksit bulunduğu için kırmızımsı bir renge sahiptir. "
        "Jüpiter, sistemdeki en büyük gezegendir.", 
        encoding="utf-8"
    )

    print("--- Test Başlıyor --- \n")

    # 2. Veritabanı ve Foundry modelini başlatalım
    async with DokumanVeritabani() as db:
        print("Model yükleniyor (ilk seferinde indirme işlemi yapabilir)...")
        # Embedding modeli ve client oluşturuluyor
        client, model_name = await foundry_vektor_istemcisi_olustur("qwen3-embedding-0.6b")
        
        # 3. Dosyayı sisteme yükleyelim (Ingestion)
        print("\nDosya okunup veritabanına yükleniyor...")
        result = await dosya_yukle(
            file_path=test_file_path,
            db=db,
            embedding_client=client,
            embedding_model=model_name,
            chunk_size=100, # Test için küçük parçalara bölelim
            chunk_overlap=20
        )
        print(f"Yükleme Sonucu: {result}\n")
        
        # 4. Soru sorup vektör araması yapalım
        soru = "Kızıl gezegen hangisidir?"
        print(f"Soru: '{soru}'")
        
        # Sorunun vektörünü hesapla
        soru_vektoru_list = await embedding_hesapla([soru], client, model_name)
        soru_vektoru = soru_vektoru_list[0]
        
        # Veritabanında (UDF kullanarak) en benzer parçaları bul
        sonuclar = await db.benzer_ara(soru_vektoru, top_k=2)
        
        print("\nBulunan En Alakalı Parçalar:")
        for idx, sonuc in enumerate(sonuclar, 1):
            print(f"{idx}. Skor: {sonuc['score']:.4f} | Dosya: {sonuc['file_name']}")
            print(f"   Metin: {sonuc['chunk_content']}\n")

    # Test bittikten sonra dosyayı temizle
    if test_file_path.exists():
        test_file_path.unlink()

if __name__ == "__main__":
    asyncio.run(main())
