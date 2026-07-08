"""
DS-51 — Uçtan Uca Asistan (çok-turlu + sadakat)

Senaryo: Kurgusal online kurs platformu **ÖğrenHub** için bir destek asistanı
kuruyorsun. DS-49'daki temel RAG hattının (parçala → embed → getir → kaynaklı
cevap) TAMAMINI kurup üzerine iki ileri özellik ekliyorsun:

  • ÇOK-TURLU SOHBET  → asistan önceki soru-cevapları hatırlar (konuşma hafızası).
  • SADAKAT KONTROLÜ  → üretilen cevap gerçekten bağlama dayanıyor mu? Bir HAKEM
    (judge LLM) cevabı 1-5 arası puanlar; düşük puan = uydurma (hallucination).

Derste LangChain memory + faithfulness metrikleri bunları senin için yapıyordu;
burada içlerinde ne olduğunu elle kurup anlıyorsun.

⚠️ `embed`, `llm` ve `hakem` DIŞARIDAN parametre gelir (dependency injection):
  - Gerçek kullanımda kendi modelini verirsin (README'ye bak).
  - Testlerde deterministik sahte fonksiyonlar verilir → API key GEREKMEZ.
  `embed`: metin (str) → sayı listesi (vektör).
  `llm`  : prompt (str) → cevap (str).
  `hakem`: prompt (str) → "1".."5" (str) — sadakat puanı.

Her fonksiyonun `pass` kısmını doldur. Test: `python watch.py` veya
`pytest tests/test_question.py -v`
"""
import math


def load_belgeler():
    """
    Koda gömülü ~5 belge döndür. Her belge bir dict:
      {"baslik": str, "metin": str}
    Başlıklar ÖğrenHub konuları olsun:
      "Kayıt", "Ödeme ve İade", "Sertifika", "Teknik Destek", "Canlı Dersler".
    Metinler konu kelimeleriyle örtüşsün (ör. "Sertifika" metninde "sertifika",
    "sınav" geçsin) — böylece getir doğru belgeyi bulabilir.

    Returns:
        list[dict]
    """
    pass


def parcala(metin, boyut=120, overlap=20):
    """
    Metni karakter bazlı, ÜST ÜSTE BİNEN (overlap) parçalara böl.
    Her adımda `boyut` karakter al, sonra `boyut - overlap` kadar ilerle.

    Args:
        metin: str
        boyut: int — parça uzunluğu (karakter)
        overlap: int — komşu parçaların paylaştığı karakter
    Returns:
        list[str]
    """
    pass


def chunkla_belgeler(belgeler, boyut=120, overlap=20):
    """
    Her belgeyi `parcala` ile böl; her parçaya belgenin başlığını KAYNAK olarak ekle.
    Döndürülen her eleman: {"metin": str, "kaynak": str}

    Args:
        belgeler: list[dict] — load_belgeler() çıktısı
    Returns:
        list[dict]
    """
    pass


def benzerlik(v1, v2):
    """
    İki vektör arasındaki COSINE benzerliğini hesapla:
        dot(v1, v2) / (norm(v1) * norm(v2))
    Sıfır vektörde 0.0 döndür (bölme hatası olmasın).

    Returns:
        float
    """
    pass


def indeksle(chunks, embed):
    """
    Her chunk'ı embed et; chunk'a "vektor" alanı ekleyip döndür.
    Her eleman: {"metin": ..., "kaynak": ..., "vektor": embed(metin)}

    Args:
        chunks: list[dict] — chunkla_belgeler çıktısı
        embed: callable — str → list[float]
    Returns:
        list[dict]
    """
    pass


def getir(soru, index, embed, k=3):
    """
    Soruyu embed et, index'teki her chunk ile COSINE benzerliğini hesapla,
    en yüksek benzerlikli ilk `k` chunk'ı döndür (vektörleriyle birlikte).

    Args:
        soru: str
        index: list[dict] — indeksle çıktısı
        embed: callable
        k: int
    Returns:
        list[dict] — en benzer k chunk
    """
    pass


def sor(soru, index, embed, llm, k=3):
    """
    Tam RAG cevabı (tek-turlu):
      1. getir(soru, index, embed, k) → ilgili chunk'lar
      2. bağlam = chunk metinlerini birleştir
      3. cevap = llm("Bağlam:\n{baglam}\n\nSoru: {soru}\nCevap:")
      4. kaynaklar = getirilen chunk'ların benzersiz "kaynak"ları (sıralı)
    (cevap, kaynaklar) ikilisini döndür.

    Returns:
        tuple(str, list[str])
    """
    pass


def sohbet(soru, gecmis, index, embed, llm, k=3):
    """
    ÇOK-TURLU sohbet (konuşma hafızalı RAG):
      - `gecmis`: önceki satırların listesi, ör.
            ["Kullanıcı: kayıt nasıl olur?", "Asistan: e-posta ile ..."]
      - getir(soru, index, embed, k) → ilgili chunk'lar → bağlam
      - Prompt'u kur:
          * gecmis DOLUYSA en başa "Önceki konuşma:\n" + satırları ekle (\n ile birleşik),
            sonra bir boş satır bırak.
          * ardından "Bağlam:\n{baglam}\n\nSoru: {soru}\nCevap:" ekle.
      - cevap = llm(prompt)
      - yeni_gecmis = gecmis + [f"Kullanıcı: {soru}", f"Asistan: {cevap}"]
    (cevap, yeni_gecmis) ikilisini döndür.

    Returns:
        tuple(str, list[str])
    """
    pass


def sadik_mi(cevap, baglam, hakem):
    """
    SADAKAT (faithfulness) kontrolü:
      Cevabın bağlamla desteklenip desteklenmediğini `hakem`'e sorup puanlat.
      - hakem: callable, prompt (str) → "1".."5" (str)
      - Bir prompt kur (bağlam + cevap/iddia + "1-5 puanla"), hakem'i çağır.
      - Dönen puanı int'e çevir; puan >= 4 ise True, değilse False döndür.

    Returns:
        bool
    """
    pass
