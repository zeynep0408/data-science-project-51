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
    return [
        {
            "baslik": "Kayıt",
            "metin": "ÖğrenHub platformuna kayıt olmak için ana sayfadaki üye ol butonunu kullanabilirsiniz. Kayıt işlemi sırasında e-posta adresiniz ve şifreniz gereklidir."
        },
        {
            "baslik": "Ödeme ve İade",
            "metin": "Kurs ödemelerinizi kredi kartı veya banka kartı ile güvenle yapabilirsiniz. Satın alım sonrasındaki ilk 14 gün içinde koşulsuz iade hakkınız bulunmaktadır."
        },
        {
            "baslik": "Sertifika",
            "metin": "Eğitimleri başarıyla tamamladığınızda resmi bir sertifika almaya hak kazanırsınız. Sertifika alabilmek için bitirme sınavından yeterli puanı almanız gerekir."
        },
        {
            "baslik": "Teknik Destek",
            "metin": "Platform üzerindeki video oynatma sorunları veya giriş hataları için teknik destek ekibimizle her zaman iletişime geçebilirsiniz."
        },
        {
            "baslik": "Canlı Dersler",
            "metin": "Canlı dersler belirlenen takvime göre eğitmenler eşliğinde gerçekleşir. Canlı ders kayıtları daha sonra izlemeniz için sistemde saklanır."
        }
    ]
pass


def parcala(metin, boyut=120, overlap=20):
    return [metin[i:i + boyut] for i in range(0, len(metin), boyut - overlap)]
pass


def chunkla_belgeler(belgeler, boyut=120, overlap=20):
    return [
        {"metin": parca, "kaynak": doc["baslik"]}
        for doc in belgeler
        for parca in parcala(doc["metin"], boyut, overlap)
    ]
pass


def benzerlik(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0
pass


def indeksle(chunks, embed):
    return [
        {**chunk, "vektor": embed(chunk["metin"])}
        for chunk in chunks
    ]
pass


def getir(soru, index, embed, k=3):
    soru_vektor = embed(soru)
    return [
        item for _, item in sorted(
            ((benzerlik(soru_vektor, item["vektor"]), item) for item in index),
            key=lambda x: x[0],
            reverse=True
        )[:k]
    ]
pass

def sor(soru, index, embed, llm, k=3):
    ilgili_chunklar = getir(soru, index, embed, k=k)
    baglam = "\n".join([chunk["metin"] for chunk in ilgili_chunklar])
    prompt = f"Bağlam:\n{baglam}\n\nSoru: {soru}\nCevap:"
    cevap = llm(prompt)
    
    kaynaklar = sorted(list(set(chunk["kaynak"] for chunk in ilgili_chunklar)))
    return cevap, kaynaklar
pass


def sohbet(soru, gecmis, index, embed, llm, k=3):
    ilgili_chunklar = getir(soru, index, embed, k=k)
    baglam = "\n".join([chunk["metin"] for chunk in ilgili_chunklar])
    
    prompt_parcalari = []
    if gecmis:
        prompt_parcalari.append("Önceki konuşma:\n" + "\n".join(gecmis))
        prompt_parcalari.append("")
        
    prompt_parcalari.append(f"Bağlam:\n{baglam}\n\nSoru: {soru}\nCevap:")
    prompt = "\n".join(prompt_parcalari)
    
    cevap = llm(prompt)
    yeni_gecmis = gecmis + [f"Kullanıcı: {soru}", f"Asistan: {cevap}"]
    
    return cevap, yeni_gecmis
pass


def sadik_mi(cevap, baglam, hakem):
    
    prompt = f"Bağlam: {baglam}\nİddia: {cevap}\nİddia bağlamla"
    puan_str = hakem(prompt)
    
    try:
        puan = int(str(puan_str).strip())
    except (ValueError, AttributeError):
        puan = 0
        
    return puan >= 4
pass