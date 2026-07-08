import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_belgeler, parcala, chunkla_belgeler, benzerlik,
    indeksle, getir, sor, sohbet, sadik_mi,
)


# ──────────────────────────────────────────────────────
# MOCK embed + llm + hakem — deterministik, key gerektirmez.
# Gerçek projede: embed = sentence-transformers/OpenAI, llm/hakem = OpenAI (README).
# fake_embed: anahtar-kelime tabanlı sayım vektörü (anlamca yakın metin → yakın vektör).
# ──────────────────────────────────────────────────────
VOCAB = ["kayıt", "ücret", "ödeme", "iade", "sertifika",
         "sınav", "teknik", "şifre", "canlı", "ders", "zoom"]


def fake_embed(metin):
    s = metin.lower()
    return [float(s.count(w)) for w in VOCAB]


def fake_llm(prompt):
    # Gerçekte akıcı cevap üretir; testte bağlamın ilk kısmını döndürür.
    baglam = prompt.split("Bağlam:")[1].split("Soru:")[0].strip()
    cevap = "Cevap: " + baglam[:40]
    # Çok-turlu ipucu: geçmiş prompt'a girdiyse cevabı işaretle.
    if "Önceki konuşma:" in prompt:
        cevap = "[bağlamlı] " + cevap
    return cevap


def fake_hakem(prompt):
    # Sadakat hakemi: İddia'daki anlamlı kelimeler Bağlam'da geçiyorsa yüksek puan.
    baglam = prompt.split("Bağlam:")[1].split("İddia:")[0].strip().lower()
    iddia = prompt.split("İddia:")[1].split("İddia bağlamla")[0].strip().lower()
    kelimeler = [w for w in iddia.split() if len(w) > 3]
    if not kelimeler:
        return "1"
    ortak = sum(1 for w in kelimeler if w in baglam)
    return "5" if ortak / len(kelimeler) >= 0.5 else "1"


@pytest.fixture(scope="module")
def belgeler():
    return load_belgeler()


@pytest.fixture(scope="module")
def index(belgeler):
    return indeksle(chunkla_belgeler(belgeler, 120, 20), fake_embed)


# 0. Belgeler
def test_load_belgeler(belgeler):
    assert isinstance(belgeler, list) and len(belgeler) >= 4
    assert all(set(b.keys()) >= {"baslik", "metin"} for b in belgeler)
    basliklar = {b["baslik"] for b in belgeler}
    assert len(basliklar) == len(belgeler)  # başlıklar benzersiz


# 1. chunkla_belgeler — kaynak ekli
def test_chunkla_belgeler(belgeler):
    chunks = chunkla_belgeler(belgeler, 120, 20)
    assert len(chunks) >= len(belgeler)
    assert all(set(c.keys()) == {"metin", "kaynak"} for c in chunks)
    assert {c["kaynak"] for c in chunks} == {b["baslik"] for b in belgeler}


# 2. benzerlik — cosine
def test_benzerlik():
    assert abs(benzerlik([1, 0], [1, 0]) - 1.0) < 1e-6      # aynı
    assert abs(benzerlik([1, 0], [0, 1]) - 0.0) < 1e-6      # dik
    assert benzerlik([0, 0], [1, 1]) == 0.0                 # sıfır vektör
    assert benzerlik([1, 1], [2, 2]) > 0.99                 # aynı yön


# 3. indeksle — vektör ekli
def test_indeksle(index):
    assert all("vektor" in c for c in index)
    assert all(len(c["vektor"]) == len(VOCAB) for c in index)


# 4. getir — doğru chunk (anlamsal), iki konu
def test_getir_odeme(index):
    top = getir("iade ve ödeme ücret geri alınır mı?", index, fake_embed, k=1)
    assert top[0]["kaynak"] == "Ödeme ve İade"


def test_getir_sertifika(index):
    top = getir("sertifika almak için sınav gerekir mi?", index, fake_embed, k=1)
    assert top[0]["kaynak"] == "Sertifika"


def test_getir_k(index):
    assert len(getir("kayıt", index, fake_embed, k=3)) == 3


# 5. sor — cevap + kaynak
def test_sor_kaynak(index):
    cevap, kaynaklar = sor("iade nasıl olur?", index, fake_embed, fake_llm, k=2)
    assert isinstance(cevap, str) and len(cevap) > 0
    assert "Ödeme ve İade" in kaynaklar
    assert kaynaklar == sorted(set(kaynaklar))   # sıralı + benzersiz


# 6. sohbet — çok-turlu (konuşma hafızası)
def test_sohbet_gecmis_uzar(index):
    # Boş geçmişle başlayınca yeni_gecmis 2 eleman uzar (kullanıcı + asistan).
    cevap, yeni = sohbet("iade nasıl olur?", [], index, fake_embed, fake_llm, k=2)
    assert len(yeni) == 2
    assert yeni[0].startswith("Kullanıcı:")
    assert yeni[1].startswith("Asistan:")


def test_sohbet_gecmis_prompta_girer(index):
    # Geçmiş DOLU verilince fake_llm cevabı "[bağlamlı]" ile başlar
    # → geçmiş prompt'a gerçekten eklenmiş demektir.
    gecmis = ["Kullanıcı: sertifika var mı?", "Asistan: Evet, sertifika verilir."]
    cevap, yeni = sohbet("peki sınav zor mu?", gecmis, index, fake_embed, fake_llm, k=2)
    assert cevap.startswith("[bağlamlı] ")
    assert len(yeni) == 4   # önceki 2 + yeni 2


# 7. sadik_mi — sadakat (faithfulness) kontrolü
def test_sadik_mi():
    baglam = "Sertifika almak için bitirme sınavını geçmen gerekir. Sınav online yapılır."
    sadik_cevap = "Sertifika için bitirme sınavını geçmen gerekir."
    uydurma_cevap = "Sertifika ücretsiz kargoyla evine gönderilir."
    assert sadik_mi(sadik_cevap, baglam, fake_hakem) is True
    assert sadik_mi(uydurma_cevap, baglam, fake_hakem) is False


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────
import requests


def _send_score(user_score):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return
    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return
    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {"user_id": USER_ID, "project_id": PROJECT_ID, "user_score": user_score, "is_auto": True}
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
