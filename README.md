# DS-51 — Uçtan Uca Asistan (çok-turlu + sadakat)

**Modül**: RAG ve Vektör Veritabanları (ML-08) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Kurgusal online kurs platformu **ÖğrenHub** için bir **destek asistanı** kuruyorsun. Kullanıcılar doğal dille soruyor: *"Kayıt nasıl olur?"*, *"İade alabilir miyim?"*, *"Sertifika için sınav şart mı?"*

DS-49'da temel RAG hattını (parçala → embed → getir → kaynaklı cevap) sıfırdan kurmuştun. Burada o hattın **tamamını** kurup üzerine gerçek bir asistanı ürüne dönüştüren **iki ileri özellik** ekliyorsun:

- 🧵 **Çok-turlu sohbet** — asistan önceki soru-cevapları **hatırlar**. Kullanıcı *"peki ya iade?"* dediğinde konuşma bağlamını korur (konuşma hafızası).
- 🛡️ **Sadakat kontrolü (faithfulness)** — üretilen cevap gerçekten **bağlama dayanıyor mu**, yoksa model uydurdu mu? Bir **hakem** (judge LLM) cevabı 1-5 arası puanlar; düşük puan = güvenilmez cevap.

Derste LangChain memory + RAGAS faithfulness bunları senin için yapıyordu. Bu projede aynı mekanizmayı **elle** kurup içinde ne olduğunu tam anlıyorsun.

## 🧩 Kuracağın Hat

1. `load_belgeler` / `chunkla_belgeler` — ÖğrenHub belgelerini üst üste binen (overlap) parçalara böl, her parçaya kaynak ekle
2. `benzerlik` — iki vektör arası **cosine** (elle)
3. `indeksle` — her parçayı embed et
4. `getir` — soruya en yakın top-k parçayı cosine ile bul
5. `sor` — getir → bağlamlı prompt → LLM → **cevap + kaynak** (tek-turlu)
6. `sohbet` — **çok-turlu**: geçmişi prompt'a ekle → cevap üret → güncellenmiş geçmişi döndür
7. `sadik_mi` — **hakem** cevabı bağlamla puanlar; puan ≥ 4 ise cevap sadıktır

## 🔌 embed, llm ve hakem Nasıl Çalışıyor? (önemli)

Fonksiyonların `embed`, `llm` ve `hakem`'i **parametre olarak alır** (dependency injection):

- **`embed`**: metin (str) → sayı listesi (vektör). Gerçekte `sentence-transformers` ya da OpenAI embedding.
- **`llm`**: prompt (str) → cevap (str). Gerçekte OpenAI.
- **`hakem`**: prompt (str) → `"1"`..`"5"` (str). Sadakat puanı üreten ikinci bir LLM (LLM-as-a-judge).
- **Testlerde** deterministik sahte `embed`/`llm`/`hakem` verilir → **API key GEREKMEZ**, puan hep aynı.

Aynı kod, sadece verdiğin fonksiyonlar değişir — bu, test edilebilir RAG kodu yazmanın profesyonel yoludur.

## 📦 Kurulum

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python watch.py                 # otomatik test (dosya değişince)
# veya
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

`USER_ID`'ni Kaizu profilinden alıp yaz (`PROJECT_ID`'ye dokunma). Testler geçince skor otomatik gider.

## 🔑 API key? (opsiyonel)

Puanlama için **gerekmez** (mock). Gerçek embed/LLM/hakem ile denemek istersen `cp .env.example .env` + anahtarını koy (`.env` asla commit edilmez).
