## Project
openvoicecs-bench: voice-agent'ların (sesli müşteri hizmetleri botları)
çok-turlu tool-call davranışını otomatik puanlayan bir benchmark/eval sistemi.
220 senaryo, her biri oracle (doğru cevap anahtarı) içeriyor.

## What we're doing
check_factual_grounding grader'ını iyileştiriyoruz (openvoicecs.py:2721).
Sorun: eski sistem sadece literal kelime/regex eşleşmesi yapıyordu, eşanlamlı
ifadeleri (örn. "no fee" vs "fee waiver") ve dürüst hata raporlarını
("couldn't complete, escalated") yanlış cezalandırıyordu. Bu, repo'nun kendi
known-limitations.md §7 ve CONTRIBUTING.md dosyasında "en değerli katkı" diye
işaretlenmiş.

## Design decisions (already made, don't relitigate without asking)
- Cascade/hibrit mantık: önce literal regex çalışır (default davranış,
  değişmedi). Sadece literal miss olursa semantic LLM fallback devreye girer.
- Bu hibrit mantık DEFAULT (açık) davranış. Eski saf-literal davranış
  OPENVOICECS_GROUNDING_MODE=legacy env var ile erişilebilir kalıyor,
  karşılaştırma/debug amaçlı, default değil.
- Oracle agent literal terimleri birebir kullanıyor -> semantic fallback'e
  hiç düşmüyor -> score --agent oracle API key'siz, offline çalışmaya devam
  ediyor. Bu bir regresyon testiyle kilitli, BOZULMAMALI.
- forbidden_claims tarafında _forbidden_claim_near_miss filtresi var (yanlış
  pozitif tetiklemeyi engellemek için).
- Tek trace'teki eksik claim'ler tek batch LLM çağrısında soruluyor
  (claim başına ayrı çağrı DEĞİL).
- Judge çıktısı yapılandırılmış JSON ({"grounded"/"violated", "reason"}),
  serbest metin değil.
- temperature=0, model pinned (default openai/gpt-4o-mini), 
  OPENVOICECS_GROUNDING_JUDGE env ile override edilebilir.
- Judge çağrısı hata verirse classify_trial_error ile "infrastructure" 
  sınıflanır, skor sessizce sıfırlanmaz, trial dışlanır.

## Constraints
- Windows dev ortamı. CRLF ve path-separator ile ilgili sorunlar zaten
  çözüldü/not edildi, tekrar gündeme getirme.
- Minimum bloat: gereksiz soyutlama, kullanılmayan parametre, 
  over-engineering YASAK. Basit çözüm yetiyorsa onu kullan.
- pytest tests/unit yeşil kalmalı. Bilinen istisna: release_bundle/
  release_verification testlerindeki Windows path-separator hatası -
  bu bizim işimizle ilgisiz, dokunma, göz ardı et.
- ruff temiz kalmalı.
- Scope dışına ASLA çıkma: sadece factual_grounding grader'ı geliştiriyoruz.
  Başka bug/iyileştirme fark edersen kod değiştirme, docs/known-limitations.md'ye
  not olarak düş, bana sor.

## Git/GitHub conventions
- Commit mesajları İNGİLİZCE, imperative present tense, kısa özet satırı +
  gerekiyorsa 1-3 satır açıklama. Örnek: "Add semantic fallback for 
  factual_grounding claim matching"
- Her anlamlı değişiklikten sonra otomatik commit+push yap, onay bekleme
  (permission mode zaten buna izin veriyor).
- PR açıklamaları repo'nun kendi üslubuna uysun: dürüst, ölçülü, 
  abartısız, sınırlamaları da söyleyen (known-limitations.md tarzı).
- Ana upstream repoya (0daycloud) asla doğrudan yazma, sadece fork'ta çalış,
  PR ile öner.
- Feature başına ayrı branch/PR.
