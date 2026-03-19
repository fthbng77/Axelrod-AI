# Axelrod-AI

**Bireysel kaostan simbiyotik kolektif zekaya gecisi, Oyun Teorisi ve Pekistirmeli Ogrenme ile arastiran cok-ajanli simulasyon platformu.**

Iterated Prisoner's Dilemma (Tekrarlanan Mahkum Ikilemi) ortaminda RL ajanlari egitir ve [Axelrod](https://github.com/Axelrod-Python/Axelrod) turnuvalarinda 243+ stratejiye karsi yaristirir. Hedef: Harper et al. (2017) tarafindan belirlenen performans olcutlerini asmak.

---

## Icindekiler

- [Teorik Arka Plan](#teorik-arka-plan)
- [Baz Alinan Calismalar](#baz-alinan-calismalar)
- [Algoritmalar ve Yontemler](#algoritmalar-ve-yontemler)
- [Ortam: Iterated Prisoner's Dilemma](#ortam-iterated-prisoners-dilemma)
- [Egitim Pipeline'lari](#egitim-pipelinelari)
- [Turnuva Sistemi](#turnuva-sistemi)
- [Proje Yapisi](#proje-yapisi)
- [Kurulum ve Calistirma](#kurulum-ve-calistirma)
- [Baseline Sonuclari](#baseline-sonuclari)
- [Yol Haritasi](#yol-haritasi)

---

## Teorik Arka Plan

### Mahkum Ikilemi (Prisoner's Dilemma)

Iki oyuncu es zamanli olarak **Isbirligi (C)** veya **Ihanet (D)** secimi yapar:

```
                  Oyuncu B
                  C           D
Oyuncu A   C    (R=3, R=3)   (S=0, T=5)
           D    (T=5, S=0)   (P=1, P=1)
```

| Parametre | Deger | Anlami |
|-----------|-------|--------|
| **R** (Reward) | 3 | Karsilikli isbirligi odulu |
| **T** (Temptation) | 5 | Ihanet cazibesi (tek tarafli ihanet) |
| **S** (Sucker) | 0 | Saf odulu (isbirligi yapan taraf ihanete ugrar) |
| **P** (Punishment) | 1 | Karsilikli ihanet cezasi |

**Temel gerilim:** T > R > P > S siralamasi, bireysel rasyonalite (D sec) ile kolektif optimallik (her ikisi C secsin) arasinda catisma yaratir.

### Tekrarlanan Oyun (IPD)

Oyun **200 tur** boyunca tekrarlanir. Bu tekrar, oyunculara:
- Gecmis hareketleri **hatirma** (memory depth)
- Rakibin davranisini **modelleme**
- Uzun vadeli **stratejik iliskiler** kurma

imkani tanir. Axelrod'un 1980'lerde gosterdigi gibi, tekrarlanan oyunlarda **isbirligi** evrimsel olarak ortaya cikabilir.

---

## Baz Alinan Calismalar

### Birincil Referans: Harper et al. (2017)

> *"Reinforcement Learning Produces Dominant Strategies for the Iterated Prisoner's Dilemma"*
> PLOS ONE | [arXiv:1707.06307](https://arxiv.org/abs/1707.06307)

**Neden baz aliyoruz:**
- Ayni Axelrod kutuphanesini kullanarak ML stratejileri (ANN, FSM, HMM) egitti
- Evrimsel + PSO algoritmalariyla egitilen stratejiler, Tit-for-Tat dahil tum klasikleri yendi
- Turnuvadaki ilk 15 stratejinin neredeyse tamami ML-egitimli cikti
- Bizim hedefimiz: Bu sonuclari modern RL teknikleriyle asmak

### Ikincil Referanslar

| Calisma | Yil | Projemize Katkisi |
|---------|-----|-------------------|
| **Foerster et al. — LOLA** | 2018 | Rakip-farkinda ogrenme algoritmasi (LOLA ajanimizin temeli) |
| **"Self-Play Q-learners Collude in IPD"** | 2023 | Q-Learning ajanimizin teorik temeli: epsilon-greedy Q-ogrenme isbirligini kesfeder |
| **DeepMind — Alpha-Rank** | 2019 | Nash dengesi yerine evrimsel degerlendirme — turnuva analizimiz icin |
| **Nature Comm. — MARL for dominant strategies** | 2025 | En guncel MARL framework; evrimsel turnuva yaklasimi |
| **Knight et al. — Axelrod Library** | 2016 | 243+ strateji, turnuva altyapisi (kutuphanemiz) |

### Axelrod Kutuphanesindeki Harper 2017 Stratejileri

Bizim yenmemiz gereken hedef stratejiler:

| Strateji Ailesi | Varyantlar | Yontem |
|----------------|-----------|--------|
| **EvolvedANN** | ANN, ANN5, ANN5Noise05 | Evrimsel olarak egitilmis yapay sinir agi |
| **EvolvedFSM** | FSM4, FSM6, FSM16, FSM16Noise05 | Evrimsel sonlu durum makinesi |
| **EvolvedHMM** | HMM5 | Gizli Markov Modeli |
| **EvolvedLookerUp** | 1_1_1, 2_2_2 | Tablo tabanli arama stratejisi |
| **PSOGambler** | 1_1_1, 2_2_2, 2_2_2Noise05, Mem1 | Parcacik Surusu Optimizasyonu |
| **EvolvedAttention** | - | Dikkat mekanizmali strateji |

> **Benchmark:** Bu 15 strateji ortalama **2.72 skor/tur** ile klasik stratejilerin (2.64) ustunde performans gosterir.

---

## Algoritmalar ve Yontemler

### 1. Tabular Q-Learning

**Dosya:** `src/agents/q_learning.py`
**Referans:** Watkins & Dayan (1992) + "Self-Play Q-learners" (2023)

**Teori:**
Durum-aksiyon cifti icin Q-degeri tablosu tutar. Her adimda Bellman denklemini yaklasik cozer:

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

| Hiperparametre | Deger | Aciklama |
|----------------|-------|----------|
| `learning_rate` (alpha) | 0.15 | Ogrenme hizi |
| `discount_factor` (gamma) | 0.95 | Gelecek odul iskonto faktoru |
| `epsilon` | 0.3 -> 0.01 | Epsilon-greedy kesif orani |
| `epsilon_decay` | 0.9999 | Her adimda epsilon azalimi |
| `memory_depth` | 3 | Son kac tur hatirlanir |

**Neden bu yontem:**
- 2023 calismasi kanatladi ki epsilon-greedy Q-Learning ajanlari, dogru gamma ve epsilon ayariyla dogal olarak **isbirligi** kesfeder
- Basit ama etkili; tablo tabanli yapisi sayesinde ogrenilen politika tamamen yorumlanabilir
- Harper'in PSO Gambler stratejilerine karsilik gelir (ikisi de tablo-tabanli karar verir)

**Durum Kodlamasi:**
```
State = [kendi_son_3_aksiyonu, rakip_son_3_aksiyonu]
       = [C/D, C/D, C/D, C/D, C/D, C/D]  (6 boyutlu)
Cooperate=0, Defect=1, Gecmis_yok=-1
```

---

### 2. Deep Q-Network (DQN)

**Dosya:** `src/agents/deep_q.py`
**Referans:** Mnih et al. (2015) + Harper'in EvolvedANN yaklasimi

**Teori:**
Q-tablosu yerine sinir agi ile Q-degeri yaklasimi yapar. Iki kritik yenilik:

1. **Experience Replay:** Gecmis deneyimleri buffer'da depolar, rastgele ornekleyerek korelasyonu kirar
2. **Target Network:** Ayri bir hedef ag kullanarak egitim kararliligi saglar

```
Loss = MSE(Q_main(s, a), r + gamma * max_a' Q_target(s', a'))
Target ag her 100 adimda guncellenir
```

**Ag Mimarisi:**
```
Girdi (6) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Cikti (2)
                                                         [Q(C), Q(D)]
```

| Hiperparametre | Deger | Aciklama |
|----------------|-------|----------|
| `hidden_dims` | [128, 64] | Gizli katman boyutlari |
| `learning_rate` | 5e-4 | Adam optimizer ogrenme hizi |
| `discount_factor` | 0.95 | Gelecek odul iskontosu |
| `epsilon` | 1.0 -> 0.01 | Baslangicta tam kesif, kademeli azalim |
| `epsilon_decay` | 0.999 | Epsilon carpani |
| `batch_size` | 64 | Replay buffer'dan orneklem boyutu |
| `buffer_size` | 10000 | Experience replay kapasitesi |
| `target_update_freq` | 100 | Hedef ag guncelleme sikligi |

**Neden bu yontem:**
- Harper'in EvolvedANN stratejilerinin **modern karsiligi**
- Harper evrimsel algoritma ile ag agirliklarini optimize etti; biz gradient-based DQN ile egitiyoruz
- Experience replay + target network ile daha kararli ogrenme
- Surekli durum uzayini tablo olmadan isleyebilir

---

### 3. Policy Gradient (REINFORCE)

**Dosya:** `src/agents/policy_gradient.py`
**Referans:** Williams (1992) — REINFORCE algoritmasi

**Teori:**
Q-degeri yerine dogrudan **politika** optimize eder. Ag, durumdan aksiyon **olasiliklarina** esler:

```
pi(a|s) = softmax(network(s))
```

Guncelleme kurali (policy gradient teoremi):
```
nabla J = E[nabla log pi(a|s) * G_t]

G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...  (iskontolu toplam odul)
```

**Varyans azaltma teknikleri:**
- **Baseline:** Getiriler normalize edilir (ortalama cikarilir, std'ye bolunur)
- **Entropi regularizasyonu:** Politikanin erken yakinsamamasi icin entropi terimi eklenir

```
Loss = -log(pi(a|s)) * G_normalized - entropy_coef * H(pi)
```

| Hiperparametre | Deger | Aciklama |
|----------------|-------|----------|
| `hidden_dims` | [128, 64] | Politika agi boyutlari |
| `learning_rate` | 1e-3 | Adam optimizer |
| `discount_factor` | 0.95 | Getiri iskontosu |
| `entropy_coef` | 0.01 | Kesif tesvik katsayisi |
| Gradient clipping | norm=1.0 | Gradyan patlamasini onler |

**Neden bu yontem:**
- Stokastik politikalar ogrenir — saf C veya D yerine **olasiliksal** stratejiler
- Entropi terimi sayesinde **cok erken bir stratejiye kilitlenmeyi** onler
- DQN'den farkli olarak, dogrudan politika uzayinda calisir (daha puruzsu optimizasyon)

---

### 4. LOLA (Learning with Opponent-Learning Awareness)

**Dosya:** `src/agents/lola.py`
**Referans:** Foerster et al. (2018) — [arXiv:1709.04326](https://arxiv.org/abs/1709.04326)

**Teori:**
Diger uc yontem rakibi **sabit** varsayar. LOLA'nin temel yeniligi: rakibin de **ogrenen bir ajan** oldugunu hesaba katar.

**Standart PG yaklasimi:**
```
theta_1 <- theta_1 + alpha * nabla_1 V_1(theta_1, theta_2)
  "Rakip sabit, ben optimize ediyorum"
```

**LOLA yaklasimi:**
```
theta_1 <- theta_1 + alpha * [nabla_1 V_1 + lola_lr * nabla_1 (nabla_2 V_2)^T * nabla_2 V_1]
  "Rakip de ogrenecek, onun ogrenmesinin bana etkisini hesaba katiyorum"
```

**Uclu guncelleme sureci:**

1. **Rakip modeli guncelle:** Rakibin aksiyonlariyla egitilen ayri bir sinir agi
2. **Standart politika gradyani:** Kendi aksiyonlarimiz icin normal PG
3. **LOLA duzeltme terimi:** Rakibin gelecek gradyan adiminin bizim odullerimize etkisi

**Mimari:**
```
Kendi Politikamiz:   Girdi(6) -> Dense(128) -> ReLU -> Dense(128) -> ReLU -> Softmax(2)
Rakip Modeli:        Girdi(6) -> Dense(128) -> ReLU -> Dense(128) -> ReLU -> Softmax(2)
(Ikisi de turevlenebilir — rakip modeli uzerinden gradyan akisi saglanir)
```

| Hiperparametre | Deger | Aciklama |
|----------------|-------|----------|
| `hidden_dim` | 128 | Her iki ag icin gizli katman |
| `learning_rate` | 1e-3 | Kendi politikamizin ogrenme hizi |
| `opponent_lr` | 1e-3 | Rakip modelinin ogrenme hizi |
| `lola_lr` | 0.3 | LOLA duzeltme teriminin agirligi |
| `discount_factor` | 0.95 | Getiri iskontosu |

**Neden bu yontem:**
- IPD'de isbirliginin **ortaya cikisini** saglayan en etkili yontem
- Foerster et al. gosterdi ki LOLA ajanlari self-play'de **Tit-for-Tat benzeri** davranis gelistirir
- Diger yontemler rakibi sabit varsayarak "ihanet yarisi"na girer; LOLA bunu kirar
- Projemizin **en yenilikci** bileseni — Harper 2017'de bu yaklaism yoktu

---

### Yontem Karsilastirma Tablosu

| Ozellik | Q-Learning | DQN | Policy Gradient | LOLA |
|---------|-----------|-----|----------------|------|
| **Politika tipi** | Tablo | Sinir agi | Sinir agi | Sinir agi + Rakip modeli |
| **Ogrenme hedefi** | Q-degeri | Q-degeri | Dogrudan politika | Politika + rakip etkisi |
| **Rakip varsayimi** | Sabit | Sabit | Sabit | Ogrenen ajan |
| **Kesif** | Epsilon-greedy | Epsilon-greedy | Entropi terimi | Stokastik politika |
| **Guncelleme** | Her adim | Her adim (batch) | Bolum sonu | Bolum sonu |
| **Bellek** | Q-tablosu | Replay buffer | Bolum buffer'i | Bolum buffer'i + rakip gecmisi |
| **Harper karsiligi** | PSO Gambler | EvolvedANN | — | — (yeni) |
| **Avantaj** | Yorumlanabilir | Genellestirme | Puruzsuz optimizasyon | Isbirligi kesfeder |
| **Dezavantaj** | Olceklenmez | Replay gerektirir | Yuksek varyans | Hesaplama maliyeti |

---

## Ortam: Iterated Prisoner's Dilemma

**Dosya:** `src/environments/ipd.py`

### Durum Temsili

Her oyuncu icin durum vektoru, son `memory_depth` turun aksiyonlarini kodlar:

```
State = [kendi_aksiyonlari(t-3, t-2, t-1), rakip_aksiyonlari(t-3, t-2, t-1)]
        Cooperate = 0.0, Defect = 1.0, Gecmis yok = -1.0
```

Ornek (memory_depth=3, 2. turdan sonra):
```
Tur 1: Ben=C, Rakip=D
Tur 2: Ben=D, Rakip=C
State = [-1, 0, 1, -1, 1, 0]
         ^        ^
     gecmis yok  gecmis yok
```

### Gurultu (Noise)

Aksiyonlar `noise` olasiligiyla ters cevriliblir (C->D veya D->C). Bu:
- Gercek dunya belirsizligini simule eder
- Stratejilerin **hataya dayanakliligi**ni test eder
- Harper'in "Noise05" varyantlarinin karsiligidir

### Konfigurasyonlar

```yaml
environment:
  memory_depth: 3      # Son kac tur hatirlanir
  num_rounds: 200      # Bir macdaki tur sayisi
  noise: 0.0           # Aksiyon cevrilme olasiligi (0.0 - 1.0)
payoff:
  R: 3  T: 5  S: 0  P: 1
```

---

## Egitim Pipeline'lari

### Pipeline 1: Self-Play (Kendine Karsi Oynama)

**Dosya:** `src/training/self_play.py` — `SelfPlayTrainer`

Iki ajan birbirine karsi 200 turluk maclar oynar. Her bolum sonunda her iki ajan da guncellenir.

```
Bolum dongusu:
  1. Ortam sifirla
  2. 200 tur boyunca:
     a. Ajan 1 aksiyonunu sec (kendi perspektifinden state)
     b. Ajan 2 aksiyonunu sec (ayna state)
     c. Ortam adim at -> odulleri al
     d. Her iki ajani guncelle
     e. LOLA icin: rakip aksiyonlarini kaydet
  3. Istatistikleri logla (skor, isbirligi orani)
```

### Pipeline 2: Populasyon Egitimi

**Dosya:** `src/training/self_play.py` — `PopulationTrainer`

Tek bir RL ajanini Axelrod kutuphanesindeki **tum stratejilere** karsi egitir. Bu, Harper 2017'nin evrimsel yaklasimiyla ayni secilim baskisini olusturur:

```
Her nesil:
  Her rakip strateji icin:
    N bolum oyna
    RL ajanini guncelle
  Nesil istatistiklerini logla
```

### Pipeline 3: Egitim + Degerlendirme

**Dosya:** `src/training/train_and_evaluate.py`

```
Faz 1: Self-Play Egitimi
  -> Q-Learning vs Q-Learning (500-5000 bolum)
  -> DQN vs DQN
  -> Policy Gradient vs Policy Gradient
  -> LOLA vs LOLA

Faz 2: Turnuva Degerlendirmesi
  -> Egitilmis ajanlar Axelrod turnuvasina girer
  -> Klasik stratejiler (TFT, Cooperator, Defector, vb.)
  -> Harper 2017 stratejileri (EvolvedANN, FSM, PSO, vb.)
  -> Siralama ve performans karsilastirmasi
```

---

## Turnuva Sistemi

**Dosya:** `src/tournament/axelrod_bridge.py`

### RL -> Axelrod Koprusu

`RLPlayer` sinifi, egitilmis RL ajanlarini Axelrod turnuva formatina uyumlu hale getirir:

```
Axelrod gecmis (History) -> Durum vektoru (numpy) -> RL ajan karari -> Axelrod aksiyonu
```

Bu kopru sayesinde RL ajanlarimiz, Axelrod kutuphanesindeki **243 strateji** ile ayni turnuvada yarisabilir.

### Turnuva Formati

- **Oyuncular:** RL ajanlari + 10 klasik + 15 Harper ML stratejisi
- **Maclar:** Her oyuncu cifti karsitasir (round-robin)
- **Tur:** 200 tur/mac
- **Tekrar:** 5 tekrar (stokastik varyans icin)
- **Skor:** Ortalama skor/tur ile siralama

---

## Proje Yapisi

```
Axelrod-AI/
├── configs/
│   └── default.yaml              # Tum hiperparametreler
├── src/
│   ├── agents/
│   │   ├── base.py               # Soyut temel sinif (BaseAgent)
│   │   ├── q_learning.py         # Tabular Q-Learning
│   │   ├── deep_q.py             # DQN + Replay + Target Net
│   │   ├── policy_gradient.py    # REINFORCE + Baseline + Entropi
│   │   └── lola.py               # LOLA (rakip-farkinda ogrenme)
│   ├── environments/
│   │   └── ipd.py                # IPD ortami (gurultu, bellek, odul matrisi)
│   ├── training/
│   │   ├── self_play.py          # Self-play + populasyon egitimi
│   │   └── train_and_evaluate.py # Ana pipeline (egit + turnuva)
│   ├── tournament/
│   │   └── axelrod_bridge.py     # RL <-> Axelrod entegrasyonu
│   └── analysis/                 # Gorsellestirme (gelistirilecek)
├── tests/
├── requirements.txt
└── README.md
```

---

## Kurulum ve Calistirma

### Gereksinimler

```
Python >= 3.10
axelrod >= 4.14.0    # Turnuva altyapisi (243+ strateji)
numpy >= 1.26.0      # Sayisal hesaplama
torch >= 2.0.0       # Sinir agi (DQN, PG, LOLA)
matplotlib >= 3.8.0  # Gorsellestirme
pyyaml >= 6.0        # Konfigurasyon
```

### Kurulum

```bash
git clone https://github.com/<kullanici>/Axelrod-AI.git
cd Axelrod-AI
pip install -r requirements.txt
```

### Calistirma

```bash
# Tam egitim + turnuva (5000 bolum)
python src/training/train_and_evaluate.py --episodes 5000

# Hizli test (500 bolum)
python src/training/train_and_evaluate.py --episodes 500

# Sadece turnuva (egitim atlayarak)
python src/training/train_and_evaluate.py --skip-training
```

---

## Baseline Sonuclari

### Harper 2017 ML vs Klasik Stratejiler (25 oyuncu turnuvasi)

```
ML Ortalama:      2.72 skor/tur
Klasik Ortalama:  2.64 skor/tur
ML Avantaji:      +0.08
```

### Ilk Egitim Sonuclari (500 bolum, erken asama)

| Ajan | Self-Play Isbirligi Orani | Not |
|------|--------------------------|-----|
| Q-Learning | %43-50 | Kismi isbirligi kesfetti |
| DQN | %2 | Henuz yetersiz egitim |
| Policy Gradient | %6 | Henuz yetersiz egitim |
| LOLA | %1-2 | Daha uzun egitim gerekli |

> **Not:** 500 bolum erken bir asama. Harper 2017 binlerce nesil boyunca evrimsel arama yapti. 5000+ bolumle sonuclarin dramatik olarak iyilesmesi beklenir.

---

## Yol Haritasi

### Faz 1: Temel Altyapi (Tamamlandi)
- [x] IPD ortami (gurultu, konfigurasyon destegi)
- [x] 4 RL ajani (Q-Learning, DQN, Policy Gradient, LOLA)
- [x] Self-play egitim pipeline'i
- [x] Axelrod turnuva entegrasyonu
- [x] Baseline turnuva sonuclari

### Faz 2: Egitim Optimizasyonu
- [ ] Populasyon egitimi (243 stratejiye karsi)
- [ ] Hiperparametre optimizasyonu (Optuna)
- [ ] Curriculum learning (kolay rakiplerden zor rakiplere)
- [ ] Gurultulu ortam varyantlari (noise=0.05)

### Faz 3: Ileri Teknikler
- [ ] Evrimsel strateji secimi (en iyi ajanlarin yeni nesillere aktarimi)
- [ ] Derin bellek (memory_depth=5+, LSTM/Transformer)
- [ ] Coklu oyun destegi (Chicken Game, Stag Hunt)
- [ ] Alpha-Rank degerlendirme entegrasyonu

### Faz 4: Ekolojik Simulasyonlar
- [ ] Populasyon dinamikleri (Moran process)
- [ ] Stratejilerin evrimsel kararlilik analizi
- [ ] Isbirliginin ortaya cikis kosullarinin haritalanmasi

---

## Referanslar

1. **Axelrod, R.** (1984). *The Evolution of Cooperation*. Basic Books.
2. **Harper, M. et al.** (2017). "Reinforcement learning produces dominant strategies for the iterated prisoner's dilemma." *PLOS ONE*. [arXiv:1707.06307](https://arxiv.org/abs/1707.06307)
3. **Foerster, J. et al.** (2018). "Learning with Opponent-Learning Awareness." *AAMAS*. [arXiv:1709.04326](https://arxiv.org/abs/1709.04326)
4. **Knight, V. et al.** (2016). "An open framework for the reproducible study of the iterated prisoner's dilemma." [arXiv:1604.00896](https://arxiv.org/abs/1604.00896)
5. **Mnih, V. et al.** (2015). "Human-level control through deep reinforcement learning." *Nature*.
6. **Williams, R. J.** (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*.
7. **Lanctot, M. et al.** (2019). "OpenSpiel: A Framework for Reinforcement Learning in Games." [arXiv:1908.09453](https://arxiv.org/abs/1908.09453)
8. *"Self-Play Q-learners Can Provably Collude in the Iterated Prisoner's Dilemma"* (2023). [arXiv:2312.08484](https://arxiv.org/abs/2312.08484)
9. *"A multi-agent reinforcement learning framework for exploring dominant strategies"* (2025). *Nature Communications*.
