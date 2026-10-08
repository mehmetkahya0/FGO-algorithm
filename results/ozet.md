# Deney Sonuçları Özeti (otomatik üretildi)

Monte Carlo deneme sayısı: **9** (ana görev + 8 farklı tohum). Değerler ortalama ± standart sapmadır.

## 1. Navigasyon doğruluğu (GPS yok)

| Yöntem | ATE RMSE [m] | Son konum hatası [m] | Maks. hata [m] | Yön RMSE [°] |
|---|---|---|---|---|
| Ölü hesap (DVL+AHRS) | 41.69 ± 17.04 | 54.36 ± 27.59 | 59.39 ± 22.22 | 3.21 ± 1.41 |
| EKF-SLAM | 2.33 ± 0.65 | 1.34 ± 0.67 | 6.38 ± 1.32 | 0.75 ± 0.07 |
| FGO (çevrimiçi) | 2.55 ± 0.71 | 1.59 ± 0.81 | 6.93 ± 1.28 | 0.81 ± 0.08 |
| FGO (düzleştirilmiş) | 1.61 ± 0.43 | 1.56 ± 0.77 | 3.88 ± 1.05 | 0.43 ± 0.07 |

## 2. FGO ablasyonu (düzleştirilmiş ATE RMSE [m])

| Yapılandırma | ATE RMSE [m] |
|---|---|
| FGO (tam model) | 1.61 ± 0.43 |
| Optik kamera yok | 1.62 ± 0.45 |
| Jiroskop sapma kestirimi yok | 1.93 ± 0.64 |
| Pusula faktörü yok | 1.95 ± 0.57 |
| DVL ölçek kalibrasyonu yok | 2.04 ± 0.51 |
| Sonar/optik nirengi (SLAM) yok | 2.07 ± 0.62 |
| Sağlam çekirdek yok (L2) | 2.21 ± 0.79 |
| Pusula kalibrasyonu yok | 3.85 ± 1.00 |
| Manyetik harita faktörü yok | 9.82 ± 3.27 |

## 3. Zor koşullar (çoklu-yol %25, yanlış alarm 0.6/tarama, 2× pusula bozulması)

| Yöntem | ATE RMSE [m] |
|---|---|
| Ölü hesap (DVL+AHRS) | 29.41 ± 20.11 |
| EKF-SLAM (χ² kapısı) | 2.53 ± 0.55 |
| FGO, sağlam çekirdek yok | 4.88 ± 2.71 |
| FGO (Huber+Cauchy) | 2.22 ± 0.78 |

## 4. Haritalama (nesne konum hatası [m])

| Kaynak | RMSE, doğrulanmış (≥3 gözlem) | Medyan, doğrulanmış | RMSE, tüm nirengiler |
|---|---|---|---|
| FGO | 2.81 ± 1.23 | 1.43 ± 0.56 | 15.21 ± 1.83 |
| EKF-SLAM | 3.62 ± 1.13 | 1.29 ± 0.34 | 15.31 ± 1.77 |
| Ölü hesap | 36.41 ± 12.55 | 36.25 ± 12.23 | 40.64 ± 12.17 |

## 5. Gizlilik ve görev

| Planlayıcı | Tespit olasılığı | Maruziyet süresi [s] | Görev süresi [s] | Yol uzunluğu [m] |
|---|---|---|---|---|
| En kısa yol | 0.86 ± 0.00 | 330.94 ± 5.24 | 1314.89 ± 9.19 | 2423.02 ± 12.91 |
| Risk-farkında (önerilen) | 0.30 ± 0.02 | 55.33 ± 15.17 | 1637.78 ± 26.44 | 2944.77 ± 40.64 |

## 6. Hedef tespiti ve sınıflandırma

Tek gözlem (sentetik test kümesi, modalite ablasyonu):

| Model | Doğruluk | Makro-F1 | Mayın duyarlılığı | Doğruluk (optik mevcut) |
|---|---|---|---|---|
| Yalnız sonar | 0.794 | 0.756 | 0.896 | 0.813 |
| Sonar + optik | 0.833 | 0.804 | 0.924 | 0.971 |
| Sonar + manyetik | 0.813 | 0.783 | 0.891 | 0.843 |
| Tam füzyon (sonar+optik+manyetik) | 0.843 | 0.819 | 0.929 | 0.978 |

Görev içi (çoklu gözlemin zamansal Bayes füzyonu):

- Nesne sınıflandırma doğruluğu: 0.92 ± 0.02
- Doğrulanmış mayın raporu kuralı: P(mayın) > 0.6 ve ≥ 2 gözlem
- Mayın kesinliği (precision): 0.97 ± 0.06
- Mayın duyarlılığı (görülen mayınlar içinde): 0.81 ± 0.20
- Görülen / toplam mayın: 6.33 ± 1.49 / 12
- Tespit edilen mayınların FGO ile konumlandırma hatası: 2.25 ± 1.39 m

## 7. Faktör grafiği (ana görev)

- Değişken sayısı: 4450, faktör sayısı: 3963, poz: 809, nirengi: 134
- Son toplu optimizasyon süresi: 0.41 s; tüm görev simülasyonu: 70.1 s
- Kalibrasyon (gerçek → FGO): pusula montaj sapması -1.50° → -1.25°, anomali-sapma katsayısı 0.0400 → 0.0358 rad/100nT, DVL ölçek -0.596% → -0.554%
- Ortalama NEES (Monte Carlo): 4.56 ± 1.75 (ideal = 2)
- 3σ tutarlılık (ana görev): hataların %58'i %95 güven elipsi içinde (ortalama NEES=6.21, ideal=2)

Toplam çalışma süresi: 874 s
