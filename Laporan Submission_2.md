# Laporan Proyek Machine Learning - Laetisha Haryanto

## Project Overview

**Latar Belakang**: Dalam era digital saat ini, masyarakat dihadapkan dengan informasi berlimpah, termasuk dalam industri game yang sangat kompetitif. Jumlah game yang terus bertambah membuat pengguna kesulitan memilih game yang sesuai dengan preferensi mereka. Seiring meningkatnya jumlah judul game yang tersedia di berbagai platform digital seperti Steam, PlayStation Store, dan Xbox Marketplace, pengguna sering kali kesulitan memilih game yang sesuai dengan preferensi mereka. Studi menunjukkan bahwa lebih dari 60% pengguna mengandalkan sistem rekomendasi untuk menemukan game baru yang menarik[^1]. 

**Pentingnya proyek diselesaikan**: Tanpa sistem rekomendasi yang baik, pengguna akan mengalami information overload dan kesulitan dalam menemukan game yang relevan. Ini berdampak pada pengalaman pengguna secara keseluruhan dan mengurangi engagement. Pendekatan sistem rekomendasi berbasis data menjadi solusi penting, dengan menerapkan dua teknik populer dalam domain ini, yaitu Collaborative Filtering dan Content-Based Filtering, sistem dapat memberikan saran yang lebih akurat dan personal. Rekomendasi personal tidak hanya meningkatkan kepuasan pengguna, tetapi juga berdampak pada loyalitas dan retensi pengguna pada platform digital[^2].

**Referensi**:  
[^1]: Park, Y. J., & Chang, H. J. (2009). "Individual and Group Behavior-Based Customer Profile Model for Personalized Product Recommendation." Expert Systems with Applications, 36(2), 1932–1939. 
[^2]: Jannach, D., & Adomavicius, G. (2016). Recommendation systems: Challenges, insights and research opportunities. ACM Computing Surveys, 49(4), 1–34.

## Business Understanding

### Problem Statements

- Pengguna mengalami kesulitan dalam menemukan game yang relevan karena banyaknya pilihan yang tersedia tanpa bantuan sistem rekomendasi
- Sistem rekomendasi tunggal dengan pendekatan tertentu sering kali menghasilkan rekomendasi yang kurang akurat jika tidak menggabungkan aspek perilaku pengguna dan karakteristik konten game.

### Goals

- Mengembangkan model content-based filtering yang menggunakan informasi genre untuk merekomendasikan game dengan konten serupa yang disukai pengguna.
- Mengembangkan model collaborative filtering untuk merekomendasikan game berdasarkan pola rating pengguna lain yang serupa.

### Solution Approach
- Content-Based Filtering: Menggunakan pendekatan TF-IDF dan cosine similarity untuk menghitung kemiripan antar game berdasarkan genre, dan merekomendasikan game yang paling mirip dengan yang pernah disukai pengguna.
- Collaborative Filtering: Menggunakan algoritma SVD (Singular Value Decomposition) dari pustaka Surprise untuk mempelajari interaksi user–item dan menghasilkan rekomendasi berdasarkan pola rating serupa antar pengguna.

## Data Understanding
Sumber Data: [Kaggle Dataset](https://www.kaggle.com/datasets/jahnavipaliwal/video-game-reviews-and-ratings/data)

### Variabel-variabel pada Video Games Kaggle Dataset adalah sebagai berikut:
- Terdapat 18 Variabel dan 47774 baris untuk dataset game, namun untuk proyek sistem rekomendasi ini hanya menggunakan 3 variabel sebagai berikut:
    - `Game Title` : Nama game 
    - `Genre` : Kategori game (misal: Action, Puzzle)
    - `User Rating` : Penilaian dari pengguna (sebelum penyekalaan: 10-50)
      
### Kondisi Data Awal
- - Missing Values di deteksi dengan `df.isnull().sum()` dan diperoleh bahwa pada dataset tidak terdapat Missing Values
- Data duplikat di deteksi dengan `df.duplicated().sum())` dan diperoleh bahwa tidak terdapat Data Duplikat

### Exploratory Data Analysis and Visualization:
- ![Wordcloud Genre](https://drive.google.com/uc?export=view&id=1Qg-O-iH0IP28kGJjsDR5CIHXm8VnfdtC)  
Berdasarkan plot, 3 Genre Game terbanyak yaitu **Strategy**, **Shooter**, dan **RPG**
- ![Distribusi Variabel Rating](https://drive.google.com/uc?export=view&id=1qRKInLADStH0HsZ2ckFwZ8siwXtNhy9u)
Berdasarkan plot, terdapat lebih dari 3000 game yang memiliki rating sekitar 30. Kemudian rentang ratingnya dari 10-50 sedikit janggal, maka dibutuhkan penyekalaan ulang menjadi 1-5


## Data Preparation
Langkah-langkah Data Preparation:
1. Melakukan penyekalaan variabel `User Rating` dari 10-50 menjadi 1-5 agar lebih umum
2. Menyiapkan data untuk Content-Based Filtering dengan melakukan:
   - Mengambil variabel `Game Title` dan `Genre` untuk menjadi dataframe `cbf_df`
   - Melakukan vektorisasi untuk variabel `Genre` dengan mengubahnya menjadi bentuk matriks, tahap ini dilakukan agar variabel kategorikal dapat terbaca mesin
4. Menyiapkan data untuk Collaborative Filtering dengan melakukan:
   - Menggunakan index sebagai `userID` karena tidak terdapat ID pengguna dari data awal
   - Mengambil variabel `userID`, `Game Title`, `User Rating`, dan `Genre` untuk menjadi `cf_df`
   - Mengubah dataframe menjadi objek dataset `Surprise`
   - Melakukan split dataset menjadi data latih sebesar 80% dan data uji 20%
     
## Modeling
Model yang Digunakan:
1. Content-Based Filtering:
   - Pendekatan ini memberikan rekomendasi berdasarkan kemiripan konten game (dalam proyek ini genre), menggunakan cosine similarity. Jika seorang pengguna menyukai game dengan genre tertentu, sistem akan merekomendasikan game lain dengan genre yang mirip.
   - Kelebihan:
     - Bekerja dengan baik untuk item baru yang belum memiliki interaksi pengguna.
     - Dapat memberikan rekomendasi yang dapat dijelaskan (misalnya berdasarkan genre).
   - Kekurangan:
     - Rekomendasi terbatas pada konten yang mirip, sehingga cenderung kurang bervariasi.
     - Tidak menangkap pola kolaboratif antar pengguna.
2. Collaborative Filtering:
   -  Pendekatan ini menggunakan algoritma Singular Value Decomposition (SVD) dari pustaka Surprise. Model dilatih menggunakan data rating pengguna terhadap game. Sistem ini mempelajari pola dari pengguna yang memberikan rating serupa, lalu merekomendasikan item yang disukai oleh pengguna dengan preferensi serupa.
   -  Kelebihan:
     -  Dapat menangkap hubungan kompleks antar pengguna dan item.
     -  Tidak memerlukan metadata dari item seperti genre atau deskripsi.
   -  Kekurangan:
     - Tidak bekerja baik untuk pengguna atau item baru (cold-start problem).
     - Bergantung pada jumlah interaksi yang cukup untuk belajar pola.


## Evaluation
Metrik Evaluasi yang digunakan:
- MAE (Mean Absolute Error): Mengukur rata-rata kesalahan absolut antara nilai aktual dan nilai prediksi. Metrik ini memberikan gambaran langsung seberapa jauh prediksi model dari data sebenarnya secara rata-rata, tanpa mempertimbangkan arah kesalahan (positif atau negatif). MAE mudah dipahami dan tahan terhadap outlier kecil, tetapi tidak memberikan penalti lebih besar untuk kesalahan besar.
- MSE (Mean Squared Error): Mengukur rata-rata dari kuadrat selisih antara nilai aktual dan prediksi. Karena selisihnya dikuadratkan, MSE memberikan penalti lebih besar terhadap kesalahan besar. MSE sangat berguna jika kita ingin meminimalkan kesalahan besar dalam model prediksi.
- RMSE (Root Mean Squared Error): Akar dari MSE dan memiliki satuan yang sama dengan target (traffic volume), sehingga lebih mudah diinterpretasikan dalam konteks dunia nyata. RMSE juga memberikan penalti lebih besar untuk kesalahan prediksi yang besar, dan sering dipakai untuk membandingkan performa model secara umum dalam kasus regresi.

| Model        | RMSE (Train) | MAE (Train) | RMSE (Test) | MAE (Test) |
|--------------|--------------|-------------|-------------|------------|
| **XGBoost**  | **244.18**   | **155.74**  | **294.26**  | **174.27** |
| LightGBM     | 255.45       | 164.98      | 306.81      | 182.42     |
| SARIMAX      | 461.18       | 286.88      | 714.49      | 651.99     |

| Model       | MSE (Train) | MSE (Test) |
| ----------- | ----------- | ---------- |
| **XGBoost** | **59.62**   | **86.59**  |
| LightGBM    | 65.26       | 94.13      |
| SARIMAX     | 212.69      | 510.49     |

**Interpretasi**:

- XGBoost memberikan hasil prediksi terbaik dengan error terendah untuk seluruh metrik (RMSE, MAE, dan MSE).
- SARIMAX digunakan sebagai baseline, namun performanya jauh tertinggal.
- Model tree-based (XGBoost dan LightGBM) mampu menangkap kompleksitas hubungan non-linier antar fitur.

**Referensi**:
