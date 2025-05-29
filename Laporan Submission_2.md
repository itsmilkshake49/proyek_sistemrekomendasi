# Laporan Proyek Machine Learning - Laetisha Haryanto

## Project Overview

**Latar Belakang**: Dalam era digital saat ini, masyarakat dihadapkan dengan informasi berlimpah, termasuk dalam industri game yang sangat kompetitif. Seiring meningkatnya jumlah judul game yang tersedia di berbagai platform digital seperti Steam, PlayStation Store, dan Xbox Marketplace, pengguna sering kali kesulitan memilih game yang sesuai dengan preferensi mereka. Studi menunjukkan bahwa lebih dari 60% pengguna mengandalkan sistem rekomendasi untuk menemukan game baru yang menarik[^1]. 

**Pentingnya proyek diselesaikan**: Tanpa sistem rekomendasi yang baik, pengguna akan mengalami information overload dan kesulitan dalam menemukan game yang relevan. Ini berdampak pada pengalaman pengguna secara keseluruhan dan mengurangi engagement. Pendekatan sistem rekomendasi berbasis data menjadi solusi penting, dengan menerapkan dua teknik populer dalam domain ini, yaitu Collaborative Filtering dan Content-Based Filtering, sistem dapat memberikan saran yang lebih akurat dan personal. Rekomendasi personal tidak hanya meningkatkan kepuasan pengguna, tetapi juga berdampak pada loyalitas dan retensi pengguna pada platform digital[^2].
 
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
- Terdapat 18 Variabel dan 47774 baris untuk dataset game, berikut penjelasan setiap variabel:
    - `Game Title` : Nama game
    - `User Rating` : Penilaian numerik dari pengguna (sebelum penyekalaan: 10-50)
    - `Age Group Targeted` : Kelompok usia yang menjadi target game (misal: Adult, All Ages)
    - `Price` : Harga ritel game dalam USD saat pembelian
    - `Platform` : Tempat game tersedia (misal: Playstation, Xbox)
    - `Requires Special Device` : Apakah game membutuhkan perangkat khusus
    - `Developer` : Nama studio atau perusahaan yang mengembangkan game
    - `Publisher` : Perusahaan yang merilis atau memasarkan game
    - `Release Year` : Tahun rilis game
    - `Genre` : Kategori game (misal: Action, Puzzle)
    - `Multiplayer` : Apakah game mendukung mode multiplayer
    - `Game Length (Hours)` : Estimasi rata-rata durasi waktu bermain (dalam jam)
    - `Graphics Quality` : Penilaian kualitas grafis game berdasarkan persepsi pengguna
    - `Soundtrack Quality` : Penilaian kualitas musik dan audio game
    - `Story Quality` : Penilaian terhadap alur cerita atau narasi game
    - `User Review Text` : Ringkasan ulasan pengguna dalam bentuk teks
    - `Game Mode` : Game dimainkan secara Online atau Offline
    - `Min Number of Players` : Jumlah minimum pemain yang dibutuhkan untuk memainkan game
- Pada proyek ini, hanya menggunakan 3 variabel yaitu `Game Title`, `User Rating`, dan `Genre`
      
### Kondisi Data Awal
- Missing Values di deteksi dengan `df.isnull().sum()` dan diperoleh bahwa pada dataset tidak terdapat Missing Values
- Data duplikat di deteksi dengan `df.duplicated().sum())` dan diperoleh bahwa tidak terdapat Data Duplikat

### Exploratory Data Analysis and Visualization:
- ![wordcloud](https://github.com/user-attachments/assets/7d1eec0a-33ac-4b58-ab8d-0d4ec93aa343)
Berdasarkan plot, 3 Genre Game terbanyak yaitu **Strategy**, **Shooter**, dan **RPG**
- ![distribusi_rating](https://github.com/user-attachments/assets/397e2729-8ed7-45b5-8eeb-e604f7ac4a43)
Berdasarkan plot, terdapat lebih dari 3000 game yang memiliki rating sekitar 30. Kemudian rentang ratingnya dari 10-50 sedikit janggal, maka dibutuhkan penyekalaan ulang menjadi 1-5


## Data Preparation
Langkah-langkah Data Preparation:
1. Melakukan penyekalaan variabel `User Rating` dari 10-50 menjadi 1-5 agar lebih umum
2. Menyiapkan data untuk Content-Based Filtering dengan melakukan:
   - Menghapus duplikat berdasarkan `Game Title` dengan `cbf.drop_duplicates(subset='Game Title').reset_index(drop=True)` sehingga diperoleh 40 game unik yang digunakan untuk vektorisasi
   - Mengambil variabel `Game Title` dan `Genre` untuk menjadi dataframe `cbf_df`
   - Melakukan vektorisasi untuk variabel `Genre` dengan mengubahnya menjadi bentuk matriks, tahap ini dilakukan agar variabel kategorikal dapat terbaca mesin
3. Menyiapkan data untuk Collaborative Filtering dengan melakukan:
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

**Output Top-N Recommendation**:
1. Content-Based Filtering:
   - Berikut Top-N rekomendasi game Minecraft berdasarkan kemiripan genre dengan `game_recommendations('Minecraft')`, diperoleh hasil berikut:
     | **Rank** | **Game Title**                          | **Genre** |
     | -------- | --------------------------------------- | --------- |
     | 1        | Animal Crossing: New Horizons           | Adventure |
     | 2        | Spelunky 2                              | Adventure |
     | 3        | The Legend of Zelda: Breath of the Wild | Adventure |
     | 4        | Grand Theft Auto V                      | Adventure |
     | 5        | Cuphead                                 | Shooter   |
   - Berikut Top-N rekomendasi game Tekken 7 berdasarkan kemiripan genre dengan `game_recommendations('Tekken 7')`, diperoleh hasil berikut:
     | **Rank** | **Game Title**                   | **Genre**  |
     | -------- | -------------------------------- | ---------- |
     | 1        | Sid Meier’s Civilization VI      | Simulation |
     | 2        | Red Dead Redemption 2            | Simulation |
     | 3        | Mario Kart 8 Deluxe              | Simulation |
     | 4        | Counter-Strike: Global Offensive | Simulation |
     | 5        | The Elder Scrolls V: Skyrim      | Simulation |
2. Collaborative Filtering:
   - Berikut Top-N rekomendasi untuk pengguna 331 (`rekomendasi_user_331 = get_game_recommendations(user_id=331, df=cf_df, model=model)
print(rekomendasi_user_331`) berdasarkan rating:
     | **Rank** | **Game Title**                          | **Genre**  |
     | -------- | --------------------------------------- | ---------- |
     | 1        | 1000-Piece Puzzle                       | Sports     |
     | 2        | Fall Guys                               | Strategy   |
     | 3        | Call of Duty: Modern Warfare 2          | Strategy   |
     | 4        | The Legend of Zelda: Breath of the Wild | Adventure  |
     | 5        | Tekken 7                                | Simulation |
     | 6        | Animal Crossing: New Horizons           | Adventure  |
     | 7        | Pokémon Scarlet & Violet                | Puzzle     |
     | 8        | Portal 2                                | Strategy   |
     | 9        | Tetris                                  | Shooter    |
     | 10       | Counter-Strike: Global Offensive        | Simulation |
   - Berikut Top-N rekomendasi untuk pengguna 20 (`rekomendasi_user_20 = get_game_recommendations(user_id=20, df=cf_df, model=model)
print(rekomendasi_user_331`) berdasarkan rating:
     | **Rank** | **Game Title**                 | **Genre**  |
     | -------- | ------------------------------ | ---------- |
     | 1        | Half-Life: Alyx                | RPG        |
     | 2        | Just Dance 2024                | Strategy   |
     | 3        | 1000-Piece Puzzle              | Sports     |
     | 4        | Street Fighter V               | Fighting   |
     | 5        | Call of Duty: Modern Warfare 2 | Strategy   |
     | 6        | Tekken 7                       | Simulation |
     | 7        | Hitman 3                       | Shooter    |
     | 8        | League of Legends              | Party      |
     | 9        | Cuphead                        | Shooter    |
     | 10       | Tetris                         | Shooter    |

## Evaluation
1. Metrik Evaluasi yang digunakan untuk Content-Based Filtering:
   - **Precision** : Mengukur seberapa relevan item yang direkomendasikan dari sejumlah `k` rekomendasi teratas. Dalam konteks ini, skor cosine similarity digunakan sebagai ukuran relevansi.
     
     $$\text{Precision@k} = \frac{1}{k} \sum_{i=1}^{k} \cos(\vec{x}, \vec{y_i})$$
     
     Semakin tinggi nilai precision, semakin akurat rekomendasi terhadap preferensi konten dari game input.
   - **Recall** : Mengukur seberapa besar proporsi kemiripan yang berhasil ditangkap dari total kemiripan semua hasil. Ini menggambarkan sejauh mana rekomendasi mencakup konten yang mirip dengan game input.
     
     $$\text{Recall@k} = \frac{ \sum_{i=1}^{k} \cos(\vec{x}, \vec{y_i}) }{ \sum_{j=1}^{n} \cos(\vec{x}, \vec{y_j}) }$$
     
     Jika cosine similarity dari semua hasil sangat tersebar, maka recall membantu memahami seberapa besar bagian dari konten relevan yang berhasil direkomendasikan.
   - Berikut hasil yang diperoleh dari evaluasi model Content-Based Filtering:
     ```
     Evaluasi untuk 'Minecraft':
     Precision@5: 80.00%
     Recall@5: 100.00%

     Evaluasi untuk 'Tekken 7':
     Precision@5: 100.00%
     Recall@5: 100.00%
     ```
     Interpretasi: Model Content-Based Filtering bekerja sangat baik, terutama untuk game yang memiliki genre spesifik dan konsisten seperti Tekken 7. Untuk game yang memiliki genre lebih luas atau umum seperti Minecraft, sistem tetap memberikan hasil yang baik meskipun precision sedikit lebih rendah. Hal ini bisa disebabkan oleh adanya beberapa game dalam hasil rekomendasi yang mirip namun tidak identik secara genre. Nilai recall juga sempurna pada kedua kasus, maka **rekomendasi cukup relevan**.
2. Metrik Evaluasi yang digunakan untuk Collaborative Filtering:
   - **MAE (Mean Absolute Error)** : Mengukur rata-rata kesalahan absolut antara nilai aktual dan nilai prediksi. Metrik ini memberikan gambaran langsung seberapa jauh prediksi model dari data sebenarnya secara rata-rata.

     $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
     
   - **RMSE (Root Mean Squared Error)** : RMSE memberikan penalti lebih besar untuk kesalahan prediksi yang besar karena mengkuadratkan selisih prediksi dan aktual.
Metrik ini sangat berguna untuk membandingkan performa model secara keseluruhan.
  
     $$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$$
     
   - Berikut hasil yang diperoleh dari evaluasi model Collaborative Filtering:
     ```
     Root Mean Squared Error (RMSE): 0.7554
     Mean Absolute Error (MAE): 0.6192
     ```
     Interpretasi: Nilai RMSE dan MAE yang rendah menunjukkan bahwa model collaborative filtering memiliki performa prediksi yang cukup baik. Model dapat digunakan secara **efektif untuk memperkirakan preferensi pengguna** terhadap game yang belum mereka rating. Namun model masih bisa ditingkatkan dengan pendekatan lain seperti hybrid untuk mengatasi cold-start.
     
**Referensi**:
