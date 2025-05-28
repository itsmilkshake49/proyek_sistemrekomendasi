# Laporan Proyek Machine Learning - Laetisha Haryanto

## Domain Proyek

**Latar Belakang**: Kepadatan lalu lintas merupakan permasalahan signifikan dalam sistem transportasi kota, terutama pada jalan raya utama seperti Interstate-94 (I-94) di negara Amerika Serikat. Volume lalu lintas yang tinggi dapat menyebabkan kemacetan, meningkatkan polusi, serta memperpanjang waktu tempuh perjalanan. 

Kemacetan lalu lintas di jalan raya menimbulkan kerugian ekonomi yang mencapai miliaran dolar setiap tahun[^1]. Oleh karena itu, memprediksi volume lalu lintas secara akurat akan sangat membantu dalam pengambilan keputusan untuk manajemen lalu lintas, seperti pengaturan sinyal lalu lintas, pengelolaan jadwal konstruksi, pengoptimalisasi lampu lalu lintas, dan memberi peringatan dini kepada pengguna jalan.  

**Mengapa dan Bagaimana Masalah Diselesaikan**: Manajemen lalu lintas yang buruk dapat berdampak luas ke berbagai faktor dalam suatu negara. Maka, sistem prediktif yang mampu mengestimasi lalu lintas secara akurat dibutuhkan untuk merencanakan manajemen lalu lintas yang lebih tepat. Dengan pendekatan data-driven, solusi ini menjadi lebih terukur dan dapat diintegrasikan dalam sistem transportasi[^2]. 

**Referensi**:  
[^1]: Federal Highway Administration. (2020)."Congestion & Reliability." https://ops.fhwa.dot.gov/congestion_report/  
[^2]: Zheng, Z., et al. (2017). "Urban traffic prediction through transportation data analysis." IEEE Transactions on Big Data.

## Business Understanding

### Problem Statements

- Volume lalu lintas yang tinggi dan tidak terprediksi menghambat efisiensi manajemen lalu lintas dan menyebabkan kerugian ekonomi
- Diperlukan pemodelan yang dapat menangkap pola waktu dan pengaruh cuaca terhadap volume lalu lintas untuk membantu pengambilan keputusan berbasis data

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model prediksi berbasis time series dan machine learning untuk memperkirakan volume lalu lintas di masa mendatang
- Mengevaluasi dan menentukan model terbaik untuk diterapkan dalam sistem prediksi volume lalu lintas menggunakan data historis dan eksternal

### Solution statements
- Menggunakan algoritma SARIMAX sebagai baseline model time series multivariat
- Menggunakan algorita XGBoost dan LightGBM sebagai model machine learning
- Melakukan data preprocessing dan feature engineering
- Evaluasi dengan metrik MSE untuk mengukur secara objektif performa model

## Data Understanding
Sumber Data: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Terdapat 9 variabel, dengan jumlah data sebanyak 48204
- Berikut penjelasan setiap variabel:
    - date_time: waktu pencatatan data (datetime)
    - holiday: hari libur nasional (kategorik)
    - temp: suhu rata-rata dalam satuan Kelvin (numerik)
    - rain_1h: curah hujan/jam dalam satuan mm (numerik)
    - snow_1h: curah salju/jam dalam satuan mm (numerik)
    - clouds_all: persentase tutupan awan (numerik)
    - weather_main: deskripsi cuaca singkat (kategorik)
    - weather_description: deskripsi cuaca lengkap (kategorik)
    - traffic_volume: volume lalu lintas per jam dengan satuan kendaraan (numerik)

### Exploratory Data Analysis and Visualization:
- ![Rata-rata lalu lintas berdasarkan hari libur](https://drive.google.com/uc?export=view&id=1N1XU7W1hpPOjqVp__RxBcVvfHhZ8v-jy)  
Berdasarkan plot, rata-rata kepadatan lalu lintas terbanyak terjadi ketika libur Tahun Baru yang mencapai hampir 1500 kendaraan setiap jam
- ![Distribusi volume lalu lintas](https://drive.google.com/uc?export=view&id=158Ehujk0s68yvgmHuIEHY4GM6hB4KMgj)
Berdasarkan plot distribusi, volume lalu lintas bersifat multimodal dimana terdapat beberapa puncak yang mengindikasikan waktu-waktu tertentu seperti jam sibuk pagi atau sore hari  
- ![Korelasi antar fitur numerik](https://drive.google.com/uc?export=view&id=1lOvGcGhBM3xMj8fhkO9k7SfYn9GpLCv8)
Berdasarkan heatmap, korelasi antar fitur numerik sangat lemah terhadap volume lalu lintas (traffic_volume)


## Data Preparation
Langkah-langkah Data Preparation:
1. Membuat `date_time` menjadi index dan ekstraksi fitur waktu seperti jam, hari, bulan, dan tahun. Langkah ini dilakukan karena analisis yang digunakan berbasis time series
2. Mengecek missing values, dan tidak terdapat missing values
3. Mendeteksi data duplikat, dan melakukan penanganan dengan menghapus data duplikat. Langkah ini dilakukan untuk mencegah bias pada model dan overfitting, mengurangi generalisasi
4. Deteksi outlier dan penanganan dengan imputasi, interpolasi berbasis waktu untuk variabel `temp`, `rain_1h`, dan `snow_1h`. Langkah ini dilakukan karena model time series sangat sensitif terhadap outlier, perlu ditangani untuk meningkatkan akurasi model
5. Feature engineering dengan melakukan lag features dan cyclical encoding. Langkah ini dilakukan agar model dapat menangkap pola waktu lebih baik, lag features akan menambahkan memori ke model mengenai yang terjadi sebelumnya, dan cyclical encoding akan memberi informasi kepada model bahwa waktu itu berulang
6. Menghapus variabel yang kurang relevan untuk analisis ini, yaitu `weather_description` yang hanya berupa penjelasan lebih panjang dari `weather_main`
7. Encoding fitur kategorikal (`holiday`, `weather_main`) dengan One-Hot Encoding. Dilakukan karena model tidak bisa menangani data bertipe kategori, maka harus dikonversi menjadi numerik terlebih dahulu
8. Standardisasi dilakukan setelah split data menjadi train dan test. Langkah ini dilakukan karena model machine learning sensitif terhadap skala fitur, dan untuk menghindari data leakage maka standardisasi dilakukan setelah split data

## Modeling
Model yang Digunakan:
1. SARIMAX (Seasonal ARIMA with Exogenous Variables):
- Model baseline yang mempertimbangkan komponen musiman dan pengaruh variabel cuaca.
- Digunakan sebagai pembanding model machine learning.
- Kelebihan: Dapat menangkap tren dan musiman eksplisit.
- Kekurangan: Kurang fleksibel dalam menangani data non-stasioner dan non-linear.
2. XGBoost Regressor:
- Model boosting berbasis pohon keputusan dengan performa tinggi.
- Kelebihan: Menangani missing value, non-linearitas, dan pentingnya fitur.
- Kekurangan: Lebih lambat dibanding LightGBM untuk dataset besar.
3. LightGBM Regressor:
- Alternatif boosting dengan kecepatan pelatihan lebih tinggi.
- Kelebihan: Lebih cepat dari XGBoost, efisien untuk dataset besar.
- Kekurangan: Sensitif terhadap outlier jika tidak ditangani sebelumnya.  

**Model Terbaik**: XGBoost terpilih sebagai model terbaik berdasarkan metrik MSE, dan memberi prediksi yang paling mendekati nilai sesungguhnya

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