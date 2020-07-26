Naive Bayes Classification
================
Moh. Rosidi
7/23/2020

# Dataset Spotify

Pada artikel ini, kita akan membuat model prediktif pada dataset
`Spotify`. `Spotify` Merupakan dataset yang berisikan daftar lagu dan
fitur audio dari band/penyanyi ternama dunia, antara lain: Queens,
Maroon 5, dan Jason Mraz.

Kolom-kolom pada dataset tersebut, antara lain:

  - `id` : nomor seri lagu
  - `name` : judul lagu
  - `popularity` : tingkat popularitas lagu
  - `album.id` : nomor seri album
  - `album.name` : nama album
  - `album.total_tracks` : total lagu dalam album
  - `track_number` : nomor lagu dalam album
  - `duration_ms` : durasi lagu dalam satuan ms
  - `danceability` : elemen musik termasuk tempo, stabilitas ritme,
    kekuatan beat, dan keteraturan keseluruhan. Nilai 0,0 paling tidak
    bisa digunakan untuk menari dan 1,0 paling bisa digunakan untuk
    menari.
  - `energy` : Energi adalah ukuran dari 0,0 hingga 1,0 dan mewakili
    ukuran persepsi intensitas dan aktivitas. Biasanya, trek yang
    energik terasa cepat, keras, dan berisik. Sebagai contoh, death
    metal memiliki energi tinggi, sedangkan prelude Bach mendapat skor
    rendah pada skala. Fitur perseptual yang berkontribusi pada atribut
    ini meliputi rentang dinamis, persepsi kenyaringan, warna nada,
    onset rate, dan entropi umum.
  - `key` : Kunci dari trek adalah. Integer memetakan ke pitch
    menggunakan notasi Pitch Class standar. Misalnya. 0 = C, 1 = C♯ / D
    ♭, 2 = D, dan seterusnya.
  - `loudness` : Keseluruhan kenyaringan trek dalam desibel (dB). Nilai
    kenyaringan rata-rata di seluruh trek dan berguna untuk
    membandingkan kenyaringan relatif trek. Kenyaringan adalah kualitas
    suara yang merupakan korelasi psikologis utama dari kekuatan fisik
    (amplitudo). Nilai kisaran khas antara -60 dan 0 db.
  - `mode` : Mode menunjukkan modalitas (besar atau kecil) dari suatu
    trek, jenis skala dari mana konten melodinya diturunkan. Mayor
    diwakili oleh 1 dan minor adalah 0.
  - `speechiness` : Speechiness mendeteksi keberadaan kata-kata yang
    diucapkan di trek. Semakin eksklusif pidato-seperti rekaman (mis.
    Acara bincang-bincang, buku audio, puisi), semakin dekat dengan 1.0
    nilai atribut. Nilai di atas 0,66 menggambarkan trek yang mungkin
    seluruhnya terbuat dari kata-kata yang diucapkan. Nilai antara 0,33
    dan 0,66 menggambarkan trek yang mungkin berisi musik dan ucapan,
    baik dalam bagian atau lapisan, termasuk kasus-kasus seperti musik
    rap. Nilai di bawah 0,33 kemungkinan besar mewakili musik dan trek
    non-ucapan lainnya.
  - `acousticness` : Ukuran kepercayaan dari 0,0 hingga 1,0 dari apakah
    trek akustik. 1.0 mewakili kepercayaan tinggi trek adalah akustik.
  - `instrumentalness` : Memprediksi apakah suatu lagu tidak mengandung
    vokal. Suara “Ooh” dan “aah” diperlakukan sebagai instrumen dalam
    konteks ini. Rap atau trek kata yang diucapkan jelas “vokal”.
    Semakin dekat nilai instrumentalness ke 1.0, semakin besar
    kemungkinan trek tidak mengandung konten vokal. Nilai di atas 0,5
    dimaksudkan untuk mewakili trek instrumental, tetapi kepercayaan
    diri lebih tinggi ketika nilai mendekati 1.0.
  - `liveness` : Mendeteksi keberadaan audiens dalam rekaman. Nilai
    liveness yang lebih tinggi mewakili probabilitas yang meningkat
    bahwa trek dilakukan secara langsung. Nilai di atas 0,8 memberikan
    kemungkinan kuat bahwa trek live.
  - `valence` : Ukuran 0,0 hingga 1,0 yang menggambarkan kepositifan
    musik yang disampaikan oleh sebuah trek. Lagu dengan valensi tinggi
    terdengar lebih positif (mis. Bahagia, ceria, gembira), sedangkan
    trek dengan valensi rendah terdengar lebih negatif (mis. Sedih,
    tertekan, marah).
  - `tempo` : Perkiraan tempo trek secara keseluruhan dalam beat per
    menit (BPM). Dalam terminologi musik, tempo adalah kecepatan atau
    kecepatan dari bagian yang diberikan dan diturunkan langsung dari
    durasi beat rata-rata.
  - `time_signature` : An estimated overall time signature of a track.
    The time signature (meter) is a notational convention to specify how
    many beats are in each bar (or measure).

# Persiapan

## Library

Terdapat beberapa paket yang digunakan dalam pembuatan model prediktif
menggunakan *naive bayes*. Paket-paket yang digunakan ditampilkan sebagai
berikut:

``` r
# library pembantu
library(tidyverse)
library(rsample)
library(recipes)
library(DataExplorer)
library(skimr)
library(DMwR)
library(modeldata)
library(MLmetrics)

# library model
library(caret)
library(klaR)

# paket penjelasan model
library(vip)
library(pdp)
```

**Paket Pembantu**

3.  `foreach` : paket untuk melakukan *parallel computing*. Diperlukan
    untuk melakukan *fitting* model *parallel random forest*
4.  `import` : paket yang menangani *dependency* fungsi antar paket
    dalam proses *fitting* model *parallel random forest*
5.  `tidyverse` : kumpulan paket dalam bidang data science
6.  `rsample` : membantu proses *data splitting*
7.  `recipes`: membantu proses data pra-pemrosesan
8.  `DataExplorer` : EDA
9.  `skimr` : membuat ringkasan data
10. `DMwR` : paket untuk melakukan sampling “smote”
11. `modeldata` : kumpulan dataset untuk membuat model *machine
    learning*

**Paket untuk Membangun Model**

1.  `caret` : berisikan sejumlah fungsi yang dapat merampingkan proses
    pembuatan model regresi dan klasifikasi
2.  `klaR` : untuk membangun model naive bayes

**Paket Interpretasi Model**

2.  `vip` : visualisasi *variable importance*
3.  `pdp` : visualisasi plot ketergantungan parsial

## Import Dataset

Import dataset dilakukan dengan menggunakan fungsi `readr()`. Fungsi ini
digunakan untuk membaca file dengan ekstensi `.csv`.

``` r
spotify <- read_csv("data/spotify.csv")

# data cleaning
key_labs = c('c', 'c#', 'd', 'd#', 'e', 'f', 
             'f#', 'g', 'g#', 'a', 'a#', 'b')
mode_labs = c('minor', 'major')

spotify <- spotify %>%
  dplyr::select(popularity, duration_ms:artist) %>%
  mutate(time_signature = factor(time_signature),
         key = factor(key, labels = key_labs),
         mode = factor(mode, labels = mode_labs),
         artist = factor(artist, labels = c("Jason_Mraz", "Maroon_5", "Queen" )))
```

# Data Splitting

Proses *data splitting* dilakukan setelah data di import ke dalam
sistem. Hal ini dilakukan untuk memastikan tidak adanya kebocoran data
yang mempengaruhi proses pembuatan model. Data dipisah menjadi dua buah
set, yaitu: *training* dan *test*. Data *training* adalah data yang akan
kita gunakan untuk membentuk model. Seluruh proses sebelum uji model
akan menggunakan data *training*. Proses tersebut, antara lain: EDA,
*feature engineering*, dan validasi silang. Data *test* hanya digunakan
saat kita akan menguji performa model dengan data baru yang belum pernah
dilihat sebelumnya.

Terdapat dua buah jenis sampling pada tahapan *data splitting*, yaitu:

1.  *random sampling* : sampling acak tanpa mempertimbangkan adanya
    strata dalam data
2.  *startified random sampling* : sampling dengan memperhatikan strata
    dalam sebuah variabel.

Dalam proses pembentukan model kali ini, kita akan menggunakan metode
kedua dengan tujuan untuk memperoleh distribusi yang seragam dari
variabel target (`artist`).

``` r
set.seed(123)

split  <- initial_split(spotify, prop = 0.8, strata = "artist")
spotify_train  <- training(split)
spotify_test   <- testing(split)
```

Untuk mengecek distribusi dari kedua set data, kita dapat
mevisualisasikan distribusi dari variabel target pada kedua set
tersebut.

``` r
# training set
ggplot(spotify_train, aes(x = artist)) + 
  geom_bar() 
```

![](temp_files/figure-gfm/target-vis-1.png)<!-- -->

``` r
# test set
ggplot(spotify_test, aes(x = artist)) + 
  geom_bar() 
```

![](temp_files/figure-gfm/target-vis-2.png)<!-- -->

# Analisis Data Eksploratif

Analsiis data eksploratif (EDA) ditujukan untuk mengenali data sebelum
kita menentukan algoritma yang cocok digunakan untuk menganalisa data
lebih lanjut. EDA merupakan sebuah proses iteratif yang secara garis
besar menjawab beberapa pertanyaan umum, seperti:

1.  Bagaimana distribusi data pada masing-masing variabel?
2.  Apakah terdapat asosiasi atau hubungan antar variabel dalam data?

## Ringkasan Data

Terdapat dua buah fungsi yang digunakan dalam membuat ringkasan data,
antara lain:

1.  `glimpse()`: varian dari `str()` untuk mengecek struktur data.
    Fungsi ini menampilkan transpose dari tabel data dengan menambahkan
    informasi, seperti: jenis data dan dimensi tabel.
2.  `skim()` : fungsi dari paket `skimr` untuk membuat ringkasan data
    yang lebih detail dibanding `glimpse()`, seperti: statistika
    deskriptif masing-masing kolom, dan informasi *missing value* dari
    masing-masing kolom.
3.  `plot_missing()` : fungsi untuk memvisualisasikan persentase
    *missing value* pada masing-masing variabel atau kolom data

<!-- end list -->

``` r
glimpse(spotify_train)
```

    ## Rows: 982
    ## Columns: 15
    ## $ popularity       <dbl> 54, 74, 64, 54, 55, 53, 54, 68, 53, 53, 55, 70, 68, …
    ## $ duration_ms      <dbl> 239751, 199849, 190642, 196120, 193603, 183427, 2104…
    ## $ danceability     <dbl> 0.526, 0.799, 0.655, 0.759, 0.934, 0.812, 0.604, 0.6…
    ## $ energy           <dbl> 0.608, 0.597, 0.603, 0.604, 0.564, 0.670, 0.405, 0.4…
    ## $ key              <fct> a#, f, g#, g#, b, f, a#, c, c, c#, c, g, a, g#, e, d…
    ## $ loudness         <dbl> -5.776, -5.131, -5.014, -6.663, -4.806, -4.008, -8.2…
    ## $ mode             <fct> minor, minor, major, minor, major, major, major, maj…
    ## $ speechiness      <dbl> 0.1690, 0.0611, 0.0555, 0.0510, 0.0638, 0.0901, 0.05…
    ## $ acousticness     <dbl> 0.1270, 0.0788, 0.0959, 0.1410, 0.4610, 0.1720, 0.73…
    ## $ instrumentalness <dbl> 0.00e+00, 5.66e-06, 0.00e+00, 0.00e+00, 1.84e-05, 3.…
    ## $ liveness         <dbl> 0.1130, 0.1000, 0.1070, 0.1490, 0.1010, 0.2530, 0.10…
    ## $ valence          <dbl> 0.3720, 0.4190, 0.4520, 0.4180, 0.5430, 0.6540, 0.08…
    ## $ tempo            <dbl> 93.311, 110.001, 126.088, 121.096, 115.092, 125.087,…
    ## $ time_signature   <fct> 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4…
    ## $ artist           <fct> Maroon_5, Maroon_5, Maroon_5, Maroon_5, Maroon_5, Ma…

``` r
skim(spotify_train)
```

|                                                  |                |
| :----------------------------------------------- | :------------- |
| Name                                             | spotify\_train |
| Number of rows                                   | 982            |
| Number of columns                                | 15             |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |                |
| Column type frequency:                           |                |
| factor                                           | 4              |
| numeric                                          | 11             |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |                |
| Group variables                                  | None           |

Data summary

**Variable type: factor**

| skim\_variable  | n\_missing | complete\_rate | ordered | n\_unique | top\_counts                    |
| :-------------- | ---------: | -------------: | :------ | --------: | :----------------------------- |
| key             |          0 |              1 | FALSE   |        12 | d: 169, g: 132, a: 127, e: 119 |
| mode            |          0 |              1 | FALSE   |         2 | maj: 678, min: 304             |
| time\_signature |          0 |              1 | FALSE   |         5 | 4: 866, 3: 93, 5: 17, 0: 3     |
| artist          |          0 |              1 | FALSE   |         3 | Que: 575, Mar: 265, Jas: 142   |

**Variable type: numeric**

| skim\_variable   | n\_missing | complete\_rate |      mean |        sd |      p0 |       p25 |       p50 |       p75 |       p100 | hist  |
| :--------------- | ---------: | -------------: | --------: | --------: | ------: | --------: | --------: | --------: | ---------: | :---- |
| popularity       |          0 |              1 |     29.94 |     13.81 |    0.00 |     20.25 |     27.00 |     36.00 |      82.00 | ▂▇▃▁▁ |
| duration\_ms     |          0 |              1 | 234370.96 | 119727.20 | 4066.00 | 185690.00 | 223633.00 | 270816.50 | 2054800.00 | ▇▁▁▁▁ |
| danceability     |          0 |              1 |      0.50 |      0.19 |    0.00 |      0.34 |      0.50 |      0.65 |       0.95 | ▁▇▇▇▂ |
| energy           |          0 |              1 |      0.65 |      0.24 |    0.01 |      0.48 |      0.70 |      0.84 |       1.00 | ▁▃▅▇▇ |
| loudness         |          0 |              1 |    \-8.63 |      4.26 | \-33.59 |   \-10.81 |    \-7.68 |    \-5.60 |     \-1.87 | ▁▁▁▆▇ |
| speechiness      |          0 |              1 |      0.10 |      0.15 |    0.00 |      0.04 |      0.05 |      0.09 |       0.96 | ▇▁▁▁▁ |
| acousticness     |          0 |              1 |      0.27 |      0.28 |    0.00 |      0.03 |      0.15 |      0.43 |       0.98 | ▇▂▁▂▁ |
| instrumentalness |          0 |              1 |      0.05 |      0.18 |    0.00 |      0.00 |      0.00 |      0.00 |       1.00 | ▇▁▁▁▁ |
| liveness         |          0 |              1 |      0.34 |      0.30 |    0.00 |      0.11 |      0.20 |      0.45 |       1.00 | ▇▃▁▁▂ |
| valence          |          0 |              1 |      0.44 |      0.24 |    0.00 |      0.26 |      0.42 |      0.62 |       0.97 | ▅▇▇▆▃ |
| tempo            |          0 |              1 |    117.65 |     30.19 |    0.00 |     95.00 |    116.05 |    139.96 |     207.55 | ▁▂▇▆▁ |

``` r
plot_missing(spotify_train)
```

![](temp_files/figure-gfm/missing-vis-1.png)<!-- -->

Berdasarkan ringkasan data yang dihasilkan, diketahui dimensi data
sebesar 982 baris dan 15 kolom. Dengan rincian masing-masing kolom,
yaitu: 4 kolom dengan jenis data factor dan 11 kolom dengan jenis data
numeric. Informasi lain yang diketahui adalah seluruh kolom dalam data
tidak memiliki *missing value*.

## Variasi

Variasi dari tiap variabel dapat divisualisasikan dengan menggunakan
histogram (numerik) dan baplot (kategorikal).

``` r
plot_histogram(spotify_train, ncol = 2L, nrow = 2L)
```

![](temp_files/figure-gfm/hist-1.png)<!-- -->![](temp_files/figure-gfm/hist-2.png)<!-- -->![](temp_files/figure-gfm/hist-3.png)<!-- -->

``` r
plot_bar(spotify_train, ncol = 2L, nrow = 2L)
```

![](temp_files/figure-gfm/bar-1.png)<!-- -->

Berdasarkan hasil visualisasi diperoleh bahwa sebagian besar variabel
numerik memiliki distribusi yang tidak simetris. Sedangkan pada variabel
kategorikal diketahui bahwa seluruh variabel memiliki variasi yang tidak
mendekati nol atau nol. Untuk mengetahui variabel dengan variasi
mendekati nol atau nol, dapat menggunakan sintaks berikut:

``` r
nzvar <- nearZeroVar(spotify_train, saveMetrics = TRUE) %>% 
  rownames_to_column() %>% 
  filter(nzv)
nzvar
```

    ## [1] rowname       freqRatio     percentUnique zeroVar       nzv          
    ## <0 rows> (or 0-length row.names)

## Kovarian

Kovarian dapat dicek melalui visualisasi *heatmap* koefisien korelasi.

``` r
plot_correlation(spotify_train, 
                 cor_args = list(method = "spearman"))
```

![](temp_files/figure-gfm/heatmap-1.png)<!-- -->

# Target and Feature Engineering

*Data preprocessing* dan *engineering* mengacu pada proses penambahan,
penghapusan, atau transformasi data. Waktu yang diperlukan untuk
memikirkan identifikasi kebutuhan *data engineering* dapat berlangsung
cukup lama dan proprsinya akan menjadi yang terbesar dibandingkan
analisa lainnya. Hal ini disebabkan karena kita perlu untuk memahami
data apa yang akan kita oleh atau diinputkan ke dalam model.

Untuk menyederhanakan proses *feature engineerinh*, kita harus
memikirkannya sebagai sebuah *blueprint* dibanding melakukan tiap
tugasnya secara satu persatu. Hal ini membantu kita dalam dua hal:

1.  Berpikir secara berurutan
2.  Mengaplikasikannya secara tepat selama proses *resampling*

## Urutan Langkah-Langkah Feature Engineering

Memikirkan *feature engineering* sebagai sebuah *blueprint* memaksa kita
untuk memikirkan urutan langkah-langkah *preprocessing* data. Meskipun
setiap masalah mengharuskan kita untuk memikirkan efek *preprocessing*
berurutan, ada beberapa saran umum yang harus kita pertimbangkan:

  - Jika menggunakan log atau transformasi Box-Cox, jangan memusatkan
    data terlebih dahulu atau melakukan operasi apa pun yang dapat
    membuat data menjadi tidak positif. Atau, gunakan transformasi
    Yeo-Johnson sehingga kita tidak perlu khawatir tentang hal ini.
  - *One-hot* atau *dummy encoding* biasanya menghasilkan data jarang
    (*sparse*) yang dapat digunakan oleh banyak algoritma secara
    efisien. Jika kita menstandarisasikan data tersebut, kita akan
    membuat data menjadi padat (*dense*) dan kita kehilangan efisiensi
    komputasi. Akibatnya, sering kali lebih disukai untuk standardisasi
    fitur numerik kita dan kemudian *one-hot/dummy endode*.
  - Jika kila mengelompokkan kategori (*lumping*) yang jarang terjadi
    secara bersamaan, lakukan sebelum *one-hot/dummy endode*.
  - Meskipun kita dapat melakukan prosedur pengurangan dimensi pada
    fitur-fitur kategorikal, adalah umum untuk melakukannya terutama
    pada fitur numerik ketika melakukannya untuk tujuan rekayasa fitur.

Sementara kebutuhan proyek kita mungkin beragam, berikut ini adalah
urutan langkah-langkah potensial yang disarankan untuk sebagian besar
masalah:

1.  Filter fitur dengan varians nol (*zero varians*) atau hampir nol
    (*near zero varians*).
2.  Lakukan imputasi jika diperlukan.
3.  Normalisasi untuk menyelesaikan *skewness* fitur numerik.
4.  Standardisasi fitur numerik (*centering* dan *scaling*).
5.  Lakukan reduksi dimensi (mis., PCA) pada fitur numerik.
6.  *one-hot/dummy endode* pada fitur kategorikal.

## Meletakkan Seluruh Proses Secara Bersamaan

Untuk mengilustrasikan bagaimana proses ini bekerja bersama menggunakan
R, mari kita lakukan penilaian ulang sederhana pada set data `ames` yang
kita gunakan dan lihat apakah beberapa *feature engineering* sederhana
meningkatkan kemampuan prediksi model kita. Tapi pertama-tama, kita
berkenalan dengat paket `recipe`.

Paket `recipe` ini memungkinkan kita untuk mengembangkan *bluprint
feature engineering* secara berurutan. Gagasan di balik `recipe` mirip
dengan `caret :: preProcess()` di mana kita ingin membuat *blueprint
preprocessing* tetapi menerapkannya nanti dan dalam setiap resample.

Ada tiga langkah utama dalam membuat dan menerapkan rekayasa fitur
dengan `recipe`:

1.  `recipe()`: tempat kita menentukan langkah-langkah rekayasa fitur
    untuk membuat *blueprint*.
2.  `prep()`: memperkirakan parameter *feature engineering* berdasarkan
    data *training*.
3.  `bake()`: terapkan *blueprint* untuk data baru.

<!-- end list -->

``` r
blueprint <- recipe(artist ~ ., data = spotify_train) %>%
  step_nzv(all_nominal())  %>%
  
  # 2. imputation to missing value
  # step_medianimpute("<Num_Var_name>") %>% # median imputation
  # step_meanimpute("<Num_var_name>") %>% # mean imputation
  # step_modeimpute("<Cat_var_name>") %>% # mode imputation
  # step_bagimpute("<Var_name>") %>% # random forest imputation
  # step_knnimpute("<Var_name>") %>% # knn imputation
  
  # Label encoding for categorical variable with many classes 
  # step_integer("<Cat_var_name>") %>%
  
  # 3. normalize to resolve numeric feature skewness
  step_center(all_numeric(), -all_outcomes()) %>%
  
  # 4. standardize (center and scale) numeric feature
  step_scale(all_numeric(), -all_outcomes()) 
```

Selanjutnya, *blueprint* yang telah dibuat dilakukan *training* pada
data *training*. Perlu diperhatikan, kita tidak melakukan proses
*training* pada data *test* untuk mencegah *data leakage*.

``` r
prepare <- prep(blueprint, training = spotify_train)
prepare
```

    ## Data Recipe
    ## 
    ## Inputs:
    ## 
    ##       role #variables
    ##    outcome          1
    ##  predictor         14
    ## 
    ## Training data contained 982 data points and no missing data.
    ## 
    ## Operations:
    ## 
    ## Sparse, unbalanced variable filter removed no terms [trained]
    ## Centering for popularity, duration_ms, danceability, ... [trained]
    ## Scaling for popularity, duration_ms, danceability, ... [trained]

Langkah terakhir adalah mengaplikasikan *blueprint* pada data *training*
dan *test* menggunakan fungsi `bake()`.

``` r
baked_train <- bake(prepare, new_data = spotify_train)
baked_test <- bake(prepare, new_data = spotify_test)
baked_train
```

    ## # A tibble: 982 x 15
    ##    popularity duration_ms danceability  energy key   loudness mode  speechiness
    ##         <dbl>       <dbl>        <dbl>   <dbl> <fct>    <dbl> <fct>       <dbl>
    ##  1       1.74      0.0449        0.161 -0.173  a#      0.669  minor      0.471 
    ##  2       3.19     -0.288         1.61  -0.220  f       0.821  minor     -0.234 
    ##  3       2.47     -0.365         0.844 -0.194  g#      0.848  major     -0.270 
    ##  4       1.74     -0.319         1.39  -0.190  g#      0.461  minor     -0.300 
    ##  5       1.82     -0.341         2.32  -0.360  b       0.897  major     -0.216 
    ##  6       1.67     -0.426         1.68   0.0904 f       1.08   major     -0.0444
    ##  7       1.74     -0.200         0.574 -1.03   a#      0.0820 major     -0.243 
    ##  8       2.76     -0.160         0.611 -0.793  c       0.369  major     -0.211 
    ##  9       1.67      3.80          1.34   0.698  c       0.455  minor     -0.173 
    ## 10       1.67     -0.0152        1.51  -0.0880 c#      0.772  major     -0.196 
    ## # … with 972 more rows, and 7 more variables: acousticness <dbl>,
    ## #   instrumentalness <dbl>, liveness <dbl>, valence <dbl>, tempo <dbl>,
    ## #   time_signature <fct>, artist <fct>

# Naive Bayes

Klasifikasi Bayes naif didasarkan pada probabilitas Bayesian, yang
berasal dari [Reverend Thomas
Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes). Probabilitas
Bayesian menggabungkan konsep probabilitas kondisional, probabilitas
peristiwa A mengingat bahwa peristiwa B telah terjadi \[dilambangkan
sebagai \(P(A \vert B)\).

Teorema bayes dapat dituliskan sebagai berikut:

\[
P(C_k \vert X) = \frac{P(C_k) \cdot P(X \vert C_k)}{P(X)} \tag{1}
\]

dimana:

  - \(P(C_k)\) merupakan probabilitas *prior* output
  - \(P(X)\) merupakan probabilitas prediktor
  - \(P(X \vert C_k)\) merupakan probabilitas bersyarat atau kemungkinan
  - \(P(C_k \vert X)\) probabilitas posterior.

## Validasi Silang dan Parameter Tuning

Langkah pertama yang perlu dilakukan dalam melakukan kegiatan validasi
silang adalah menentukan spesifikasi parameter validasi silang. Fungsi
`trainControl()` merupakan fungsi yang dapat kita gunakan untu menetukan
metode validasi silang yang dilakukan dan spesifikasi terkait metode
validasi silang yang digunakan.

``` r
# spesifikasi metode validasi silang
cv <- trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 10, 
  # repeats = 5,
  classProbs = TRUE,
  savePredictions = TRUE,
  summaryFunction = multiClassSummary,
  allowParallel = TRUE
)
```

Selanjutnya spesifikasikan *hyperparameter* yang akan di *tuning*.

``` r
## Construct grid of hyperparameter values
hyper_grid <- expand.grid(
  # kernel density estimate for continous variables vs gaussian density estimate
  usekernel = c(TRUE, FALSE),
  # incorporate the Laplace smoother.
  fL = 0:5,
  # djust the bandwidth of the kernel density (larger numbers mean more flexible density estimate)
  adjust = seq(0, 5, by = 1)
)
```

Setelah parameter *tuning* dan validasi silang dispesifikasikan, proses
training dilakukan menggunakan fungsi `train()`.

``` r
system.time(
nb_fit_cv <- train(
  blueprint,
  data = spotify_train,
  method = "nb",
  trControl = cv,
  tuneGrid =  hyper_grid,
  metric = "AUC"
  )
)
```

    ##    user  system elapsed 
    ## 168.710   0.198 169.327

``` r
nb_fit_cv
```

    ## Naive Bayes 
    ## 
    ## 982 samples
    ##  14 predictor
    ##   3 classes: 'Jason_Mraz', 'Maroon_5', 'Queen' 
    ## 
    ## Recipe steps: nzv, center, scale 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 884, 883, 884, 884, 882, 884, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  fL  adjust  logLoss   AUC        prAUC      Accuracy   Kappa    
    ##   FALSE      0   0       1.278849  0.8197866  0.6438033  0.6658495  0.4403292
    ##   FALSE      0   1       1.278849  0.8197866  0.6438033  0.6658495  0.4403292
    ##   FALSE      0   2       1.278849  0.8197866  0.6438033  0.6658495  0.4403292
    ##   FALSE      0   3       1.278849  0.8197866  0.6438033  0.6658495  0.4403292
    ##   FALSE      0   4       1.278849  0.8197866  0.6438033  0.6658495  0.4403292
    ##   FALSE      0   5       1.278849  0.8197866  0.6438033  0.6658495  0.4403292
    ##   FALSE      1   0       1.273387  0.8205370  0.6436999  0.6658495  0.4398256
    ##   FALSE      1   1       1.273387  0.8205370  0.6436999  0.6658495  0.4398256
    ##   FALSE      1   2       1.273387  0.8205370  0.6436999  0.6658495  0.4398256
    ##   FALSE      1   3       1.273387  0.8205370  0.6436999  0.6658495  0.4398256
    ##   FALSE      1   4       1.273387  0.8205370  0.6436999  0.6658495  0.4398256
    ##   FALSE      1   5       1.273387  0.8205370  0.6436999  0.6658495  0.4398256
    ##   FALSE      2   0       1.269690  0.8211361  0.6438579  0.6658495  0.4391067
    ##   FALSE      2   1       1.269690  0.8211361  0.6438579  0.6658495  0.4391067
    ##   FALSE      2   2       1.269690  0.8211361  0.6438579  0.6658495  0.4391067
    ##   FALSE      2   3       1.269690  0.8211361  0.6438579  0.6658495  0.4391067
    ##   FALSE      2   4       1.269690  0.8211361  0.6438579  0.6658495  0.4391067
    ##   FALSE      2   5       1.269690  0.8211361  0.6438579  0.6658495  0.4391067
    ##   FALSE      3   0       1.266831  0.8209991  0.6449057  0.6658495  0.4391067
    ##   FALSE      3   1       1.266831  0.8209991  0.6449057  0.6658495  0.4391067
    ##   FALSE      3   2       1.266831  0.8209991  0.6449057  0.6658495  0.4391067
    ##   FALSE      3   3       1.266831  0.8209991  0.6449057  0.6658495  0.4391067
    ##   FALSE      3   4       1.266831  0.8209991  0.6449057  0.6658495  0.4391067
    ##   FALSE      3   5       1.266831  0.8209991  0.6449057  0.6658495  0.4391067
    ##   FALSE      4   0       1.264437  0.8211179  0.6450835  0.6679008  0.4414398
    ##   FALSE      4   1       1.264437  0.8211179  0.6450835  0.6679008  0.4414398
    ##   FALSE      4   2       1.264437  0.8211179  0.6450835  0.6679008  0.4414398
    ##   FALSE      4   3       1.264437  0.8211179  0.6450835  0.6679008  0.4414398
    ##   FALSE      4   4       1.264437  0.8211179  0.6450835  0.6679008  0.4414398
    ##   FALSE      4   5       1.264437  0.8211179  0.6450835  0.6679008  0.4414398
    ##   FALSE      5   0       1.262356  0.8211917  0.6451597  0.6699314  0.4439075
    ##   FALSE      5   1       1.262356  0.8211917  0.6451597  0.6699314  0.4439075
    ##   FALSE      5   2       1.262356  0.8211917  0.6451597  0.6699314  0.4439075
    ##   FALSE      5   3       1.262356  0.8211917  0.6451597  0.6699314  0.4439075
    ##   FALSE      5   4       1.262356  0.8211917  0.6451597  0.6699314  0.4439075
    ##   FALSE      5   5       1.262356  0.8211917  0.6451597  0.6699314  0.4439075
    ##    TRUE      0   0            NaN        NaN        NaN        NaN        NaN
    ##    TRUE      0   1       3.402742  0.8246174  0.6173132  0.4551503  0.2432881
    ##    TRUE      0   2       3.284285  0.8167032  0.6106976  0.4358337  0.2144767
    ##    TRUE      0   3       3.359182  0.8105104  0.6105745  0.4195267  0.1869686
    ##    TRUE      0   4       3.465201  0.8034659  0.6001372  0.4164657  0.1755901
    ##    TRUE      0   5       3.612698  0.7936872  0.5895826  0.4052515  0.1567337
    ##    TRUE      1   0            NaN        NaN        NaN        NaN        NaN
    ##    TRUE      1   1       3.398026  0.8251155  0.6182454  0.4551503  0.2433385
    ##    TRUE      1   2       3.280359  0.8168886  0.6122282  0.4348133  0.2126637
    ##    TRUE      1   3       3.355244  0.8112573  0.6114152  0.4215675  0.1901740
    ##    TRUE      1   4       3.461640  0.8044133  0.6006850  0.4185065  0.1788375
    ##    TRUE      1   5       3.609744  0.7944448  0.5911277  0.4082923  0.1598854
    ##    TRUE      2   0            NaN        NaN        NaN        NaN        NaN
    ##    TRUE      2   1       3.393827  0.8250735  0.6184354  0.4510687  0.2385625
    ##    TRUE      2   2       3.276519  0.8173167  0.6123638  0.4358337  0.2138019
    ##    TRUE      2   3       3.351642  0.8115273  0.6118250  0.4225879  0.1914937
    ##    TRUE      2   4       3.458551  0.8048494  0.6018036  0.4174861  0.1772586
    ##    TRUE      2   5       3.607268  0.7947125  0.5925347  0.4093127  0.1612011
    ##    TRUE      3   0            NaN        NaN        NaN        NaN        NaN
    ##    TRUE      3   1       3.389918  0.8254113  0.6186109  0.4520891  0.2401138
    ##    TRUE      3   2       3.272863  0.8175377  0.6134645  0.4378745  0.2169212
    ##    TRUE      3   3       3.348314  0.8117895  0.6113848  0.4215879  0.1903412
    ##    TRUE      3   4       3.455778  0.8049941  0.6019677  0.4174861  0.1769605
    ##    TRUE      3   5       3.605094  0.7948269  0.5926829  0.4103435  0.1613121
    ##    TRUE      4   0            NaN        NaN        NaN        NaN        NaN
    ##    TRUE      4   1       3.386187  0.8254601  0.6184662  0.4510687  0.2386608
    ##    TRUE      4   2       3.269386  0.8177410  0.6136063  0.4368541  0.2149974
    ##    TRUE      4   3       3.345206  0.8121171  0.6113287  0.4215879  0.1898798
    ##    TRUE      4   4       3.453232  0.8048985  0.6021824  0.4174861  0.1766474
    ##    TRUE      4   5       3.603119  0.7948345  0.5934877  0.4103435  0.1613121
    ##    TRUE      5   0            NaN        NaN        NaN        NaN        NaN
    ##    TRUE      5   1       3.382621  0.8255877  0.6185547  0.4520891  0.2398472
    ##    TRUE      5   2       3.266083  0.8178476  0.6138774  0.4368541  0.2149974
    ##    TRUE      5   3       3.342288  0.8125759  0.6143652  0.4215879  0.1897461
    ##    TRUE      5   4       3.450865  0.8050184  0.6026464  0.4174861  0.1766474
    ##    TRUE      5   5       3.601292  0.7949442  0.5932674  0.4093334  0.1602202
    ##   Mean_F1    Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value
    ##   0.5686667  0.5955958         0.8298410         0.6151744          
    ##   0.5686667  0.5955958         0.8298410         0.6151744          
    ##   0.5686667  0.5955958         0.8298410         0.6151744          
    ##   0.5686667  0.5955958         0.8298410         0.6151744          
    ##   0.5686667  0.5955958         0.8298410         0.6151744          
    ##   0.5686667  0.5955958         0.8298410         0.6151744          
    ##   0.5695578  0.5967321         0.8294909         0.6231160          
    ##   0.5695578  0.5967321         0.8294909         0.6231160          
    ##   0.5695578  0.5967321         0.8294909         0.6231160          
    ##   0.5695578  0.5967321         0.8294909         0.6231160          
    ##   0.5695578  0.5967321         0.8294909         0.6231160          
    ##   0.5695578  0.5967321         0.8294909         0.6231160          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5674313  0.5949259         0.8293586         0.6275497          
    ##   0.5688302  0.5961329         0.8301571         0.6301291          
    ##   0.5688302  0.5961329         0.8301571         0.6301291          
    ##   0.5688302  0.5961329         0.8301571         0.6301291          
    ##   0.5688302  0.5961329         0.8301571         0.6301291          
    ##   0.5688302  0.5961329         0.8301571         0.6301291          
    ##   0.5688302  0.5961329         0.8301571         0.6301291          
    ##   0.5704163  0.5972924         0.8309490         0.6361151          
    ##   0.5704163  0.5972924         0.8309490         0.6361151          
    ##   0.5704163  0.5972924         0.8309490         0.6361151          
    ##   0.5704163  0.5972924         0.8309490         0.6361151          
    ##   0.5704163  0.5972924         0.8309490         0.6361151          
    ##   0.5704163  0.5972924         0.8309490         0.6361151          
    ##         NaN        NaN               NaN               NaN          
    ##   0.4282098  0.5412341         0.7630131         0.5378131          
    ##   0.4003024  0.5112294         0.7520570         0.5266774          
    ##   0.3731002  0.4791084         0.7428777         0.5160610          
    ##   0.3629891  0.4679241         0.7388788         0.5148649          
    ##   0.3451649  0.4504068         0.7320843         0.5167879          
    ##         NaN        NaN               NaN               NaN          
    ##   0.4281641  0.5412442         0.7630196         0.5377750          
    ##   0.4000188  0.5099474         0.7513956         0.5270432          
    ##   0.3771273  0.4838703         0.7438167         0.5193258          
    ##   0.3678213  0.4726860         0.7398178         0.5190267          
    ##   0.3504977  0.4539372         0.7332121         0.5234652          
    ##         NaN        NaN               NaN               NaN          
    ##   0.4236020  0.5364591         0.7613597         0.5348895          
    ##   0.4006460  0.5112294         0.7517263         0.5275296          
    ##   0.3791301  0.4862513         0.7441539         0.5210676          
    ##   0.3658447  0.4703051         0.7393483         0.5174329          
    ##   0.3532885  0.4563181         0.7335445         0.5268057          
    ##         NaN        NaN               NaN               NaN          
    ##   0.4250643  0.5388401         0.7618292         0.5359803          
    ##   0.4040464  0.5159913         0.7526587         0.5301247          
    ##   0.3780583  0.4856766         0.7436311         0.5209406          
    ##   0.3661686  0.4703051         0.7392160         0.5189866          
    ##   0.3532733  0.4552655         0.7338042         0.5270079          
    ##         NaN        NaN               NaN               NaN          
    ##   0.4235793  0.5364591         0.7614258         0.5348817          
    ##   0.4015583  0.5136104         0.7520635         0.5278955          
    ##   0.3783303  0.4856766         0.7434131         0.5216627          
    ##   0.3663772  0.4703051         0.7390772         0.5203325          
    ##   0.3532733  0.4552655         0.7338042         0.5270079          
    ##         NaN        NaN               NaN               NaN          
    ##   0.4245551  0.5370338         0.7621269         0.5378920          
    ##   0.4015583  0.5136104         0.7520635         0.5278955          
    ##   0.3784943  0.4856766         0.7433470         0.5225836          
    ##   0.3663772  0.4703051         0.7390772         0.5203325          
    ##   0.3521452  0.4546807         0.7333347         0.5266685          
    ##   Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall  Mean_Detection_Rate
    ##   0.8227328            0.6151744       0.5955958    0.2219498          
    ##   0.8227328            0.6151744       0.5955958    0.2219498          
    ##   0.8227328            0.6151744       0.5955958    0.2219498          
    ##   0.8227328            0.6151744       0.5955958    0.2219498          
    ##   0.8227328            0.6151744       0.5955958    0.2219498          
    ##   0.8227328            0.6151744       0.5955958    0.2219498          
    ##   0.8224581            0.6231160       0.5967321    0.2219498          
    ##   0.8224581            0.6231160       0.5967321    0.2219498          
    ##   0.8224581            0.6231160       0.5967321    0.2219498          
    ##   0.8224581            0.6231160       0.5967321    0.2219498          
    ##   0.8224581            0.6231160       0.5967321    0.2219498          
    ##   0.8224581            0.6231160       0.5967321    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8225913            0.6275497       0.5949259    0.2219498          
    ##   0.8233168            0.6301291       0.5961329    0.2226336          
    ##   0.8233168            0.6301291       0.5961329    0.2226336          
    ##   0.8233168            0.6301291       0.5961329    0.2226336          
    ##   0.8233168            0.6301291       0.5961329    0.2226336          
    ##   0.8233168            0.6301291       0.5961329    0.2226336          
    ##   0.8233168            0.6301291       0.5961329    0.2226336          
    ##   0.8240802            0.6361151       0.5972924    0.2233105          
    ##   0.8240802            0.6361151       0.5972924    0.2233105          
    ##   0.8240802            0.6361151       0.5972924    0.2233105          
    ##   0.8240802            0.6361151       0.5972924    0.2233105          
    ##   0.8240802            0.6361151       0.5972924    0.2233105          
    ##   0.8240802            0.6361151       0.5972924    0.2233105          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.7730347            0.5378131       0.5412341    0.1517168          
    ##   0.7701504            0.5266774       0.5112294    0.1452779          
    ##   0.7669854            0.5160610       0.4791084    0.1398422          
    ##   0.7675136            0.5148649       0.4679241    0.1388219          
    ##   0.7659859            0.5167879       0.4504068    0.1350838          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.7730538            0.5377750       0.5412442    0.1517168          
    ##   0.7692733            0.5270432       0.5099474    0.1449378          
    ##   0.7678755            0.5193258       0.4838703    0.1405225          
    ##   0.7685230            0.5190267       0.4726860    0.1395022          
    ##   0.7670181            0.5234652       0.4539372    0.1360974          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.7713633            0.5348895       0.5364591    0.1503562          
    ##   0.7702321            0.5275296       0.5112294    0.1452779          
    ##   0.7683686            0.5210676       0.4862513    0.1408626          
    ##   0.7681193            0.5174329       0.4703051    0.1391620          
    ##   0.7675743            0.5268057       0.4563181    0.1364376          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.7718680            0.5359803       0.5388401    0.1506964          
    ##   0.7711107            0.5301247       0.5159913    0.1459582          
    ##   0.7681986            0.5209406       0.4856766    0.1405293          
    ##   0.7681097            0.5189866       0.4703051    0.1391620          
    ##   0.7677027            0.5270079       0.4552655    0.1367812          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.7713473            0.5348817       0.5364591    0.1503562          
    ##   0.7708228            0.5278955       0.5136104    0.1456180          
    ##   0.7681804            0.5216627       0.4856766    0.1405293          
    ##   0.7680960            0.5203325       0.4703051    0.1391620          
    ##   0.7677027            0.5270079       0.4552655    0.1367812          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.7717644            0.5378920       0.5370338    0.1506964          
    ##   0.7708228            0.5278955       0.5136104    0.1456180          
    ##   0.7682272            0.5225836       0.4856766    0.1405293          
    ##   0.7680960            0.5203325       0.4703051    0.1391620          
    ##   0.7674621            0.5266685       0.4546807    0.1364445          
    ##   Mean_Balanced_Accuracy
    ##   0.7127184             
    ##   0.7127184             
    ##   0.7127184             
    ##   0.7127184             
    ##   0.7127184             
    ##   0.7127184             
    ##   0.7131115             
    ##   0.7131115             
    ##   0.7131115             
    ##   0.7131115             
    ##   0.7131115             
    ##   0.7131115             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7121423             
    ##   0.7131450             
    ##   0.7131450             
    ##   0.7131450             
    ##   0.7131450             
    ##   0.7131450             
    ##   0.7131450             
    ##   0.7141207             
    ##   0.7141207             
    ##   0.7141207             
    ##   0.7141207             
    ##   0.7141207             
    ##   0.7141207             
    ##         NaN             
    ##   0.6521236             
    ##   0.6316432             
    ##   0.6109931             
    ##   0.6034015             
    ##   0.5912455             
    ##         NaN             
    ##   0.6521319             
    ##   0.6306715             
    ##   0.6138435             
    ##   0.6062519             
    ##   0.5935746             
    ##         NaN             
    ##   0.6489094             
    ##   0.6314779             
    ##   0.6152026             
    ##   0.6048267             
    ##   0.5949313             
    ##         NaN             
    ##   0.6503346             
    ##   0.6343250             
    ##   0.6146538             
    ##   0.6047605             
    ##   0.5945348             
    ##         NaN             
    ##   0.6489425             
    ##   0.6328369             
    ##   0.6145448             
    ##   0.6046911             
    ##   0.5945348             
    ##         NaN             
    ##   0.6495803             
    ##   0.6328369             
    ##   0.6145118             
    ##   0.6046911             
    ##   0.5940077             
    ## 
    ## AUC was used to select the optimal model using the largest value.
    ## The final values used for the model were fL = 5, usekernel = TRUE and adjust
    ##  = 1.

``` r
nb_fit_cv$bestTune
```

    ##    fL usekernel adjust
    ## 68  5      TRUE      1

Model terbaik dipilih berdasarkan nilai **AUC**
terbesar. Berdasarkan kriteria tersebut model yang terpilih adalalah
model yang memiliki nilai `fL` = 5, `usekernel` = TRUE dan `adjust` = 1.
Nilai **AUC** rata-rata model terbaik adalah sebagai berikut:

``` r
nb_roc <- nb_fit_cv$results %>%
  arrange(-AUC) %>%
  slice(1) %>%.[,"AUC"] 
nb_roc
```

    ## [1] 0.8255877

Berdasarkan hasil yang diperoleh, luas area dibawah kurva **ROC**
sebesar 0.8255877 Berdasarkan hasil tersebut, model klasifikasi yang
terbentuk lebih baik dibanding menebak secara acak.

Visualisasi hubungan antar parameter dan **ROC** ditampilkan pada gambar
berikut:

``` r
# visualisasi
ggplot(nb_fit_cv)
```

![](temp_files/figure-gfm/nb-cv-vis-1.png)<!-- -->

## Model Akhir

Model terbaik dari hasil proses validasi silang selanjutnya diekstrak.
Hal ini berguna untuk mengurangi ukuran model yang tersimpan. Secara
default fungsi `train()` akan mengembalikan model dengan performa
terbaik. Namun, terdapat sejumlah komponen lain dalam objek yang
terbentuk, seperti: hasil prediksi, ringkasan training, dll. yang
membuat ukuran objek menjadi besar. Untuk menguranginya, kita perlu
mengambil objek model final dari objek hasil validasi silang.

``` r
nb_fit <- nb_fit_cv$finalModel
```

Visualisasi model final *naive bayes*, dilakukan menggunakan fungsi
`plot()`.

``` r
# visualisasi
plot(nb_fit)
```

![](temp_files/figure-gfm/nb-vis-1.png)<!-- -->![](temp_files/figure-gfm/nb-vis-2.png)<!-- -->![](temp_files/figure-gfm/nb-vis-3.png)<!-- -->![](temp_files/figure-gfm/nb-vis-4.png)<!-- -->![](temp_files/figure-gfm/nb-vis-5.png)<!-- -->![](temp_files/figure-gfm/nb-vis-6.png)<!-- -->![](temp_files/figure-gfm/nb-vis-7.png)<!-- -->![](temp_files/figure-gfm/nb-vis-8.png)<!-- -->![](temp_files/figure-gfm/nb-vis-9.png)<!-- -->![](temp_files/figure-gfm/nb-vis-10.png)<!-- -->![](temp_files/figure-gfm/nb-vis-11.png)<!-- -->![](temp_files/figure-gfm/nb-vis-12.png)<!-- -->![](temp_files/figure-gfm/nb-vis-13.png)<!-- -->![](temp_files/figure-gfm/nb-vis-14.png)<!-- -->

Model yang dihasilkan selanjutnya dapat kita uji lagi menggunakan data
baru. Berikut adalah perhitungan nilai **Akurasi** model pada data
*test*.

``` r
# prediksi Attrition churn_test
pred_test <- predict(nb_fit, baked_test, type = "class")$class

## RMSE
cm <- confusionMatrix(pred_test, baked_test$artist)
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##             Reference
    ## Prediction   Jason_Mraz Maroon_5 Queen
    ##   Jason_Mraz         16        2    52
    ##   Maroon_5           19       61    61
    ##   Queen               0        2    30
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4403          
    ##                  95% CI : (0.3769, 0.5052)
    ##     No Information Rate : 0.5885          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.2289          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Jason_Mraz Class: Maroon_5 Class: Queen
    ## Sensitivity                    0.45714          0.9385       0.2098
    ## Specificity                    0.74038          0.5506       0.9800
    ## Pos Pred Value                 0.22857          0.4326       0.9375
    ## Neg Pred Value                 0.89017          0.9608       0.4645
    ## Prevalence                     0.14403          0.2675       0.5885
    ## Detection Rate                 0.06584          0.2510       0.1235
    ## Detection Prevalence           0.28807          0.5802       0.1317
    ## Balanced Accuracy              0.59876          0.7445       0.5949

Berdasarkan hasil evaluasi diperoleh nilai akurasi sebesar 0.4403292

## Interpretasi Fitur

Untuk mengetahui variabel yang paling berpengaruh secara global terhadap
hasil prediksi model, kita dapat menggunakan plot *variable importance*.

``` r
vi <- varImp(nb_fit_cv, num_features = 10) %>% ggplot()
vi
```

![](temp_files/figure-gfm/nb-vip-1.png)<!-- -->

Berdasarkan terdapat 4 buah variabel yang berpengaruh besar terhadap
prediksi yang dihasilkan oleh model, antara lain: danceability,
instrumentalness, acousticness, popularity. Untuk melihat efek dari
masing-masing variabel terhadap variabel respon, kita dapat menggunakan
*partial dependence plot*.

``` r
p1 <- pdp::partial(nb_fit_cv, pred.var = as.character(vi$data[1, 3])) %>% 
  autoplot() 

p2 <- pdp::partial(nb_fit_cv, pred.var = as.character(vi$data[2, 3])) %>% 
  autoplot()

p3 <- pdp::partial(nb_fit_cv, pred.var = as.character(vi$data[3, 3])) %>% 
  autoplot()
  

p4 <- pdp::partial(nb_fit_cv, pred.var = as.character(vi$data[4, 3])) %>% 
  autoplot()

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

![](temp_files/figure-gfm/nb-pdp-1.png)<!-- -->
