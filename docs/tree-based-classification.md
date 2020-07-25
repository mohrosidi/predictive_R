Tree Based Classification
================
Moh. Rosidi
7/22/2020

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
menggunakan *tree based algorithm*. Paket-paket ditampilkan sebagai
berikut:

``` r
# library pembantu
library(e1071)
library(foreach)
library(import)
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
library(rpart)
library(ipred)
library(ranger)
library(gbm)

# paket penjelasan model
library(rpart.plot)  
library(vip)
library(pdp)
```

**Paket Pembantu**

1.  `plyr` : paket manipulasi data yang digunakan untuk membantu proses
    *fitting* sejumlah model pohon.
2.  `e1071` : paket dengan sejumlah fungsi untuk melakukan *latent class
    analysis, short time Fourier transform, fuzzy clustering, support
    vector machines, shortest path computation, bagged clustering, naive
    Bayes classifier*, dll. Paket ini merupakan paket pembantu dalam
    proses *fitting* sejumlah model pohon
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
2.  `rpart` : membentuk model *decision trees*
3.  `ipred` : membentuk model *bagging*
4.  `ranger` : membentuk model *random forest*
5.  `gbm` : membentuk model *gradient boost model*

**Paket Interpretasi Model**

1.  `rpart.plot` : visualisasi *decision trees*
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
  select(popularity, duration_ms:artist) %>%
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

# Decision Tree Model

*Tree-based models* adalah kelas algoritma nonparametrik yang bekerja
dengan mempartisi ruang fitur ke sejumlah daerah yang lebih kecil (tidak
tumpang tindih) dengan nilai respons yang sama menggunakan seperangkat
*aturan pemisahan*. Prediksi diperoleh dengan memasang model yang lebih
sederhana (misal: Konstanta seperti nilai respons rata-rata) di setiap
wilayah. Metode membagi dan menaklukkan seperti itu dapat menghasilkan
aturan sederhana yang mudah ditafsirkan dan divisualisasikan dengan
diagram pohon.

Ada banyak metode yang dapat digunakan membangun pohon regresi, tetapi
salah satu yang tertua dikenal sebagai pendekatan pohon klasifikasi dan
regresi (CART) yang dikembangkan oleh [Breiman et
al. (1984)](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418).
Tutorial ini berfokus pada bagian regresi CART. Pohon regresi dasar
mempartisi data yang ditetapkan ke dalam subkelompok yang lebih kecil
dan kemudian melakukan fitting konstanta sederhana untuk setiap
pengamatan dalam subkelompok. Partisi dicapai dengan partisi biner
berturut-turut (alias partisi rekursif) berdasarkan pada berbagai
prediktor. Konstanta untuk memprediksi didasarkan pada nilai respons
rata-rata untuk semua pengamatan yang termasuk dalam subkelompok
tersebut.

Sebagai contoh, misalkan kita ingin memprediksi mil per galon mobil
rata-rata berdasarkan ukuran silinder (`cyl`) dan tenaga kuda (`hp`).
Semua pengamatan melalui pohon ini, dinilai pada simpul tertentu, dan
lanjutkan ke kiri jika jawabannya “ya” atau lanjutkan ke kanan jika
jawabannya “tidak”. Jadi, pertama, semua pengamatan yang memiliki 6 atau
8 silinder pergi ke cabang kiri, semua pengamatan lainnya dilanjutkan ke
cabang kanan. Selanjutnya, cabang kiri selanjutnya dipartisi oleh tenaga
kuda. Pengamatan 6 atau 8 silinder dengan tenaga kuda yang sama atau
lebih besar dari 192 dilanjutkan ke cabang kiri; mereka yang kurang dari
192 hp melanjutkan ke kanan. Cabang-cabang ini mengarah ke *terminal
node* atau *leaf nodes* yang berisi nilai respons prediksi kita. Pada
dasarnya, semua pengamatan (mobil dalam contoh ini) yang tidak memiliki
6 atau 8 silinder (cabang paling kanan) rata-rata 27 mpg. Semua
pengamatan yang memiliki 6 atau 8 silinder dan memiliki lebih dari 192
hp (cabang paling kiri) rata-rata 13 mpg.

![Prediksi mpg berdasarkan variabel cyl dan hp (Sumber:
<http://uc-r.github.io/>)](http://uc-r.github.io/public/images/analytics/regression_trees/ex_regression_tree.png)

Contoh sederhana tersebut dapat kita generalisasikan. Variabel respon
kontinu \(Y\) dan dua buah variabel input \(X_1\) dan \(X_2\). Partisi
rekursif menghasilkan tiga buah area (*nodes*), yaitu: \(R_1\), \(R_2\),
dan \(R_3\) dimana model memprediksi \(Y\) dengan sebuah konstanta
\(c_m\) pada area \(R_m\):

\[
\hat{f}\left(X\right) = \sum_{m=1}^{3} c_{m} I\left(X_1,X_2\right) \in R_m
\]

## Menentukan Split pada Decision Tree

Pertama, penting untuk mewujudkan partisi variabel yang dilakukan secara
*top-down*. Ini hanya berarti bahwa partisi yang dilakukan sebelumnya
pada pohon yang terbentuk tidak akan berubah oleh partisi selanjutnya.
Tetapi bagaimana partisi ini dibuat? Model dimulai dengan seluruh
data,\(S\), dan mencari setiap nilai berbeda dari setiap variabel input
untuk menemukan prediktor dan nilai *split* yang membagi data menjadi
dua area (\(R_1\) dan \(R_2\)) sedemikian rupa sehingga jumlah kesalahan
kuadrat keseluruhan diminimalkan:

\[
minimize \left{SSE = \sum_{i \in R_1}^{ } \left(y_i-c_1\right)^2 +  \sum_{i \in R_2}^{ } \left(y_i-c_2\right)^2\right}
\]

Setelah menemukan *split* terbaik, kita mempartisi data menjadi dua area
yang dihasilkan dan mengulangi proses *split* pada masing-masing dua
area Proses ini berlanjut sampai kriteria penghentian tercapai. Pohon
yang dihasilkan biasanya sangat dalam, kompleks yang dapat menghasilkan
prediksi yang baik pada data *training*, tetapi kemungkinan besar model
yang dibuat *overfiting* dan akan menghasilkan hasil prediksi yang buruk
pada data *test*.

## Cost complexity criterion

Seringkali ada keseimbangan yang harus dicapai dalam kedalaman dan
kompleksitas pohon untuk mengoptimalkan kinerja prediksi pada beberapa
data yang tidak terlihat. Untuk menemukan keseimbangan ini, kita
biasanya menumbuhkan pohon yang sangat besar seperti yang didefinisikan
pada bagian sebelumnya dan kemudian memangkasnya kembali untuk menemukan
sub-pohon yang optimal. Kita menemukan sub-pohon optimal dengan
menggunakan parameter kompleksitas biaya (\(\alpha\)) yang memberikan
penalti pada fungsi objektif pada persamaan penentuan *split* untuk
setiap *terminal nodes* pada tiap pohon (\(T\)).

\[
minimize\left{SSE+\alpha\left|T\right|\right}
\]

Untuk nilai \(alpha\) yang diberikan, kita dapat menemukan pohon
pemangkasan terkecil yang memiliki kesalahan penalti terendah. Jika kita
terbiasa dengan regresi dengan penalti, kita akan menyadari hubungan
dekat dengan penalti norma lasso \(L_1\). Seperti dengan metode
regularisasi ini, penalti yang lebih kecil cenderung menghasilkan model
yang lebih kompleks dan menghasilkan pohon yang lebih besar. Sedangkan
penalti yang lebih besar menghasilkan pohon yang jauh lebih kecil.
Akibatnya, ketika pohon tumbuh lebih besar, pengurangan SSE harus lebih
besar daripada penalti kompleksitas biaya. Biasanya, kita mengevaluasi
beberapa model melintasi spektrum \(\alpha\) dan menggunakan teknik
validasi silang untuk mengidentifikasi \(\alpha\) optimal dan sub-pohon
optimal.

## Kelebihan dan Kekurangan

Terdapat sejumlah kelebihan penggunaan *decision trees*, antara lain:

  - Mudah ditafsirkan.
  - Dapat membuat prediksi cepat (tidak ada perhitungan rumit, hanya
    mencari konstanta di pohon).
  - Sangat mudah untuk memahami variabel apa yang penting dalam membuat
    prediksi. Node internal (splits) adalah variabel-variabel yang
    sebagian besar mereduksi SSE.
  - Jika ada beberapa data yang hilang, kita mungkin tidak bisa pergi
    jauh-jauh ke bawah pohon menuju daun, tetapi kita masih bisa membuat
    prediksi dengan merata-rata semua daun di sub-pohon yang kita
    jangkau.
  - Model ini memberikan respons “bergerigi” non-linier, sehingga dapat
    bekerja saat permukaan regresi yang sebenarnya tidak mulus. Jika
    halus, permukaan konstan-piecewise dapat memperkirakannya secara
    dekat (dengan cukup daun).
  - Ada algoritma yang cepat dan andal untuk mempelajari pohon-pohon
    ini.

Selain kelebihan, terdapat kekurangan dalam penggunaan *decision trees*,
antara lain:

  - Pohon regresi tunggal memiliki varian yang tinggi, menghasilkan
    prediksi yang tidak stabil (subsampel alternatif dari data
    *training* dapat secara signifikan mengubah node terminal).
  - Karena varians tinggi pohon regresi tunggal memiliki akurasi
    prediksi yang buruk.

## Validasi Silang dan Parameter Tuning

Langkah pertama yang perlu dilakukan dalam melakukan kegiatan validasi
silang adalah menentukan spesifikasi parameter validasi silang. Fungsi
`trainControl()` merupakan fungsi yang dapat kita gunakan untu menetukan
metode validasi silang yang dilakukan dan spesifikasi terkait metode
validasi silang yang dugunakan.

Pada sintaks berikut dispesifikasikan `method` yang digunakan adalah
`"cv"` dengan jumlah partisi sebanyak 10 buah. Parameter lain yang ikut
ditambahkan dalam fungsi `trainControl()` adalah `search` yang
menspesifikasikan metode *parameter tuning* yang dispesifikasikan dengan
nilai `"random"`.

``` r
# spesifikasi metode validasi silang
cv <- trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 10, 
  # repeats = 5,
  classProbs = TRUE, 
  search = "random",
  sampling = "smote",
  summaryFunction = multiClassSummary,
  savePredictions = TRUE,
  allowParallel = TRUE
)
```

Selanjutnya, hasil pengaturan parameter training diinputkan ke dalam
fungsi `train()`. Dalam fungsi ini dispesifikasikan sejumlah argumen
seperti: formula yang digunakan, data training yang akan digunakan,
`method` atau `engine` yang akan digunakan untuk membentuk model. Proses
*parameter tuning* diatur melalui argumen `tuneLength` yang merupakan
kombinasi antar parameter yang akan di-*tuning* secara acak. Dalam hal
ini dipsesifkasikan nilai `tuneLength` sebesar 20 yang menunjukkan 20
kombinasi *parameter-tuning* yang digunakan.

``` r
system.time(
dt_fit_cv <- train(
  blueprint,
  data = spotify_train,
  method = "rpart",
  trControl = cv,
  tuneLength = 20,
  metric = "AUC"
  )
)
```

    ##    user  system elapsed 
    ##   4.424   0.015   4.477

``` r
dt_fit_cv
```

    ## CART 
    ## 
    ## 982 samples
    ##  14 predictor
    ##   3 classes: 'Jason_Mraz', 'Maroon_5', 'Queen' 
    ## 
    ## Recipe steps: nzv, center, scale 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 884, 883, 884, 884, 882, 884, ... 
    ## Addtional sampling using SMOTE
    ## 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp           logLoss    AUC        prAUC       Accuracy   Kappa    
    ##   0.000000000  2.5166883  0.8243107  0.53676902  0.6975346  0.4990943
    ##   0.002457002  2.1595490  0.8225172  0.52737875  0.7159021  0.5262799
    ##   0.003685504  2.0688738  0.8215409  0.53441069  0.7128510  0.5211492
    ##   0.004914005  1.7771444  0.8209020  0.54087791  0.7148299  0.5215009
    ##   0.008599509  1.4602894  0.8139481  0.50577386  0.7107269  0.5139235
    ##   0.009828010  1.3963563  0.8158808  0.48004628  0.7025636  0.5040687
    ##   0.012285012  1.2435449  0.8222083  0.48933865  0.7086658  0.5162892
    ##   0.019656020  1.0078923  0.8096960  0.42358026  0.6852270  0.4823838
    ##   0.063063063  0.9319587  0.7431252  0.22637956  0.5923873  0.3773368
    ##   0.280098280  1.0390097  0.6469959  0.09775643  0.3926319  0.1647450
    ##   Mean_F1    Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value
    ##   0.6468185  0.6804925         0.8464446         0.6566151          
    ##   0.6644143  0.6977312         0.8561310         0.6729361          
    ##   0.6610283  0.6941806         0.8540723         0.6699568          
    ##   0.6602975  0.6903130         0.8537971         0.6719011          
    ##   0.6532570  0.6837600         0.8521661         0.6696158          
    ##   0.6479062  0.6795539         0.8490373         0.6673367          
    ##   0.6581953  0.6910346         0.8536042         0.6709698          
    ##   0.6304822  0.6639836         0.8451730         0.6649376          
    ##   0.5517847  0.6045839         0.8121346         0.6259218          
    ##         NaN  0.4381952         0.7466387               NaN          
    ##   Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall  Mean_Detection_Rate
    ##   0.8325512            0.6566151       0.6804925    0.2325115          
    ##   0.8417438            0.6729361       0.6977312    0.2386340          
    ##   0.8401504            0.6699568       0.6941806    0.2376170          
    ##   0.8411376            0.6719011       0.6903130    0.2382766          
    ##   0.8391980            0.6696158       0.6837600    0.2369090          
    ##   0.8356529            0.6673367       0.6795539    0.2341879          
    ##   0.8383240            0.6709698       0.6910346    0.2362219          
    ##   0.8292541            0.6649376       0.6639836    0.2284090          
    ##   0.7956332            0.6259218       0.6045839    0.1974624          
    ##   0.7564319                  NaN       0.4381952    0.1308773          
    ##   Mean_Balanced_Accuracy
    ##   0.7634685             
    ##   0.7769311             
    ##   0.7741265             
    ##   0.7720550             
    ##   0.7679630             
    ##   0.7642956             
    ##   0.7723194             
    ##   0.7545783             
    ##   0.7083593             
    ##   0.5924170             
    ## 
    ## AUC was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.

Proses *training* berlangsung selama 4,351 detik dengan 8 buah model
yang terbentuk. Model terbaik dipilih berdasarkan nilai **AUC**
terbesar. Berdasarkan kriteria tersebut model yang terpilih adalalah
model yang memiliki nilai `cp` sebesar 0 (cp : parameter kompleksitas
atau penalti). Nilai **AUC** rata-rata model terbaik adalah sebagai
berikut:

``` r
dt_roc <- dt_fit_cv$results %>%
  arrange(-AUC) %>%
  slice(1) %>%
  select(AUC) %>% pull()
dt_roc
```

    ## [1] 0.8243107

Berdasarkan hasil yang diperoleh, luas area dibawah kurva **ROC**
sebesar 0.8243107 Berdasarkan hasil tersebut, model klasifikasi yang
terbentuk lebih baik dibanding menebak secara acak.

Visualisasi hubungan antara parameter kompleksitas dan **ROC**
ditampilkan pada gambar berikut:

``` r
# visualisasi
ggplot(dt_fit_cv)
```

![](temp_files/figure-gfm/dt-cv-vis-1.png)<!-- -->

## Model Akhir

Model terbaik dari hasil proses validasi silang selanjutnya diekstrak.
Hal ini berguna untuk mengurangi ukuran model yang tersimpan. Secara
default fungsi `train()` akan mengembalikan model dengan performa
terbaik. Namun, terdapat sejumlah komponen lain dalam objek yang
terbentuk, seperti: hasil prediksi, ringkasan training, dll. yang
membuat ukuran objek menjadi besar. Untuk menguranginya, kita perlu
mengambil objek model final dari objek hasil validasi silang.

``` r
dt_fit <- dt_fit_cv$finalModel
```

Visualisasi model final *decision tree*, dilakukan menggunakan fungsi
`rpart.plot()`.

``` r
# visualisasi
rpart.plot(dt_fit)
```

![](temp_files/figure-gfm/dt-vis-1.png)<!-- -->

Model yang dihasilkan selanjutnya dapat kita uji lagi menggunakan data
baru. Berikut adalah perhitungan nilai **Akurasi** model pada data
*test*.

``` r
pred_test <- predict(dt_fit, baked_test, type = "class")

## RMSE
cm <- confusionMatrix(pred_test, baked_test$artist)
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##             Reference
    ## Prediction   Jason_Mraz Maroon_5 Queen
    ##   Jason_Mraz         28       23    68
    ##   Maroon_5            2       35     6
    ##   Queen               5        7    69
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.5432         
    ##                  95% CI : (0.4783, 0.607)
    ##     No Information Rate : 0.5885         
    ##     P-Value [Acc > NIR] : 0.9325         
    ##                                          
    ##                   Kappa : 0.3341         
    ##                                          
    ##  Mcnemar's Test P-Value : 1.525e-15      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Jason_Mraz Class: Maroon_5 Class: Queen
    ## Sensitivity                     0.8000          0.5385       0.4825
    ## Specificity                     0.5625          0.9551       0.8800
    ## Pos Pred Value                  0.2353          0.8140       0.8519
    ## Neg Pred Value                  0.9435          0.8500       0.5432
    ## Prevalence                      0.1440          0.2675       0.5885
    ## Detection Rate                  0.1152          0.1440       0.2840
    ## Detection Prevalence            0.4897          0.1770       0.3333
    ## Balanced Accuracy               0.6813          0.7468       0.6813

Berdasarkan hasil evaluasi diperoleh nilai akurasi sebesar 0.5432099

## Interpretasi Fitur

Untuk mengetahui variabel yang paling berpengaruh secara global terhadap
hasil prediksi model *decision tree*, kita dapat menggunakan plot
*variable importance*.

``` r
vi <- vip(dt_fit_cv, num_features = 10)
vi
```

![](temp_files/figure-gfm/dt-vip-1.png)<!-- -->

Berdasarkan terdapat 4 buah variabel yang berpengaruh besar terhadap
prediksi yang dihasilkan oleh model, antara lain: popularity,
duration\_ms, acousticness, loudness. Untuk melihat efek dari
masing-masing variabel terhadap variabel respon, kita dapat menggunakan
*partial dependence plot*.

``` r
p1 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[1]) %>% 
  autoplot() 

p2 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[2]) %>% 
  autoplot()

p3 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[3]) %>% 
  autoplot()
  

p4 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[4]) %>% 
  autoplot()

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

![](temp_files/figure-gfm/dt-pdp-1.png)<!-- -->

# Bagging

Seperti disebutkan sebelumnya, model pohon tunggal memiliki kekurangan,
yaitu: varian yang tinggi. Hal ini berarti algoritma *decision tress*
memodelkan *noise* dalam data *training*-nya. Meskipun pemangkasan pohon
yang terbentuk membantu mengurangi varians ini, ada metode alternatif
yang sebenarnya mengeksploitasi variabilitas pohon tunggal dengan cara
yang secara signifikan dapat meningkatkan kinerja lebih dan di atas
pohon tunggal. Agregat bootstrap (bagging) adalah salah satu pendekatan
yang dapat digunakan (awalnya diusulkan oleh
[Breiman, 1996](https://link.springer.com/article/10.1023%2FA%3A1018054314350)).

Bagging menggabungkan dan merata-rata beberapa model. Rata-rata di
beberapa pohon mengurangi variabilitas dari satu pohon dan mengurangi
overfitting dan meningkatkan kinerja prediksi. Bagging mengikuti tiga
langkah sederhana:

1.  Buat sampel [bootstrap](http://uc-r.github.io/bootstrapping) \(m\)
    dari data *training*. Sampel bootstrap memungkinkan kita untuk
    membuat banyak set data yang sedikit berbeda tetapi dengan
    distribusi yang sama dengan set data *training* secara keseluruhan.
2.  Untuk setiap sampel bootstrap, latih satu pohon regresi tanpa
    melakukan pemangkasan (*unpruned*)
3.  Lakukan prediksi data *test* pada tiap pohon yang terbentuk dari
    setiap pohon. Hasil prediksi masing-masing pohon selanjutnya
    dirata-rata.

![Proses bagging (Sumber:
<http://uc-r.github.io/>)](http://uc-r.github.io/public/images/analytics/regression_trees/bagging3.png)

Proses ini sebenarnya dapat diterapkan pada model regresi atau
klasifikasi apa pun; Namun, metode ini memberikan peningkatan terbesar
pada model yang memiliki varian tinggi. Sebagai contoh, model parametrik
yang lebih stabil seperti regresi linier dan splines regresi
multi-adaptif cenderung kurang mengalami peningkatan dalam kinerja
prediksi.

Salah satu manfaat bagging adalah rata-rata, sampel bootstrap akan
berisi 63% (2/3) bagian dari data *training*. Ini menyisakan sekitar 33%
(1/3) data dari sampel yang di-bootstrap. Kita dapat menyebutnya sebagai
sampel *out-of-bag* (OOB). Kita dapat menggunakan pengamatan OOB untuk
memperkirakan akurasi model, menciptakan proses validasi silang alami.

## Validasi Silang

Spesifikasi validasi silang yang digunakan untuk membuat model bagging
sama dengan spesifikasi validasi silang model *decission tree*.
Perbedaannya adalah pada argumen `trainControl()` tidak ditambahkan
argumen `sample`. Hal ini disebabkan pada model bagging yang dibuat ini
tidak dilakukan *parameter tuning*.

``` r
# spesifikasi metode validasi silang
cv <-trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 10, 
  # repeats = 5,
  classProbs = TRUE, 
  search = "random",
  sampling = "smote",
  summaryFunction = multiClassSummary,
  savePredictions = TRUE,
  allowParallel = TRUE
)
```

Agumen `method` yang digunakan dalam model ini adalah `"treebag"` yang
merupakan `method` yang digunakan jika model bagging yang akan dibuat
menggunakan paket `ipred`.

``` r
system.time(
bag_fit_cv <- train(
  blueprint,
  data = spotify_train,
  method = "treebag",
  trControl = cv,
  importance = TRUE,
  metric = "AUC"
  )
)
```

    ##    user  system elapsed 
    ##   5.765   0.005   5.797

``` r
bag_fit_cv
```

    ## Bagged CART 
    ## 
    ## 982 samples
    ##  14 predictor
    ##   3 classes: 'Jason_Mraz', 'Maroon_5', 'Queen' 
    ## 
    ## Recipe steps: nzv, center, scale 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 884, 884, 884, 884, 882, 884, ... 
    ## Addtional sampling using SMOTE
    ## 
    ## Resampling results:
    ## 
    ##   logLoss    AUC       prAUC      Accuracy   Kappa      Mean_F1  
    ##   0.9524717  0.924093  0.6884414  0.7994119  0.6557456  0.7538429
    ##   Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value  Mean_Neg_Pred_Value
    ##   0.7792334         0.8941912         0.7646892            0.8875013          
    ##   Mean_Precision  Mean_Recall  Mean_Detection_Rate  Mean_Balanced_Accuracy
    ##   0.7646892       0.7792334    0.2664706            0.8367123

Proses *training* model berlangsung selama 18.209 detik dengan rata-rata
**AUC** yang diperoleh sebesar ‘r bag\_fit\_cv$results %\>% select(AUC)
%\>% pull()’. Nilai ini merupakan peningkatan dari model *decision
trees* yang telah dibuat sebelumnya.

``` r
bag_roc <- bag_fit_cv$results %>%
  arrange(-AUC) %>%
  slice(1) %>%
  select(AUC) %>% pull()

bag_roc
```

    ## [1] 0.924093

## Model Akhir

Pada tahapan ini, model yang telah di-*training*, diekstrak model
akhirnya.

``` r
bag_fit <- bag_fit_cv$finalModel
```

Adapun performa model bagging pada data baru dapat dicek dengan mengukur
nilai **Akurasi** model menggunakan data *test*.

``` r
pred_test <- predict(bag_fit, baked_test, type = "class")

## RMSE
cm <- confusionMatrix(pred_test, baked_test$artist)
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##             Reference
    ## Prediction   Jason_Mraz Maroon_5 Queen
    ##   Jason_Mraz         31       21    50
    ##   Maroon_5            2       43     7
    ##   Queen               2        1    86
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6584          
    ##                  95% CI : (0.5951, 0.7179)
    ##     No Information Rate : 0.5885          
    ##     P-Value [Acc > NIR] : 0.01508         
    ##                                           
    ##                   Kappa : 0.4877          
    ##                                           
    ##  Mcnemar's Test P-Value : 6.406e-14       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Jason_Mraz Class: Maroon_5 Class: Queen
    ## Sensitivity                     0.8857          0.6615       0.6014
    ## Specificity                     0.6587          0.9494       0.9700
    ## Pos Pred Value                  0.3039          0.8269       0.9663
    ## Neg Pred Value                  0.9716          0.8848       0.6299
    ## Prevalence                      0.1440          0.2675       0.5885
    ## Detection Rate                  0.1276          0.1770       0.3539
    ## Detection Prevalence            0.4198          0.2140       0.3663
    ## Balanced Accuracy               0.7722          0.8055       0.7857

Berdasarkan hasil evaluasi diperoleh nilai akurasi sebesar 0.6584362

## Interpretasi Fitur

Untuk melakukan interpretasi terhadap fitur paling berpengaruh dalam
model bagging, kita dapat emngetahuinya melalui *varibale importance
plot*.

``` r
vi <- vip(bag_fit_cv, num_features = 10)
vi
```

![](temp_files/figure-gfm/bag-vip-1.png)<!-- -->

Berdasarkan hasil plot, terdapat empat buah variabel paling berpengaruh,
yaitu: danceability, duration\_ms, popularity, acousticness. Untuk
melihat efek dari keempat variabel tersebut terhadap prediksi yang
dihasilkan model, kita dapat mengetahuinya melalui *patial plot
dependencies*.

``` r
p1 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[1]) %>% 
  autoplot() 

p2 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[2]) %>% 
  autoplot()

p3 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[3]) %>% 
  autoplot()
  

p4 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[4]) %>% 
  autoplot()

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

![](temp_files/figure-gfm/bag-pdp-1.png)<!-- -->

# Random Forest

Bagging (agregasi bootstrap) adalah teknik yang dapat mengubah model
pohon tunggal dengan varian tinggi dan kemampuan prediksi yang buruk
menjadi fungsi prediksi yang cukup akurat. Sayangnya, bagging biasanya
kekurangan, yiatu: adanya korelasi pada tiap pohon yang mengurangi
kinerja keseluruhan model. *Random forest* adalah modifikasi bagging
yang membangun koleksi besar pohon yang tidak berkorelasi dan telah
menjadi algoritma pembelajaran “out-of-the-box” yang sangat populer yang
dengan kinerja prediksi yang baik.

*Random forest* dibangun di atas prinsip-prinsip dasar yang sama seperti
*decision tress* dan bagging. Bagging memperkenalkan komponen acak ke
dalam proses pembangunan pohon yang mengurangi varian prediksi pohon
tunggal dan meningkatkan kinerja prediksi. Namun, pohon-pohon di bagging
tidak sepenuhnya independen satu sama lain karena semua prediktor asli
dianggap di setiap split setiap pohon. Sebaliknya, pohon dari sampel
bootstrap yang berbeda biasanya memiliki struktur yang mirip satu sama
lain (terutama di bagian atas pohon) karena hubungan yang mendasarinya.

Sebagai contoh, jika kita membuat enam pohon keputusan dengan sampel
bootstrap data perumahan Boston yang berbeda, kita melihat bahwa puncak
pohon semua memiliki struktur yang sangat mirip. Meskipun ada 15
variabel prediktor untuk dipecah, keenam pohon memiliki kedua variabel
lstat dan rm yang mendorong beberapa split pertama.

Sebagai contoh, jika kita membuat enam *decision trees* dengan sampel
bootstrap [data perumahan
Boston](http://uc-r.github.io/\(http://lib.stat.cmu.edu/datasets/boston\))
yang berbeda, kita melihat bahwa puncak pohon semua memiliki struktur
yang sangat mirip. Meskipun ada 15 variabel prediktor untuk dipecah,
keenam pohon memiliki kedua variabel `lstat` dan `rm` yang mendorong
beberapa split pertama.

![Enam decision trees berdasarkan sampel bootsrap yang
berbeda-beda](http://uc-r.github.io/public/images/analytics/random_forests/tree-correlation-1.png)

Karakteristik ini dikenal sebagai **korelasi pohon** dan mencegah
bagging dari secara optimal mengurangi varians dari nilai-nilai
prediktif. Untuk mengurangi varian lebih lanjut, kita perlu meminimalkan
jumlah korelasi antar pohon-pohon tersebut. Ini bisa dicapai dengan
menyuntikkan lebih banyak keacakan ke dalam proses penanaman pohon.
*Random Forest* mencapai ini dalam dua cara:

1.  **Bootstrap**: mirip dengan bagging, setiap pohon ditumbuhkan ke set
    data *bootstrap resampled*, yang membuatnya berbeda dan agak
    mendekorelasi antar pohon tersebut.
2.  **Split-variable randomization**: setiap kali pemisahan dilakukan,
    pencarian untuk variabel terbagi terbatas pada subset acak \(m\)
    dari variabel \(p\). Untuk pohon regresi, nilai default tipikal
    adalah \(m = p/3\) tetapi ini harus dianggap sebagai *parameter
    tuning*. Ketika \(m = p\), jumlah pengacakan hanya menggunakan
    langkah 1 dan sama dengan bagging.

Algoritma dasar dari *random forest* adalah sebagai berikut:

    1.  Diberikan set data training
    2.  Pilih jumlah pohon yang akan dibangun (n_trees)
    3.  for i = 1 to n_trees do
    4.  | Hasilkan sampel bootstrap dari data asli
    5.  | Tumbuhkan pohon regresi / klasifikasi ke data yang di-bootstrap
    6.  | for each split do
    7.  | | Pilih variabel m_try secara acak dari semua variabel p
    8.  | | Pilih variabel / titik-split terbaik di antara m_try
    9.  | | Membagi node menjadi dua node anak
    10. | end
    11. | Gunakan kriteria berhenti model pohon biasa untuk menentukan 
        | kapan pohon selesai (tapi jangan pangkas)
    12. end
    13. Output ensemble of trees 

Karena algoritma secara acak memilih sampel bootstrap untuk dilatih dan
prediktor digunakan pada setiap split, korelasi pohon akan berkurang
melebihi bagging.

## OOB Error vs Test Set Error

Mirip dengan bagging, manfaat alami dari proses *bootstrap resampling*
adalah *randomforest* memiliki sampel *out-of-bag* (OOB) yang memberikan
perkiraan kesalahan pengujian yang efisien dan masuk akal. Ini
memberikan satu set validasi bawaan tanpa kerja ekstra , dan kita tidak
perlu mengorbankan data *training* apa pun untuk digunakan untuk
validasi. Ini membuat proses identifikasi jumlah pohon yang diperlukan
untuk menstabilkan tingkat kesalahan selama proses *tuning* menjadi
lebih efisien; Namun, seperti yang diilustrasikan di bawah ini, beberapa
perbedaan antara kesalahan OOB dan kesalahan tes diharapkan.

![Random forest OOB vs validation error (Sumber:
<http://uc-r.github.io/>)](http://uc-r.github.io/public/images/analytics/random_forests/oob-error-compare-1.svg)

Selain itu, banyak paket tidak melacak pengamatan mana yang merupakan
bagian dari sampel OOB untuk pohon tertentu dan yang tidak. Jika kita
membandingkan beberapa model dengan yang lain, kita ingin membuat skor
masing-masing pada set validasi yang sama untuk membandingkan kinerja.
Selain itu, meskipun secara teknis dimungkinkan untuk menghitung metrik
tertentu seperti *root mean squared logarithmic error* (RMSLE) pada
sampel OOB, itu tidak dibangun untuk semua paket. Jadi jika kita ingin
membandingkan beberapa model atau menggunakan fungsi *loss* yang sedikit
lebih tradisional, kita mungkin ingin tetap melakukan validasi silang.

## Kelebihan dan Kekurangan

**Kelbihan**

  - Biasanya memiliki kinerja yang sangat bagus
  - “*Out-of-the-box*” yang luar biasa bagus - sangat sedikit
    penyesuaian yang diperlukan
  - Kumpulan validasi bawaan - tidak perlu mengorbankan data untuk
    validasi tambahan
  - Tidak diperlukan pra-pemrosesan
  - Bersifat *robust* dengan adanya *outlier*

**Kekurangan**

  - Dapat menjadi lambat pada set data besar
  - Meskipun akurat, seringkali tidak dapat bersaing dengan algoritma
    *boosting*
  - Kurang mudah untuk ditafsirkan

## Validasi Silang dan Parameter Tuning

Pada fungsi `trainControl()` argumen yang digunakan sama dengan model
bagging.

``` r
# spesifikasi metode validasi silang
cv <- trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 10, 
  # repeats = 5,
  classProbs = TRUE,
  sampling = "smote",
  summaryFunction = multiClassSummary,
  savePredictions = TRUE,
  allowParallel = TRUE
)
```

``` r
n_features <- length(setdiff(names(baked_train), "artist"))
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10),
  splitrule = c("gini", "extratrees", "hellinger")
)
```

Pada proses training, `method` yang digunakan adalah `rf` atau *random
forest*. Metode ini memerlukan sejumlah paket tambahan untuk memastikan
proses parallel dapat berjalan, seperti: `e1071` dan \`randomForest.

``` r
# membuat model
system.time(
rf_fit_cv <- train(
  blueprint,
  data = spotify_train,
  method = "ranger",
  trControl = cv,
  tuneGrid =  hyper_grid,
  importance = "impurity",
  keep.inbag=TRUE,
  metric = "AUC"
  )
)
```

    ##    user  system elapsed 
    ## 372.950   7.580 382.298

``` r
rf_fit_cv
```

    ## Random Forest 
    ## 
    ## 982 samples
    ##  14 predictor
    ##   3 classes: 'Jason_Mraz', 'Maroon_5', 'Queen' 
    ## 
    ## Recipe steps: nzv, center, scale 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 885, 883, 885, 883, 883, 885, ... 
    ## Addtional sampling using SMOTE
    ## 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  min.node.size  splitrule   logLoss    AUC        prAUC      Accuracy 
    ##   0      1             gini        0.4794823  0.9481178  0.8333031  0.8421375
    ##   0      1             extratrees  0.5366971  0.9502678  0.8490667  0.8277890
    ##   0      1             hellinger         NaN        NaN        NaN        NaN
    ##   0      3             gini        0.4886153  0.9459830  0.8379260  0.8319231
    ##   0      3             extratrees  0.5583821  0.9437133  0.8350860  0.8083678
    ##   0      3             hellinger         NaN        NaN        NaN        NaN
    ##   0      5             gini        0.4937870  0.9436014  0.8336056  0.8370249
    ##   0      5             extratrees  0.5659671  0.9428149  0.8389854  0.8032553
    ##   0      5             hellinger         NaN        NaN        NaN        NaN
    ##   0     10             gini        0.5044289  0.9435061  0.8281872  0.8268608
    ##   0     10             extratrees  0.5803382  0.9427956  0.8304458  0.8063998
    ##   0     10             hellinger         NaN        NaN        NaN        NaN
    ##   2      1             gini        0.5021448  0.9455042  0.8362679  0.8319218
    ##   2      1             extratrees  0.5596993  0.9518371  0.8475464  0.8185126
    ##   2      1             hellinger         NaN        NaN        NaN        NaN
    ##   2      3             gini        0.5035241  0.9456244  0.8366873  0.8452614
    ##   2      3             extratrees  0.5702178  0.9514157  0.8543368  0.8062860
    ##   2      3             hellinger         NaN        NaN        NaN        NaN
    ##   2      5             gini        0.5054336  0.9461133  0.8381754  0.8309025
    ##   2      5             extratrees  0.5839377  0.9469089  0.8436534  0.8114612
    ##   2      5             hellinger         NaN        NaN        NaN        NaN
    ##   2     10             gini        0.5273785  0.9413954  0.8259714  0.8247268
    ##   2     10             extratrees  0.6238421  0.9381897  0.8298448  0.7921339
    ##   2     10             hellinger         NaN        NaN        NaN        NaN
    ##   3      1             gini        0.4888013  0.9452812  0.8366163  0.8370144
    ##   3      1             extratrees  0.5381274  0.9476926  0.8459294  0.8206552
    ##   3      1             hellinger         NaN        NaN        NaN        NaN
    ##   3      3             gini        0.4893802  0.9442788  0.8308252  0.8412002
    ##   3      3             extratrees  0.5381185  0.9510898  0.8502458  0.8286544
    ##   3      3             hellinger         NaN        NaN        NaN        NaN
    ##   3      5             gini        0.4962615  0.9424093  0.8267435  0.8360356
    ##   3      5             extratrees  0.5562984  0.9470825  0.8377935  0.8247895
    ##   3      5             hellinger         NaN        NaN        NaN        NaN
    ##   3     10             gini        0.4954340  0.9467935  0.8374577  0.8360983
    ##   3     10             extratrees  0.5791961  0.9458621  0.8402317  0.8104608
    ##   3     10             hellinger         NaN        NaN        NaN        NaN
    ##   4      1             gini        0.4855083  0.9427252  0.8317445  0.8442629
    ##   4      1             extratrees  0.5139795  0.9512547  0.8494923  0.8256752
    ##   4      1             hellinger         NaN        NaN        NaN        NaN
    ##   4      3             gini        0.4771600  0.9460853  0.8361318  0.8370257
    ##   4      3             extratrees  0.5311890  0.9476350  0.8441551  0.8186146
    ##   4      3             hellinger         NaN        NaN        NaN        NaN
    ##   4      5             gini        0.4808501  0.9419690  0.8269197  0.8422311
    ##   4      5             extratrees  0.5298629  0.9490284  0.8487567  0.8155326
    ##   4      5             hellinger         NaN        NaN        NaN        NaN
    ##   4     10             gini        0.4987367  0.9415760  0.8277502  0.8298707
    ##   4     10             extratrees  0.5614743  0.9440651  0.8374085  0.8115435
    ##   4     10             hellinger         NaN        NaN        NaN        NaN
    ##   5      1             gini        0.4649802  0.9445386  0.8271398  0.8442210
    ##   5      1             extratrees  0.5152213  0.9499899  0.8417291  0.8288709
    ##   5      1             hellinger         NaN        NaN        NaN        NaN
    ##   5      3             gini        0.4812993  0.9424884  0.8250025  0.8309841
    ##   5      3             extratrees  0.5208553  0.9471917  0.8436161  0.8430344
    ##   5      3             hellinger         NaN        NaN        NaN        NaN
    ##   5      5             gini        0.4829729  0.9414926  0.8283030  0.8299020
    ##   5      5             extratrees  0.5215773  0.9458559  0.8364645  0.8420344
    ##   5      5             hellinger         NaN        NaN        NaN        NaN
    ##   5     10             gini        0.4806856  0.9423449  0.8244151  0.8442725
    ##   5     10             extratrees  0.5461673  0.9450957  0.8413138  0.8248099
    ##   5     10             hellinger         NaN        NaN        NaN        NaN
    ##   Kappa      Mean_F1    Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value
    ##   0.7248842  0.7992157  0.8213455         0.9162998         0.8038114          
    ##   0.7054713  0.7863283  0.8202997         0.9125200         0.8020036          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7109291  0.7899188  0.8218028         0.9138380         0.7983459          
    ##   0.6707365  0.7645668  0.7967905         0.8993077         0.7879385          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7163382  0.7929412  0.8169961         0.9133217         0.8035494          
    ##   0.6657195  0.7596192  0.7943056         0.8997259         0.7802036          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7009306  0.7815557  0.8073435         0.9093834         0.7953535          
    ##   0.6675615  0.7586870  0.7917357         0.9000590         0.7792752          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7096179  0.7883840  0.8146153         0.9122443         0.8005815          
    ##   0.6919710  0.7771561  0.8159423         0.9098664         0.7919078          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7299509  0.8014419  0.8254102         0.9170991         0.8145158          
    ##   0.6703278  0.7586424  0.7952294         0.9026835         0.7847949          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7079781  0.7876054  0.8159684         0.9121344         0.7979082          
    ##   0.6805690  0.7664435  0.8053269         0.9068237         0.7893771          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.6958515  0.7780745  0.8053726         0.9072909         0.7954569          
    ##   0.6461623  0.7447699  0.7816964         0.8941895         0.7733578          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7169284  0.7938602  0.8208980         0.9141317         0.8047087          
    ##   0.6917042  0.7777253  0.8083025         0.9067641         0.7960451          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7243783  0.7996130  0.8258529         0.9163665         0.8122153          
    ##   0.7055504  0.7864391  0.8165546         0.9119442         0.8016495          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7156094  0.7938785  0.8209731         0.9132273         0.8028424          
    ##   0.7009696  0.7820854  0.8161661         0.9119836         0.7969911          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7141399  0.7950246  0.8207825         0.9113972         0.8073984          
    ##   0.6752812  0.7654795  0.7977839         0.9019034         0.7829287          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7302906  0.8017066  0.8292701         0.9195763         0.8094912          
    ##   0.7010805  0.7829827  0.8136048         0.9102115         0.7979791          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7174268  0.7937322  0.8200573         0.9147617         0.8002165          
    ##   0.6901041  0.7753549  0.8095554         0.9080277         0.7880562          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7236749  0.7965781  0.8146903         0.9151807         0.8060925          
    ##   0.6854685  0.7742180  0.8054982         0.9053512         0.7886684          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7051050  0.7875371  0.8095669         0.9088031         0.7935146          
    ##   0.6775641  0.7657010  0.7986744         0.9038305         0.7841197          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7294959  0.8024428  0.8256941         0.9177004         0.8087622          
    ##   0.7031549  0.7839591  0.8109236         0.9096259         0.8034700          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7066916  0.7877451  0.8127664         0.9100622         0.7928394          
    ##   0.7299285  0.8044317  0.8311130         0.9184093         0.8173588          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7054376  0.7872904  0.8129328         0.9106051         0.7931854          
    ##   0.7257147  0.7991505  0.8246422         0.9166594         0.8116021          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   0.7274834  0.8035281  0.8227314         0.9148123         0.8114506          
    ##   0.6992451  0.7806037  0.8116356         0.9101558         0.7966795          
    ##         NaN        NaN        NaN               NaN               NaN          
    ##   Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall  Mean_Detection_Rate
    ##   0.9112709            0.8038114       0.8213455    0.2807125          
    ##   0.9036719            0.8020036       0.8202997    0.2759297          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9051604            0.7983459       0.8218028    0.2773077          
    ##   0.8935514            0.7879385       0.7967905    0.2694559          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9092265            0.8035494       0.8169961    0.2790083          
    ##   0.8904663            0.7802036       0.7943056    0.2677518          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9041116            0.7953535       0.8073435    0.2756203          
    ##   0.8927701            0.7792752       0.7917357    0.2687999          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9063940            0.8005815       0.8146153    0.2773073          
    ##   0.8984162            0.7919078       0.8159423    0.2728375          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9157257            0.8145158       0.8254102    0.2817538          
    ##   0.8941319            0.7847949       0.7952294    0.2687620          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9052611            0.7979082       0.8159684    0.2769675          
    ##   0.8961165            0.7893771       0.8053269    0.2704871          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9035922            0.7954569       0.8053726    0.2749089          
    ##   0.8847696            0.7733578       0.7816964    0.2640446          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9089670            0.8047087       0.8208980    0.2790048          
    ##   0.8994418            0.7960451       0.8083025    0.2735517          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9113469            0.8122153       0.8258529    0.2804001          
    ##   0.9045193            0.8016495       0.8165546    0.2762181          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9082623            0.8028424       0.8209731    0.2786785          
    ##   0.9019780            0.7969911       0.8161661    0.2749298          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9082466            0.8073984       0.8207825    0.2786994          
    ##   0.8948104            0.7829287       0.7977839    0.2701536          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9125163            0.8094912       0.8292701    0.2814210          
    ##   0.9029558            0.7979791       0.8136048    0.2752251          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9082836            0.8002165       0.8200573    0.2790086          
    ##   0.8983752            0.7880562       0.8095554    0.2728715          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9133173            0.8060925       0.8146903    0.2807437          
    ##   0.8960908            0.7886684       0.8054982    0.2718442          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9042783            0.7935146       0.8095669    0.2766236          
    ##   0.8951817            0.7841197       0.7986744    0.2705145          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9126307            0.8087622       0.8256941    0.2814070          
    ##   0.9057037            0.8034700       0.8109236    0.2762903          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9045055            0.7928394       0.8127664    0.2769947          
    ##   0.9132270            0.8173588       0.8311130    0.2810115          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9038422            0.7931854       0.8129328    0.2766340          
    ##   0.9121257            0.8116021       0.8246422    0.2806781          
    ##         NaN                  NaN             NaN          NaN          
    ##   0.9123681            0.8114506       0.8227314    0.2814242          
    ##   0.9028689            0.7966795       0.8116356    0.2749366          
    ##         NaN                  NaN             NaN          NaN          
    ##   Mean_Balanced_Accuracy
    ##   0.8688227             
    ##   0.8664099             
    ##         NaN             
    ##   0.8678204             
    ##   0.8480491             
    ##         NaN             
    ##   0.8651589             
    ##   0.8470157             
    ##         NaN             
    ##   0.8583634             
    ##   0.8458973             
    ##         NaN             
    ##   0.8634298             
    ##   0.8629043             
    ##         NaN             
    ##   0.8712547             
    ##   0.8489565             
    ##         NaN             
    ##   0.8640514             
    ##   0.8560753             
    ##         NaN             
    ##   0.8563318             
    ##   0.8379429             
    ##         NaN             
    ##   0.8675149             
    ##   0.8575333             
    ##         NaN             
    ##   0.8711097             
    ##   0.8642494             
    ##         NaN             
    ##   0.8671002             
    ##   0.8640749             
    ##         NaN             
    ##   0.8660898             
    ##   0.8498437             
    ##         NaN             
    ##   0.8744232             
    ##   0.8619081             
    ##         NaN             
    ##   0.8674095             
    ##   0.8587916             
    ##         NaN             
    ##   0.8649355             
    ##   0.8554247             
    ##         NaN             
    ##   0.8591850             
    ##   0.8512524             
    ##         NaN             
    ##   0.8716972             
    ##   0.8602748             
    ##         NaN             
    ##   0.8614143             
    ##   0.8747612             
    ##         NaN             
    ##   0.8617689             
    ##   0.8706508             
    ##         NaN             
    ##   0.8687718             
    ##   0.8608957             
    ##         NaN             
    ## 
    ## AUC was used to select the optimal model using the largest value.
    ## The final values used for the model were mtry = 2, splitrule = extratrees
    ##  and min.node.size = 1.

Proses *training* berlangsung selama 370.509 detik dengan 11 model
terbentuk. Dari seluruh model tersebut, model dengan parameter `mtry` =
2, `splitrule` = `rf_fit_cv$bestTune[1,2]`, dan `min.node.size` = 1
memiliki rata-rata **AUC** yang paling baik. Untuk dapat mengakses
**AUC** model terbaik, jalankan sintaks berikut:

``` r
rf_roc <- rf_fit_cv$results %>%
  arrange(-AUC) %>%
  slice(1) %>%
  select(AUC) %>% pull()

rf_roc
```

    ## [1] 0.9518371

Nilai **UUC** model *random forest* yang dihasilkan jauh lebih baik
dibandingkan dua model awal. Reduksi terhadap jumlah pohon yang saling
berkorelasi telah meningkatkan performa model secara signifikan.

Berikut adalah ringkasan performa masing-masing model:

``` r
# visualisasi
ggplot(rf_fit_cv)
```

![](temp_files/figure-gfm/rf-vis-1.png)<!-- -->

## Model Akhir

Untuk mengekstrak model final, jalankan sintaks berikut:

``` r
rf_fit <- rf_fit_cv$finalModel
```

Adapun performa model bagging pada data baru dapat dicek dengan mengukur
nilai **Akurasi** model menggunakan data *test*.

``` r
pred_test <- predict(rf_fit, baked_test)

pred_test <-
  as.data.frame(pred_test$predictions) %>%
  rowid_to_column("row") %>%
  pivot_longer(cols = Jason_Mraz:Queen, names_to = "artist", values_to = "prob") %>%
  group_by(row) %>%
  summarise(prediction = which.max(prob)) %>%
  mutate(prediction = factor(prediction, 
                             labels = c("Jason_Mraz", "Maroon_5", "Queen" ))) %>%
  select(prediction) %>%
  pull()

## RMSE
cm <- confusionMatrix(pred_test, baked_test$artist, mode='everything')
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##             Reference
    ## Prediction   Jason_Mraz Maroon_5 Queen
    ##   Jason_Mraz         30       18    44
    ##   Maroon_5            3       45     1
    ##   Queen               2        2    98
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7119         
    ##                  95% CI : (0.6506, 0.768)
    ##     No Information Rate : 0.5885         
    ##     P-Value [Acc > NIR] : 4.405e-05      
    ##                                          
    ##                   Kappa : 0.5531         
    ##                                          
    ##  Mcnemar's Test P-Value : 1.075e-10      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Jason_Mraz Class: Maroon_5 Class: Queen
    ## Sensitivity                     0.8571          0.6923       0.6853
    ## Specificity                     0.7019          0.9775       0.9600
    ## Pos Pred Value                  0.3261          0.9184       0.9608
    ## Neg Pred Value                  0.9669          0.8969       0.6809
    ## Precision                       0.3261          0.9184       0.9608
    ## Recall                          0.8571          0.6923       0.6853
    ## F1                              0.4724          0.7895       0.8000
    ## Prevalence                      0.1440          0.2675       0.5885
    ## Detection Rate                  0.1235          0.1852       0.4033
    ## Detection Prevalence            0.3786          0.2016       0.4198
    ## Balanced Accuracy               0.7795          0.8349       0.8227

Berdasarkan hasil evaluasi diperoleh nilai akurasi sebesar 0.7119342

## Interpretasi Fitur

Untuk mengetahui variabel apa yang paling berpengaruh terhadap performa
model, kita dapat menggunakan visualisasi *variabel importance plot*.

``` r
vi <- vip(rf_fit, num_features = 10)
vi
```

![](temp_files/figure-gfm/rf-vip-1.png)<!-- -->

Berdasarkan hasil plot, terdapat empat buah variabel paling berpengaruh,
yaitu: danceability, loudness, acousticness, popularity. Untuk melihat
efek dari keempat variabel tersebut terhadap prediksi yang dihasilkan
model, kita dapat mengetahuinya melalui *patial plot dependencies*.

``` r
p1 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[1]) %>% 
  autoplot() 

p2 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[2]) %>% 
  autoplot()

p3 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[3]) %>% 
  autoplot()
  

p4 <- pdp::partial(dt_fit_cv, pred.var = vi$data %>% select(Variable) %>% pull() %>%.[4]) %>% 
  autoplot()

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

![](temp_files/figure-gfm/rf-pdp-1.png)<!-- -->

Berdasarkan output yang dihasilkan, ketiga variabel memiliki relasi
non-linier terhadap variabel target.
