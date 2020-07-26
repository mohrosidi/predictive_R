K-Nearest Neighbors
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
menggunakan *k-nearest neighbors*. Paket-paket yang digunakan ditampilkan sebagai
berikut:

``` r
# library pembantu
library(DMwR)
library(modeldata)
library(MLmetrics)
library(tidyverse)
library(rsample)
library(recipes)
library(DataExplorer)
library(skimr)
library(magrittr)

# library model
library(caret)

# paket penjelasan model
library(vip)
library(pdp)
```

**Paket Pembantu**

1.  `foreach` : paket untuk melakukan *parallel computing*. Diperlukan
    untuk melakukan *fitting* model *parallel random forest*
2.  `import` : paket yang menangani *dependency* fungsi antar paket
    dalam proses *fitting* model *parallel random forest*
3.  `tidyverse` : kumpulan paket dalam bidang data science
4.  `rsample` : membantu proses *data splitting*
5.  `recipes`: membantu proses data pra-pemrosesan
6.  `DataExplorer` : EDA
7.  `skimr` : membuat ringkasan data
8. `DMwR` : paket untuk melakukan sampling “smote”
9. `modeldata` : kumpulan dataset untuk membuat model *machine
    learning*

**Paket untuk Membangun Model**

1.  `caret` : berisikan sejumlah fungsi yang dapat merampingkan proses
    pembuatan model regresi dan klasifikasi

**Paket Interpretasi Model**

2.  `vip` : visualisasi *variable importance*
3.  `pdp` : visualisasi plot ketergantungan parsial

## Import Dataset

Import dataset dilakukan dengan menggunakan fungsi `readr()`. Fungsi ini
digunakan untuk membaca file dengan ekstensi `.csv`.

``` r
spotify <- read_csv("data/spotify.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double(),
    ##   id = col_character(),
    ##   name = col_character(),
    ##   album.id = col_character(),
    ##   album.name = col_character(),
    ##   artist = col_character()
    ## )

    ## See spec(...) for full column specifications.

``` r
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
  step_scale(all_numeric(), -all_outcomes()) %>%
  
  # 5. dummy encoding
  step_dummy(all_nominal(), -all_outcomes())

blueprint
```

    ## Data Recipe
    ## 
    ## Inputs:
    ## 
    ##       role #variables
    ##    outcome          1
    ##  predictor         14
    ## 
    ## Operations:
    ## 
    ## Sparse, unbalanced variable filter on all_nominal()
    ## Centering for all_numeric(), -all_outcomes()
    ## Scaling for all_numeric(), -all_outcomes()
    ## Dummy variables from all_nominal(), -all_outcomes()

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
    ## Dummy variables from key, mode, time_signature [trained]

Langkah terakhir adalah mengaplikasikan *blueprint* pada data *training*
dan *test* menggunakan fungsi `bake()`.

``` r
baked_train <- bake(prepare, new_data = spotify_train)
baked_test <- bake(prepare, new_data = spotify_test)
baked_train
```

    ## # A tibble: 982 x 28
    ##    popularity duration_ms danceability  energy loudness speechiness acousticness
    ##         <dbl>       <dbl>        <dbl>   <dbl>    <dbl>       <dbl>        <dbl>
    ##  1       1.74      0.0449        0.161 -0.173    0.669       0.471        -0.498
    ##  2       3.19     -0.288         1.61  -0.220    0.821      -0.234        -0.668
    ##  3       2.47     -0.365         0.844 -0.194    0.848      -0.270        -0.608
    ##  4       1.74     -0.319         1.39  -0.190    0.461      -0.300        -0.449
    ##  5       1.82     -0.341         2.32  -0.360    0.897      -0.216         0.682
    ##  6       1.67     -0.426         1.68   0.0904   1.08       -0.0444       -0.339
    ##  7       1.74     -0.200         0.574 -1.03     0.0820     -0.243         1.65 
    ##  8       2.76     -0.160         0.611 -0.793    0.369      -0.211         1.03 
    ##  9       1.67      3.80          1.34   0.698    0.455      -0.173        -0.300
    ## 10       1.67     -0.0152        1.51  -0.0880   0.772      -0.196        -0.378
    ## # … with 972 more rows, and 21 more variables: instrumentalness <dbl>,
    ## #   liveness <dbl>, valence <dbl>, tempo <dbl>, artist <fct>, key_c. <dbl>,
    ## #   key_d <dbl>, key_d. <dbl>, key_e <dbl>, key_f <dbl>, key_f. <dbl>,
    ## #   key_g <dbl>, key_g. <dbl>, key_a <dbl>, key_a. <dbl>, key_b <dbl>,
    ## #   mode_major <dbl>, time_signature_X1 <dbl>, time_signature_X3 <dbl>,
    ## #   time_signature_X4 <dbl>, time_signature_X5 <dbl>

# K-Nearest Neighbors

K-nearest Neighbors (KNN) adalah algoritma yang sangat sederhana di mana
setiap pengamatan diprediksi berdasarkan “kesamaan” dengan pengamatan
lainnya. Tidak seperti kebanyakan metode lainnya, KNN adalah algoritma
berbasis memori dan tidak dapat diringkas oleh model bentuk tertutup.
Ini berarti sampel pelatihan diperlukan saat run-time dan prediksi
dibuat langsung dari hubungan sampel. Akibatnya, KNN juga dikenal
sebagai *lazy learners* (Cunningham dan Delany 2007) dan dapat menjadi
tidak efisien secara komputasi. Namun, KNN telah berhasil dalam sejumlah
besar masalah bisnis (Jiang et al. (2012) dan Mccord dan Chuah (2011))
dan berguna untuk tujuan preprocessing juga.

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
  sampling = "smote",
  savePredictions = TRUE,
  summaryFunction = multiClassSummary,
  allowParallel = TRUE
)
```

Selanjutnya spesifikasikan *hyperparameter* yang akan di *tuning*.

``` r
## Construct grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 30, by = 1))
```

Setelah parameter *tuning* dan validasi silang dispesifikasikan, proses
training dilakukan menggunakan fungsi `train()`.

``` r
system.time(
knn_fit_cv <- train(
  blueprint, 
  data = spotify_train, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "AUC"
  )
)
```

    ##    user  system elapsed 
    ##  62.794   0.109  63.116

``` r
knn_fit_cv
```

    ## k-Nearest Neighbors 
    ## 
    ## 982 samples
    ##  14 predictor
    ##   3 classes: 'Jason_Mraz', 'Maroon_5', 'Queen' 
    ## 
    ## Recipe steps: nzv, center, scale, dummy 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 884, 883, 884, 884, 884, 884, ... 
    ## Addtional sampling using SMOTE
    ## 
    ## Resampling results across tuning parameters:
    ## 
    ##   k   logLoss    AUC        prAUC      Accuracy   Kappa      Mean_F1  
    ##    2  5.9450947  0.8466467  0.2304105  0.7331890  0.5536478  0.6796629
    ##    3  4.2813125  0.8732723  0.2956495  0.7423727  0.5754221  0.6981440
    ##    4  3.7026185  0.8805061  0.3596728  0.7332303  0.5599883  0.6891707
    ##    5  2.6954788  0.8948392  0.4225110  0.7617708  0.6032514  0.7157992
    ##    6  2.6083176  0.8955824  0.4343281  0.7393218  0.5705255  0.6947200
    ##    7  2.1678041  0.8957942  0.4852156  0.7484848  0.5855877  0.7051669
    ##    8  1.7764140  0.8994642  0.5378956  0.7658009  0.6104367  0.7183302
    ##    9  1.7645701  0.9079446  0.5809065  0.7596784  0.6037733  0.7164141
    ##   10  1.6399957  0.9023827  0.5806076  0.7331684  0.5646905  0.6880916
    ##   11  1.5767864  0.9043929  0.5800930  0.7465059  0.5859096  0.7046755
    ##   12  1.2206414  0.9061173  0.6128320  0.7321995  0.5631060  0.6899915
    ##   13  1.1226609  0.9106279  0.6523238  0.7434756  0.5813481  0.7008840
    ##   14  1.3906658  0.9067177  0.6775237  0.7454752  0.5822916  0.7026907
    ##   15  1.0725317  0.9056320  0.6626033  0.7362503  0.5688089  0.6927120
    ##   16  1.2400677  0.9097145  0.6982306  0.7525768  0.5932544  0.7080590
    ##   17  0.9866190  0.9082583  0.6897656  0.7352917  0.5696781  0.6908481
    ##   18  0.9884073  0.9108349  0.7159577  0.7342713  0.5660011  0.6891967
    ##   19  1.0153053  0.9110854  0.7118721  0.7383426  0.5756641  0.6986662
    ##   20  0.8974295  0.9140258  0.7127405  0.7423933  0.5853608  0.7020461
    ##   21  1.0192702  0.9057891  0.7148527  0.7271284  0.5575675  0.6841170
    ##   22  0.9480304  0.9078765  0.7311933  0.7281901  0.5589819  0.6845760
    ##   23  1.0797047  0.9057715  0.7444497  0.7342610  0.5677365  0.6902012
    ##   24  0.8776484  0.9126304  0.7388395  0.7220058  0.5493603  0.6757614
    ##   25  0.8631305  0.9076989  0.7338651  0.7424449  0.5799965  0.6952424
    ##   26  0.8042807  0.9081365  0.7543258  0.7342816  0.5713736  0.6919502
    ##   27  0.8133787  0.9038759  0.7492799  0.7158730  0.5410236  0.6673118
    ##   28  0.8798540  0.9076606  0.7576502  0.7260874  0.5601303  0.6837455
    ##   29  0.8660805  0.9141198  0.7614201  0.7108122  0.5362161  0.6634030
    ##   30  0.9605787  0.9018690  0.7399189  0.6995980  0.5194024  0.6540571
    ##   Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value  Mean_Neg_Pred_Value
    ##   0.7121148         0.8660676         0.6888074            0.8526176          
    ##   0.7423709         0.8753158         0.6975022            0.8572013          
    ##   0.7353864         0.8688678         0.7032759            0.8526326          
    ##   0.7529246         0.8833966         0.7212088            0.8666145          
    ##   0.7413639         0.8732897         0.7083603            0.8565545          
    ##   0.7529297         0.8792117         0.7107342            0.8607021          
    ##   0.7610808         0.8878969         0.7212406            0.8700224          
    ##   0.7660595         0.8851132         0.7308221            0.8678912          
    ##   0.7409603         0.8738510         0.7078471            0.8556728          
    ##   0.7552226         0.8812802         0.7222299            0.8612435          
    ##   0.7413794         0.8728690         0.7156829            0.8542908          
    ##   0.7520806         0.8793828         0.7231815            0.8605142          
    ##   0.7513997         0.8777727         0.7261190            0.8609688          
    ##   0.7453788         0.8746003         0.7194978            0.8570568          
    ##   0.7567265         0.8830922         0.7358181            0.8652238          
    ##   0.7436259         0.8767220         0.7203300            0.8582962          
    ##   0.7437985         0.8746998         0.7219115            0.8565193          
    ##   0.7545672         0.8782062         0.7283053            0.8583329          
    ##   0.7556192         0.8837709         0.7347380            0.8623417          
    ##   0.7397159         0.8728026         0.7175962            0.8531008          
    ##   0.7413916         0.8734624         0.7232138            0.8540293          
    ##   0.7448378         0.8755300         0.7224589            0.8574323          
    ##   0.7323122         0.8704027         0.7219226            0.8524712          
    ##   0.7498102         0.8811022         0.7300040            0.8621429          
    ##   0.7480506         0.8786672         0.7345860            0.8587528          
    ##   0.7267591         0.8695020         0.7136159            0.8502582          
    ##   0.7425000         0.8762538         0.7219416            0.8545485          
    ##   0.7227025         0.8689585         0.7131435            0.8486798          
    ##   0.7138114         0.8624615         0.7043370            0.8424115          
    ##   Mean_Precision  Mean_Recall  Mean_Detection_Rate  Mean_Balanced_Accuracy
    ##   0.6888074       0.7121148    0.2443963            0.7890912             
    ##   0.6975022       0.7423709    0.2474576            0.8088433             
    ##   0.7032759       0.7353864    0.2444101            0.8021271             
    ##   0.7212088       0.7529246    0.2539236            0.8181606             
    ##   0.7083603       0.7413639    0.2464406            0.8073268             
    ##   0.7107342       0.7529297    0.2494949            0.8160707             
    ##   0.7212406       0.7610808    0.2552670            0.8244888             
    ##   0.7308221       0.7660595    0.2532261            0.8255864             
    ##   0.7078471       0.7409603    0.2443895            0.8074057             
    ##   0.7222299       0.7552226    0.2488353            0.8182514             
    ##   0.7156829       0.7413794    0.2440665            0.8071242             
    ##   0.7231815       0.7520806    0.2478252            0.8157317             
    ##   0.7261190       0.7513997    0.2484917            0.8145862             
    ##   0.7194978       0.7453788    0.2454168            0.8099896             
    ##   0.7358181       0.7567265    0.2508589            0.8199093             
    ##   0.7203300       0.7436259    0.2450972            0.8101739             
    ##   0.7219115       0.7437985    0.2447571            0.8092492             
    ##   0.7283053       0.7545672    0.2461142            0.8163867             
    ##   0.7347380       0.7556192    0.2474644            0.8196950             
    ##   0.7175962       0.7397159    0.2423761            0.8062593             
    ##   0.7232138       0.7413916    0.2427300            0.8074270             
    ##   0.7224589       0.7448378    0.2447537            0.8101839             
    ##   0.7219226       0.7323122    0.2406686            0.8013575             
    ##   0.7300040       0.7498102    0.2474816            0.8154562             
    ##   0.7345860       0.7480506    0.2447605            0.8133589             
    ##   0.7136159       0.7267591    0.2386243            0.7981306             
    ##   0.7219416       0.7425000    0.2420291            0.8093769             
    ##   0.7131435       0.7227025    0.2369374            0.7958305             
    ##   0.7043370       0.7138114    0.2331993            0.7881365             
    ## 
    ## AUC was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 29.

Model terbaik dipilih berdasarkan nilai **AUC**
terbesar. Berdasarkan kriteria tersebut model yang terpilih adalalah
model yang memiliki nilai `k` = 29. Nilai **AUC** rata-rata model
terbaik adalah sebagai berikut:

``` r
knn_roc <- knn_fit_cv$results %>%
  arrange(-AUC) %>%
  slice(1) %>%.[,"AUC"] 
knn_roc
```

    ## [1] 0.9141198

Berdasarkan hasil yang diperoleh, luas area dibawah kurva **ROC**
sebesar 0.9141198 Berdasarkan hasil tersebut, model klasifikasi yang
terbentuk lebih baik dibanding menebak secara acak.

Visualisasi hubungan antar parameter dan **ROC** ditampilkan pada gambar
berikut:

``` r
# visualisasi
ggplot(knn_fit_cv)
```

![](temp_files/figure-gfm/knn-cv-vis-1.png)<!-- -->

## Model Akhir

Model terbaik dari hasil proses validasi silang selanjutnya diekstrak.
Hal ini berguna untuk mengurangi ukuran model yang tersimpan. Secara
default fungsi `train()` akan mengembalikan model dengan performa
terbaik. Namun, terdapat sejumlah komponen lain dalam objek yang
terbentuk, seperti: hasil prediksi, ringkasan training, dll. yang
membuat ukuran objek menjadi besar. Untuk menguranginya, kita perlu
mengambil objek model final dari objek hasil validasi silang.

``` r
knn_fit <- knn_fit_cv$finalModel
```

Ringkasan model final *KNN* ditampilkan menggunakan sintaks berikut:

``` r
knn_fit
```

    ## 29-nearest neighbor model
    ## Training set outcome distribution:
    ## 
    ## Jason_Mraz   Maroon_5      Queen 
    ##        426        181        387

Model yang dihasilkan selanjutnya dapat kita uji lagi menggunakan data
baru. Berikut adalah perhitungan nilai **Akurasi** model pada data
*test*.

``` r
pred_test <- predict(knn_fit, {baked_test %>% 
    dplyr::select(!artist)})

pred_test <-
  as.data.frame(pred_test) %>%
  rowid_to_column("row") %>%
  pivot_longer(cols = Jason_Mraz:Queen, names_to = "artist", values_to = "prob") %>%
  group_by(row) %>%
  summarise(prediction = which.max(prob)) %>%
  mutate(prediction = factor(prediction, 
                             labels = c("Jason_Mraz", "Maroon_5", "Queen" ))) %>%
  dplyr::select(prediction) %>%
  pull()
```

    ## `summarise()` ungrouping output (override with `.groups` argument)

``` r
## RMSE
cm <- confusionMatrix(pred_test, baked_test$artist)
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##             Reference
    ## Prediction   Jason_Mraz Maroon_5 Queen
    ##   Jason_Mraz         33       33    56
    ##   Maroon_5            1       27     3
    ##   Queen               1        5    84
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.5926         
    ##                  95% CI : (0.5279, 0.655)
    ##     No Information Rate : 0.5885         
    ##     P-Value [Acc > NIR] : 0.4756         
    ##                                          
    ##                   Kappa : 0.397          
    ##                                          
    ##  Mcnemar's Test P-Value : <2e-16         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Jason_Mraz Class: Maroon_5 Class: Queen
    ## Sensitivity                     0.9429          0.4154       0.5874
    ## Specificity                     0.5721          0.9775       0.9400
    ## Pos Pred Value                  0.2705          0.8710       0.9333
    ## Neg Pred Value                  0.9835          0.8208       0.6144
    ## Prevalence                      0.1440          0.2675       0.5885
    ## Detection Rate                  0.1358          0.1111       0.3457
    ## Detection Prevalence            0.5021          0.1276       0.3704
    ## Balanced Accuracy               0.7575          0.6965       0.7637

Berdasarkan hasil evaluasi diperoleh nilai akurasi sebesar 0.5925926

## Interpretasi Fitur

Untuk mengetahui variabel yang paling berpengaruh secara global terhadap
hasil prediksi model, kita dapat menggunakan plot *variable importance*.

``` r
vi <- varImp(knn_fit_cv, num_features = 10) %>% ggplot()
vi
```

![](temp_files/figure-gfm/knn-vip-1.png)<!-- -->

Berdasarkan terdapat 4 buah variabel yang berpengaruh besar terhadap
prediksi yang dihasilkan oleh model, antara lain: danceability,
instrumentalness, acousticness, popularity. Untuk melihat efek dari
masing-masing variabel terhadap variabel respon, kita dapat menggunakan
*partial dependence plot*.

``` r
p1 <- pdp::partial(knn_fit_cv, pred.var = as.character(vi$data[1, 3])) %>% 
  autoplot() 

p2 <- pdp::partial(knn_fit_cv, pred.var = as.character(vi$data[2, 3])) %>% 
  autoplot()

p3 <- pdp::partial(knn_fit_cv, pred.var = as.character(vi$data[3, 3])) %>% 
  autoplot()
  

p4 <- pdp::partial(knn_fit_cv, pred.var = as.character(vi$data[4, 3])) %>% 
  autoplot()

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

    ## Warning: Use of `object[[1L]]` is discouraged. Use `.data[[1L]]` instead.

    ## Warning: Use of `object[["yhat"]]` is discouraged. Use `.data[["yhat"]]`
    ## instead.

    ## Warning: Use of `object[[1L]]` is discouraged. Use `.data[[1L]]` instead.

    ## Warning: Use of `object[["yhat"]]` is discouraged. Use `.data[["yhat"]]`
    ## instead.

    ## Warning: Use of `object[[1L]]` is discouraged. Use `.data[[1L]]` instead.

    ## Warning: Use of `object[["yhat"]]` is discouraged. Use `.data[["yhat"]]`
    ## instead.

    ## Warning: Use of `object[[1L]]` is discouraged. Use `.data[[1L]]` instead.

    ## Warning: Use of `object[["yhat"]]` is discouraged. Use `.data[["yhat"]]`
    ## instead.

![](temp_files/figure-gfm/knn-pdp-1.png)<!-- -->
