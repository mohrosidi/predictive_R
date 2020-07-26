Naive Bayes
================
Moh. Rosidi
7/26/2020

# Dataset Credit Data

Pada artikel ini, kita akan membuat model prediktif menggunakan dataset
`credit_data` dari paket `modeldata`. Dataset ini berasal dari website
Dr. Lluís A. Belanche Muñoz yang diambil dari github Dr. Gaston Sanchez.
Untuk info lebih lanjut terkait data kunjungi
\<<https://github.com/gastonstat/CreditScoring>,
<http://bit.ly/2kkBFrk>\>

# Persiapan

## Library

Terdapat beberapa paket yang digunakan dalam pembuatan model prediktif
menggunakan *naive bayes*. Paket-paket yang digunakan ditampilkan sebagai berikut:

``` r
# library pembantu
library(doParallel)
library(tidyverse)
library(rsample)
library(recipes)
library(DataExplorer)
library(skimr)
library(DMwR)
library(modeldata)

# library model
library(caret)
library(klaR)

# paket penjelasan model
library(vip)
library(pdp)
```

**Paket Pembantu**

1.  `doParallel` : parallel processing di `R`
2.  `tidyverse` : kumpulan paket dalam bidang data science
3.  `rsample` : membantu proses *data splitting*
4.  `recipes`: membantu proses data pra-pemrosesan
5.  `DataExplorer` : EDA
6.  `skimr` : membuat ringkasan data
7.  `DMwR` : paket untuk melakukan sampling “smote”
8.  `modeldata` : kumpulan dataset untuk membuat model *machine
    learning*

**Paket untuk Membangun Model**

1.  `caret` : berisikan sejumlah fungsi yang dapat merampingkan proses
    pembuatan model regresi dan klasifikasi.
2.  `klaR` : membuat model naive bayes

**Paket Interpretasi Model**

2.  `vip` : visualisasi *variable importance*
3.  `pdp` : visualisasi plot ketergantungan parsial

## Import Dataset

Dataset `credit_data` berada dalam paket `modeldata`. Untuk
mengimportnya, gunakan fungsi `data()`.

``` r
data("credit_data")
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
variabel target (`Status`).

``` r
set.seed(123)

split  <- initial_split(credit_data, prop = 0.8, strata = "Status")
data_train  <- training(split)
data_test   <- testing(split)
```

Untuk mengecek distribusi dari kedua set data, kita dapat
mevisualisasikan distribusi dari variabel target pada kedua set
tersebut.

``` r
# training set
ggplot(data_train, aes(x = Status)) + 
  geom_bar() 
```

![](temp_files/figure-gfm/target-vis-1.png)<!-- -->

``` r
# test set
ggplot(data_test, aes(x = Status)) + 
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
glimpse(data_train)
```

    ## Rows: 3,565
    ## Columns: 14
    ## $ Status    <fct> good, bad, good, good, good, good, good, bad, good, good,...
    ## $ Seniority <int> 9, 10, 0, 0, 1, 29, 9, 0, 8, 19, 0, 0, 15, 33, 0, 2, 5, 1...
    ## $ Home      <fct> rent, owner, rent, rent, owner, owner, parents, parents, ...
    ## $ Time      <int> 60, 36, 60, 36, 60, 60, 12, 48, 60, 36, 18, 24, 24, 24, 4...
    ## $ Age       <int> 30, 46, 24, 26, 36, 44, 27, 41, 30, 37, 21, 68, 52, 68, 3...
    ## $ Marital   <fct> married, married, single, single, married, married, singl...
    ## $ Records   <fct> no, yes, no, no, no, no, no, no, no, no, yes, no, no, no,...
    ## $ Job       <fct> freelance, freelance, fixed, fixed, fixed, fixed, fixed, ...
    ## $ Expenses  <int> 73, 90, 63, 46, 75, 75, 35, 90, 75, 75, 35, 75, 35, 65, 4...
    ## $ Income    <int> 129, 200, 182, 107, 214, 125, 80, 80, 199, 170, 50, 131, ...
    ## $ Assets    <int> 0, 3000, 2500, 0, 3500, 10000, 0, 0, 5000, 3500, 0, 4162,...
    ## $ Debt      <int> 0, 0, 0, 0, 0, 0, 0, 0, 2500, 260, 0, 0, 0, 2000, 0, 0, 0...
    ## $ Amount    <int> 800, 2000, 900, 310, 650, 1600, 200, 1200, 1500, 600, 400...
    ## $ Price     <int> 846, 2985, 1325, 910, 1645, 1800, 1093, 1468, 1650, 940, ...

``` r
skim(data_train)
```

|                                                  |             |
| :----------------------------------------------- | :---------- |
| Name                                             | data\_train |
| Number of rows                                   | 3565        |
| Number of columns                                | 14          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |             |
| Column type frequency:                           |             |
| factor                                           | 5           |
| numeric                                          | 9           |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |             |
| Group variables                                  | None        |

Data summary

**Variable type: factor**

| skim\_variable | n\_missing | complete\_rate | ordered | n\_unique | top\_counts                             |
| :------------- | ---------: | -------------: | :------ | --------: | :-------------------------------------- |
| Status         |          0 |              1 | FALSE   |         2 | goo: 2561, bad: 1004                    |
| Home           |          6 |              1 | FALSE   |         6 | own: 1681, ren: 777, par: 622, oth: 263 |
| Marital        |          1 |              1 | FALSE   |         5 | mar: 2573, sin: 804, sep: 103, wid: 56  |
| Records        |          0 |              1 | FALSE   |         2 | no: 2957, yes: 608                      |
| Job            |          2 |              1 | FALSE   |         4 | fix: 2265, fre: 796, par: 367, oth: 135 |

**Variable type: numeric**

| skim\_variable | n\_missing | complete\_rate |    mean |       sd |  p0 |  p25 |  p50 |  p75 |   p100 | hist  |
| :------------- | ---------: | -------------: | ------: | -------: | --: | ---: | ---: | ---: | -----: | :---- |
| Seniority      |          0 |           1.00 |    7.94 |     8.15 |   0 |    2 |    5 |   12 |     48 | ▇▃▁▁▁ |
| Time           |          0 |           1.00 |   46.32 |    14.69 |   6 |   36 |   48 |   60 |     72 | ▁▂▅▃▇ |
| Age            |          0 |           1.00 |   36.97 |    11.07 |  18 |   28 |   35 |   45 |     68 | ▇▇▆▃▁ |
| Expenses       |          0 |           1.00 |   55.33 |    19.53 |  35 |   35 |   48 |   70 |    173 | ▇▂▁▁▁ |
| Income         |        312 |           0.91 |  142.18 |    81.05 |   6 |   90 |  125 |  172 |    959 | ▇▂▁▁▁ |
| Assets         |         37 |           0.99 | 5301.80 | 11166.28 |   0 |    0 | 3000 | 6000 | 300000 | ▇▁▁▁▁ |
| Debt           |         15 |           1.00 |  360.30 |  1324.55 |   0 |    0 |    0 |    0 |  30000 | ▇▁▁▁▁ |
| Amount         |          0 |           1.00 | 1036.59 |   469.54 | 100 |  700 | 1000 | 1300 |   5000 | ▇▆▁▁▁ |
| Price          |          0 |           1.00 | 1461.75 |   595.82 | 105 | 1128 | 1403 | 1690 |   6900 | ▇▆▁▁▁ |

``` r
plot_missing(data_train)
```

![](temp_files/figure-gfm/missing-vis-1.png)<!-- -->

Berdasarkan ringkasan data yang dihasilkan, diketahui dimensi data
sebesar 3565 baris dan 14 kolom. Dengan rincian masing-masing kolom,
yaitu: 5 kolom dengan jenis data factor dan 9 kolom dengan jenis data
numeric. Informasi lain yang diketahui adalah beberapa kolom dalam data
memiliki *missing value*.

## Variasi

Variasi dari tiap variabel dapat divisualisasikan dengan menggunakan
histogram (numerik) dan baplot (kategorikal).

``` r
plot_histogram(data_train, ncol = 2L, nrow = 2L)
```

![](temp_files/figure-gfm/hist-1.png)<!-- -->![](temp_files/figure-gfm/hist-2.png)<!-- -->![](temp_files/figure-gfm/hist-3.png)<!-- -->

``` r
plot_bar(data_train, ncol = 2L, nrow = 2L)
```

![](temp_files/figure-gfm/bar-1.png)<!-- -->![](temp_files/figure-gfm/bar-2.png)<!-- -->

Berdasarkan hasil visualisasi diperoleh bahwa sebagian besar variabel
numerik memiliki distribusi yang tidak simetris. Sedangkan pada variabel
kategorikal diketahui bahwa seluruh variabel memiliki variasi yang tidak
mendekati nol atau nol. Untuk mengetahui variabel dengan variasi
mendekati nol atau nol, dapat menggunakan sintaks berikut:

``` r
nzvar <- nearZeroVar(data_train, saveMetrics = TRUE) %>% 
  rownames_to_column() %>% 
  filter(nzv)
nzvar
```

    ##   rowname freqRatio percentUnique zeroVar  nzv
    ## 1    Debt     58.54      4.684432   FALSE TRUE

## Kovarian

Kovarian dapat dicek melalui visualisasi *heatmap* koefisien korelasi.

``` r
plot_correlation(data_train, 
                 cor_args = list(method = "spearman",
                                 use = "pairwise.complete.obs"))
```

![](temp_files/figure-gfm/heatmap-1.png)<!-- -->

# Target and Feature Engineering

*Data preprocessing* dan *engineering* mengacu pada proses penambahan,
penghapusan, atau transformasi data. Waktu yang diperlukan untuk
memikirkan identifikasi kebutuhan *data engineering* dapat berlangsung
cukup lama dan proprsinya akan menjadi yang terbesar dibandingkan
analisa lainnya. Hal ini disebabkan karena kita perlu untuk memahami
data apa yang akan kita oleh atau diinputkan ke dalam model.

Untuk menyederhanakan proses *feature engineering*, kita harus
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
blueprint <- recipe(Status ~ ., data = data_train) %>%
  step_nzv(all_nominal())  %>%
  
  # 2. imputation to missing value
  # step_medianimpute("<Num_Var_name>") %>% # median imputation
  # step_meanimpute("<Num_var_name>") %>% # mean imputation
  # step_modeimpute("<Cat_var_name>") %>% # mode imputation
  step_bagimpute(all_predictors()) %>% # random forest imputation
  # step_knnimpute("<Var_name>") %>% # knn imputation
  
  # Label encoding for categorical variable with many classes 
  # step_integer("<Cat_var_name>") %>%
  
  # 3. standardization 
  step_center(all_numeric())  %>%
  step_scale(all_numeric()) %>%
  
  # 4. upsampling
  step_upsample(all_outcomes())
  

blueprint
```

    ## Data Recipe
    ## 
    ## Inputs:
    ## 
    ##       role #variables
    ##    outcome          1
    ##  predictor         13
    ## 
    ## Operations:
    ## 
    ## Sparse, unbalanced variable filter on all_nominal()
    ## Bagged tree imputation for all_predictors()
    ## Centering for all_numeric()
    ## Scaling for all_numeric()
    ## Up-sampling based on all_outcomes()

Selanjutnya, *blueprint* yang telah dibuat dilakukan *training* pada
data *training*. Perlu diperhatikan, kita tidak melakukan proses
*training* pada data *test* untuk mencegah *data leakage*.

``` r
prepare <- prep(blueprint, training = data_train)
prepare
```

    ## Data Recipe
    ## 
    ## Inputs:
    ## 
    ##       role #variables
    ##    outcome          1
    ##  predictor         13
    ## 
    ## Training data contained 3565 data points and 336 incomplete rows. 
    ## 
    ## Operations:
    ## 
    ## Sparse, unbalanced variable filter removed no terms [trained]
    ## Bagged tree imputation for Seniority, Home, Time, Age, Marital, ... [trained]
    ## Centering for Seniority, Time, Age, Expenses, Income, ... [trained]
    ## Scaling for Seniority, Time, Age, Expenses, Income, ... [trained]
    ## Up-sampling based on Status [trained]

Langkah terakhir adalah mengaplikasikan *blueprint* pada data *training*
dan *test* menggunakan fungsi `bake()`.

``` r
baked_train <- bake(prepare, new_data = data_train)
baked_test <- bake(prepare, new_data = data_test)
baked_train
```

    ## # A tibble: 3,565 x 14
    ##    Seniority Home    Time      Age Marital Records Job   Expenses Income  Assets
    ##        <dbl> <fct>  <dbl>    <dbl> <fct>   <fct>   <fct>    <dbl>  <dbl>   <dbl>
    ##  1   0.130   rent   0.931 -0.629   married no      free~    0.905 -0.175 -0.479 
    ##  2   0.253   owner -0.702  0.816   married yes     free~    1.78   0.734 -0.209 
    ##  3  -0.974   rent   0.931 -1.17    single  no      fixed    0.393  0.504 -0.254 
    ##  4  -0.974   rent  -0.702 -0.990   single  no      fixed   -0.478 -0.457 -0.479 
    ##  5  -0.851   owner  0.931 -0.0872  married no      fixed    1.01   0.914 -0.164 
    ##  6   2.58    owner  0.931  0.635   married no      fixed    1.01  -0.227  0.419 
    ##  7   0.130   pare~ -2.34  -0.900   single  no      fixed   -1.04  -0.803 -0.479 
    ##  8  -0.974   pare~  0.114  0.364   married no      part~    1.78  -0.803 -0.479 
    ##  9   0.00747 owner  0.931 -0.629   married no      fixed    1.01   0.721 -0.0298
    ## 10   1.36    priv  -0.702  0.00307 married no      fixed    1.01   0.350 -0.164 
    ## # ... with 3,555 more rows, and 4 more variables: Debt <dbl>, Amount <dbl>,
    ## #   Price <dbl>, Status <fct>

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
cl <- makePSOCKcluster(2)
registerDoParallel(cl)

# spesifikasi metode validasi silang
cv <- trainControl(
  # possible value: "boot", "boot632", "optimism_boot", "boot_all", "cv", 
  #                 "repeatedcv", "LOOCV", "LGOCV"
  method = "cv", 
  number = 4, 
  # repeats = 5,
  classProbs = TRUE,
  savePredictions = TRUE,
  summaryFunction = twoClassSummary,
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
model_fit_cv <- train(
  blueprint,
  data = data_train,
  method = "nb",
  trControl = cv,
  tuneGrid =  hyper_grid,
  metric = "ROC"
  )
)
```

    ##    user  system elapsed 
    ##   32.95    5.30 2964.10

``` r
stopCluster(cl)

model_fit_cv
```

    ## Naive Bayes 
    ## 
    ## 3565 samples
    ##   13 predictor
    ##    2 classes: 'bad', 'good' 
    ## 
    ## Recipe steps: nzv, bagimpute, center, scale, upsample 
    ## Resampling: Cross-Validated (4 fold) 
    ## Summary of sample sizes: 2673, 2674, 2674, 2674 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  fL  adjust  ROC        Sens       Spec     
    ##   FALSE      0   0       0.7764198  0.8077689  0.5931028
    ##   FALSE      0   1       0.7764198  0.8077689  0.5931028
    ##   FALSE      0   2       0.7764198  0.8077689  0.5931028
    ##   FALSE      0   3       0.7764198  0.8077689  0.5931028
    ##   FALSE      0   4       0.7764198  0.8077689  0.5931028
    ##   FALSE      0   5       0.7764198  0.8077689  0.5931028
    ##   FALSE      1   0       0.7763638  0.8077689  0.5934934
    ##   FALSE      1   1       0.7763638  0.8077689  0.5934934
    ##   FALSE      1   2       0.7763638  0.8077689  0.5934934
    ##   FALSE      1   3       0.7763638  0.8077689  0.5934934
    ##   FALSE      1   4       0.7763638  0.8077689  0.5934934
    ##   FALSE      1   5       0.7763638  0.8077689  0.5934934
    ##   FALSE      2   0       0.7762860  0.8077689  0.5934941
    ##   FALSE      2   1       0.7762860  0.8077689  0.5934941
    ##   FALSE      2   2       0.7762860  0.8077689  0.5934941
    ##   FALSE      2   3       0.7762860  0.8077689  0.5934941
    ##   FALSE      2   4       0.7762860  0.8077689  0.5934941
    ##   FALSE      2   5       0.7762860  0.8077689  0.5934941
    ##   FALSE      3   0       0.7762066  0.8077689  0.5934941
    ##   FALSE      3   1       0.7762066  0.8077689  0.5934941
    ##   FALSE      3   2       0.7762066  0.8077689  0.5934941
    ##   FALSE      3   3       0.7762066  0.8077689  0.5934941
    ##   FALSE      3   4       0.7762066  0.8077689  0.5934941
    ##   FALSE      3   5       0.7762066  0.8077689  0.5934941
    ##   FALSE      4   0       0.7760712  0.8067729  0.5934941
    ##   FALSE      4   1       0.7760712  0.8067729  0.5934941
    ##   FALSE      4   2       0.7760712  0.8067729  0.5934941
    ##   FALSE      4   3       0.7760712  0.8067729  0.5934941
    ##   FALSE      4   4       0.7760712  0.8067729  0.5934941
    ##   FALSE      4   5       0.7760712  0.8067729  0.5934941
    ##   FALSE      5   0       0.7759639  0.8067729  0.5942747
    ##   FALSE      5   1       0.7759639  0.8067729  0.5942747
    ##   FALSE      5   2       0.7759639  0.8067729  0.5942747
    ##   FALSE      5   3       0.7759639  0.8067729  0.5942747
    ##   FALSE      5   4       0.7759639  0.8067729  0.5942747
    ##   FALSE      5   5       0.7759639  0.8067729  0.5942747
    ##    TRUE      0   0             NaN        NaN        NaN
    ##    TRUE      0   1       0.7825463  0.7500000  0.6801903
    ##    TRUE      0   2       0.7925464  0.7609562  0.6872209
    ##    TRUE      0   3       0.7928308  0.7639442  0.6860515
    ##    TRUE      0   4       0.7943797  0.7798805  0.6637950
    ##    TRUE      0   5       0.7933295  0.7978088  0.6391960
    ##    TRUE      1   0             NaN        NaN        NaN
    ##    TRUE      1   1       0.7824778  0.7480080  0.6805809
    ##    TRUE      1   2       0.7924748  0.7609562  0.6876115
    ##    TRUE      1   3       0.7927919  0.7639442  0.6868327
    ##    TRUE      1   4       0.7943190  0.7798805  0.6645762
    ##    TRUE      1   5       0.7932766  0.7998008  0.6395866
    ##    TRUE      2   0             NaN        NaN        NaN
    ##    TRUE      2   1       0.7824156  0.7470120  0.6809715
    ##    TRUE      2   2       0.7924173  0.7609562  0.6883922
    ##    TRUE      2   3       0.7927391  0.7649402  0.6872233
    ##    TRUE      2   4       0.7942832  0.7798805  0.6649668
    ##    TRUE      2   5       0.7932392  0.7998008  0.6399766
    ##    TRUE      3   0             NaN        NaN        NaN
    ##    TRUE      3   1       0.7823487  0.7470120  0.6813621
    ##    TRUE      3   2       0.7923861  0.7609562  0.6887822
    ##    TRUE      3   3       0.7926830  0.7649402  0.6876140
    ##    TRUE      3   4       0.7942630  0.7798805  0.6653569
    ##    TRUE      3   5       0.7931723  0.7988048  0.6407578
    ##    TRUE      4   0             NaN        NaN        NaN
    ##    TRUE      4   1       0.7822849  0.7470120  0.6817528
    ##    TRUE      4   2       0.7923379  0.7609562  0.6887822
    ##    TRUE      4   3       0.7926254  0.7649402  0.6880046
    ##    TRUE      4   4       0.7941712  0.7798805  0.6657475
    ##    TRUE      4   5       0.7930820  0.7988048  0.6407578
    ##    TRUE      5   0             NaN        NaN        NaN
    ##    TRUE      5   1       0.7821993  0.7470120  0.6821434
    ##    TRUE      5   2       0.7922461  0.7599602  0.6883916
    ##    TRUE      5   3       0.7925663  0.7649402  0.6880046
    ##    TRUE      5   4       0.7940669  0.7798805  0.6661381
    ##    TRUE      5   5       0.7930478  0.7988048  0.6411479
    ## 
    ## ROC was used to select the optimal model using the largest value.
    ## The final values used for the model were fL = 0, usekernel = TRUE and adjust
    ##  = 4.

Model terbaik dipilih berdasarkan nilai **AUC** terbesar. Berdasarkan
kriteria tersebut model yang terpilih adalalah model yang memiliki nilai
`ft` = 0, `usekernel` = TRUE dan `adjust` = 4. Nilai **AUC** rata-rata
model terbaik adalah sebagai berikut:

``` r
roc <- model_fit_cv$results %>%
  arrange(-ROC) %>%
  slice(1) %>%
  dplyr::select(ROC) %>%
  pull()
roc
```

    ## [1] 0.7943797

Berdasarkan hasil yang diperoleh, luas area dibawah kurva **ROC**
sebesar 0.7943797 Berdasarkan hasil tersebut, model klasifikasi yang
terbentuk lebih baik dibanding menebak secara acak.

Visualisasi hubungan antar parameter dan **ROC** ditampilkan pada gambar
berikut:

``` r
# visualisasi
ggplot(model_fit_cv)
```

![](temp_files/figure-gfm/model-cv-vis-1.png)<!-- -->

## Model Akhir

Model terbaik dari hasil proses validasi silang selanjutnya diekstrak.
Hal ini berguna untuk mengurangi ukuran model yang tersimpan. Secara
default fungsi `train()` akan mengembalikan model dengan performa
terbaik. Namun, terdapat sejumlah komponen lain dalam objek yang
terbentuk, seperti: hasil prediksi, ringkasan training, dll. yang
membuat ukuran objek menjadi besar. Untuk menguranginya, kita perlu
mengambil objek model final dari objek hasil validasi silang.

``` r
model_fit <- model_fit_cv$finalModel
```

Ringkasan model final *KNN* ditampilkan menggunakan sintaks berikut:

``` r
plot(model_fit)
```

![](temp_files/figure-gfm/model-vis-1.png)<!-- -->![](temp_files/figure-gfm/model-vis-2.png)<!-- -->![](temp_files/figure-gfm/model-vis-3.png)<!-- -->![](temp_files/figure-gfm/model-vis-4.png)<!-- -->![](temp_files/figure-gfm/model-vis-5.png)<!-- -->![](temp_files/figure-gfm/model-vis-6.png)<!-- -->![](temp_files/figure-gfm/model-vis-7.png)<!-- -->![](temp_files/figure-gfm/model-vis-8.png)<!-- -->![](temp_files/figure-gfm/model-vis-9.png)<!-- -->![](temp_files/figure-gfm/model-vis-10.png)<!-- -->![](temp_files/figure-gfm/model-vis-11.png)<!-- -->![](temp_files/figure-gfm/model-vis-12.png)<!-- -->![](temp_files/figure-gfm/model-vis-13.png)<!-- -->

Model yang dihasilkan selanjutnya dapat kita uji lagi menggunakan data
baru. Berikut adalah perhitungan nilai **Akurasi** model pada data
*test*.

``` r
pred_test <- predict(model_fit, baked_test %>% dplyr::select(!Status),
                     type = "class")


## RMSE
cm <- confusionMatrix(pred_test$class, baked_test$Status)
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction bad good
    ##       bad  188  235
    ##       good  62  404
    ##                                           
    ##                Accuracy : 0.6659          
    ##                  95% CI : (0.6338, 0.6969)
    ##     No Information Rate : 0.7188          
    ##     P-Value [Acc > NIR] : 0.9998          
    ##                                           
    ##                   Kappa : 0.3174          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##             Sensitivity : 0.7520          
    ##             Specificity : 0.6322          
    ##          Pos Pred Value : 0.4444          
    ##          Neg Pred Value : 0.8670          
    ##              Prevalence : 0.2812          
    ##          Detection Rate : 0.2115          
    ##    Detection Prevalence : 0.4758          
    ##       Balanced Accuracy : 0.6921          
    ##                                           
    ##        'Positive' Class : bad             
    ## 

Berdasarkan hasil evaluasi diperoleh nilai akurasi sebesar 0.6659168

## Interpretasi Fitur

Untuk mengetahui variabel yang paling berpengaruh secara global terhadap
hasil prediksi model, kita dapat menggunakan plot *variable importance*.

``` r
vi <- varImp(model_fit_cv, num_features = 10) %>% ggplot()
vi
```

![](temp_files/figure-gfm/model-vip-1.png)<!-- -->

Berdasarkan terdapat 4 buah variabel yang berpengaruh besar terhadap
prediksi yang dihasilkan oleh model, antara lain: Seniority, Assets,
Job, Records. Untuk melihat efek dari masing-masing variabel terhadap
variabel respon, kita dapat menggunakan *partial dependence plot*.

``` r
pred.fun <- function(object, newdata) {
  Good <- mean(predict(object, newdata, type = "prob")$good)
  as.data.frame(Good)
}

p1 <- pdp::partial(model_fit_cv, pred.var = vi$data %>% rownames_to_column("ID") %>% .[1,1], pred.fun = pred.fun) %>% 
  autoplot() 

p2 <- pdp::partial(model_fit_cv, pred.var = vi$data %>% rownames_to_column("ID") %>% .[2,1], pred.fun = pred.fun) %>% 
  autoplot()

p3 <- pdp::partial(model_fit_cv, pred.var = vi$data %>% rownames_to_column("ID") %>% .[3,1], pred.fun = pred.fun) %>% 
  autoplot()
  

p4 <- pdp::partial(model_fit_cv, pred.var = vi$data %>% rownames_to_column("ID") %>% .[4,1], pred.fun = pred.fun) %>% 

  
    autoplot()

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

![](temp_files/figure-gfm/model-pdp-1.png)<!-- -->
