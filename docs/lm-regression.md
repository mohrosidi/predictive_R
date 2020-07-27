Regresi Linier
================
Moh. Rosidi
7/24/2020

# Dataset Ames

Sebuah dataset terkait data properti yang ada di Ames IA. Dataset ini
memiliki 82 variabel dan 2930 baris. Untuk informasi lebih lanjut
terkait dataset ini, kunjungin tautan berikut:

  - <https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt>
  - <http://ww2.amstat.org/publications/jse/v19n3/decock.pdf>

# Persiapan

## Library

Terdapat beberapa paket yang digunakan dalam pembuatan model prediktif
menggunakan regresi linier. Paket-paket yang digunakan ditampilkan
sebagai berikut:

``` r
# library pembantu
library(tidyverse)
library(rsample)
library(recipes)
library(DataExplorer)
library(skimr)
library(modeldata)

# library model
library(caret)

# paket penjelasan model
library(vip)
library(pdp)
```

**Paket Pembantu**

1.  `tidyverse` : kumpulan paket dalam bidang data science
2.  `rsample` : membantu proses *data splitting*
3.  `recipes`: membantu proses data pra-pemrosesan
4.  `DataExplorer` : EDA
5.  `skimr` : membuat ringkasan data
6.  `modeldata` : kumpulan dataset untuk membuat model *machine
    learning*

**Paket untuk Membangun Model**

1.  `caret` : berisikan sejumlah fungsi yang dapat merampingkan proses
    pembuatan model regresi dan klasifikasi

**Paket Interpretasi Model**

1.  `vip` : visualisasi *variable importance*
2.  `pdp` : visualisasi plot ketergantungan parsial

## Import Dataset

Import dataset dilakukan dengan menggunakan fungsi `data()`. Fungsi ini
digunakan untuk mengambil data yang ada dalam sebuah paket.

``` r
data("ames")
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
variabel target (`Sale_Price`).

``` r
set.seed(123)

split  <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
data_train  <- training(split)
data_test   <- testing(split)
```

Untuk mengecek distribusi dari kedua set data, kita dapat
mevisualisasikan distribusi dari variabel target pada kedua set
tersebut.

``` r
# training set
ggplot(data_train, aes(x = Sale_Price)) + 
  geom_density() 
```

![](lm-regression_files/figure-gfm/target-vis-1.png)<!-- -->

``` r
# test set
ggplot(data_test, aes(x = Sale_Price)) + 
  geom_density() 
```

![](lm-regression_files/figure-gfm/target-vis-2.png)<!-- -->

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

    ## Rows: 2,053
    ## Columns: 74
    ## $ MS_SubClass        <fct> One_Story_1946_and_Newer_All_Styles, One_Story_1...
    ## $ MS_Zoning          <fct> Residential_Low_Density, Residential_High_Densit...
    ## $ Lot_Frontage       <dbl> 141, 80, 81, 78, 41, 39, 60, 75, 63, 85, 47, 152...
    ## $ Lot_Area           <int> 31770, 11622, 14267, 9978, 4920, 5389, 7500, 100...
    ## $ Street             <fct> Pave, Pave, Pave, Pave, Pave, Pave, Pave, Pave, ...
    ## $ Alley              <fct> No_Alley_Access, No_Alley_Access, No_Alley_Acces...
    ## $ Lot_Shape          <fct> Slightly_Irregular, Regular, Slightly_Irregular,...
    ## $ Land_Contour       <fct> Lvl, Lvl, Lvl, Lvl, Lvl, Lvl, Lvl, Lvl, Lvl, Lvl...
    ## $ Utilities          <fct> AllPub, AllPub, AllPub, AllPub, AllPub, AllPub, ...
    ## $ Lot_Config         <fct> Corner, Inside, Corner, Inside, Inside, Inside, ...
    ## $ Land_Slope         <fct> Gtl, Gtl, Gtl, Gtl, Gtl, Gtl, Gtl, Gtl, Gtl, Gtl...
    ## $ Neighborhood       <fct> North_Ames, North_Ames, North_Ames, Gilbert, Sto...
    ## $ Condition_1        <fct> Norm, Feedr, Norm, Norm, Norm, Norm, Norm, Norm,...
    ## $ Condition_2        <fct> Norm, Norm, Norm, Norm, Norm, Norm, Norm, Norm, ...
    ## $ Bldg_Type          <fct> OneFam, OneFam, OneFam, OneFam, TwnhsE, TwnhsE, ...
    ## $ House_Style        <fct> One_Story, One_Story, One_Story, Two_Story, One_...
    ## $ Overall_Cond       <fct> Average, Above_Average, Above_Average, Above_Ave...
    ## $ Year_Built         <int> 1960, 1961, 1958, 1998, 2001, 1995, 1999, 1993, ...
    ## $ Year_Remod_Add     <int> 1960, 1961, 1958, 1998, 2001, 1996, 1999, 1994, ...
    ## $ Roof_Style         <fct> Hip, Gable, Hip, Gable, Gable, Gable, Gable, Gab...
    ## $ Roof_Matl          <fct> CompShg, CompShg, CompShg, CompShg, CompShg, Com...
    ## $ Exterior_1st       <fct> BrkFace, VinylSd, Wd Sdng, VinylSd, CemntBd, Cem...
    ## $ Exterior_2nd       <fct> Plywood, VinylSd, Wd Sdng, VinylSd, CmentBd, Cme...
    ## $ Mas_Vnr_Type       <fct> Stone, None, BrkFace, BrkFace, None, None, None,...
    ## $ Mas_Vnr_Area       <dbl> 112, 0, 108, 20, 0, 0, 0, 0, 0, 0, 603, 0, 350, ...
    ## $ Exter_Cond         <fct> Typical, Typical, Typical, Typical, Typical, Typ...
    ## $ Foundation         <fct> CBlock, CBlock, CBlock, PConc, PConc, PConc, PCo...
    ## $ Bsmt_Cond          <fct> Good, Typical, Typical, Typical, Typical, Typica...
    ## $ Bsmt_Exposure      <fct> Gd, No, No, No, Mn, No, No, No, No, Gd, Gd, Av, ...
    ## $ BsmtFin_Type_1     <fct> BLQ, Rec, ALQ, GLQ, GLQ, GLQ, Unf, Unf, Unf, GLQ...
    ## $ BsmtFin_SF_1       <dbl> 2, 6, 1, 3, 3, 3, 7, 7, 7, 3, 1, 3, 3, 4, 1, 2, ...
    ## $ BsmtFin_Type_2     <fct> Unf, LwQ, Unf, Unf, Unf, Unf, Unf, Unf, Unf, Unf...
    ## $ BsmtFin_SF_2       <dbl> 0, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163,...
    ## $ Bsmt_Unf_SF        <dbl> 441, 270, 406, 324, 722, 415, 994, 763, 789, 663...
    ## $ Total_Bsmt_SF      <dbl> 1080, 882, 1329, 926, 1338, 1595, 994, 763, 789,...
    ## $ Heating            <fct> GasA, GasA, GasA, GasA, GasA, GasA, GasA, GasA, ...
    ## $ Heating_QC         <fct> Fair, Typical, Typical, Excellent, Excellent, Ex...
    ## $ Central_Air        <fct> Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, ...
    ## $ Electrical         <fct> SBrkr, SBrkr, SBrkr, SBrkr, SBrkr, SBrkr, SBrkr,...
    ## $ First_Flr_SF       <int> 1656, 896, 1329, 926, 1338, 1616, 1028, 763, 789...
    ## $ Second_Flr_SF      <int> 0, 0, 0, 678, 0, 0, 776, 892, 676, 0, 1589, 672,...
    ## $ Gr_Liv_Area        <int> 1656, 896, 1329, 1604, 1338, 1616, 1804, 1655, 1...
    ## $ Bsmt_Full_Bath     <dbl> 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, ...
    ## $ Bsmt_Half_Bath     <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    ## $ Full_Bath          <int> 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 3, 2, 1, 1, 2, 2, ...
    ## $ Half_Bath          <int> 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, ...
    ## $ Bedroom_AbvGr      <int> 3, 2, 3, 3, 2, 2, 3, 3, 3, 2, 4, 4, 1, 2, 3, 3, ...
    ## $ Kitchen_AbvGr      <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
    ## $ TotRms_AbvGrd      <int> 7, 5, 6, 7, 6, 5, 7, 7, 7, 5, 12, 8, 8, 4, 7, 7,...
    ## $ Functional         <fct> Typ, Typ, Typ, Typ, Typ, Typ, Typ, Typ, Typ, Typ...
    ## $ Fireplaces         <int> 2, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 1, ...
    ## $ Garage_Type        <fct> Attchd, Attchd, Attchd, Attchd, Attchd, Attchd, ...
    ## $ Garage_Finish      <fct> Fin, Unf, Unf, Fin, Fin, RFn, Fin, Fin, Fin, Unf...
    ## $ Garage_Cars        <dbl> 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, ...
    ## $ Garage_Area        <dbl> 528, 730, 312, 470, 582, 608, 442, 440, 393, 506...
    ## $ Garage_Cond        <fct> Typical, Typical, Typical, Typical, Typical, Typ...
    ## $ Paved_Drive        <fct> Partial_Pavement, Paved, Paved, Paved, Paved, Pa...
    ## $ Wood_Deck_SF       <int> 210, 140, 393, 360, 0, 237, 140, 157, 0, 192, 50...
    ## $ Open_Porch_SF      <int> 62, 0, 36, 36, 0, 152, 60, 84, 75, 0, 36, 12, 0,...
    ## $ Enclosed_Porch     <int> 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...
    ## $ Three_season_porch <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    ## $ Screen_Porch       <int> 0, 120, 0, 0, 0, 0, 0, 0, 0, 0, 210, 0, 0, 0, 0,...
    ## $ Pool_Area          <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    ## $ Pool_QC            <fct> No_Pool, No_Pool, No_Pool, No_Pool, No_Pool, No_...
    ## $ Fence              <fct> No_Fence, Minimum_Privacy, No_Fence, No_Fence, N...
    ## $ Misc_Feature       <fct> None, None, Gar2, None, None, None, None, None, ...
    ## $ Misc_Val           <int> 0, 0, 12500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ Mo_Sold            <int> 5, 6, 6, 6, 4, 3, 6, 4, 5, 2, 6, 6, 6, 6, 2, 1, ...
    ## $ Year_Sold          <int> 2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010, ...
    ## $ Sale_Type          <fct> WD , WD , WD , WD , WD , WD , WD , WD , WD , WD ...
    ## $ Sale_Condition     <fct> Normal, Normal, Normal, Normal, Normal, Normal, ...
    ## $ Sale_Price         <int> 215000, 105000, 172000, 195500, 213500, 236500, ...
    ## $ Longitude          <dbl> -93.61975, -93.61976, -93.61939, -93.63893, -93....
    ## $ Latitude           <dbl> 42.05403, 42.05301, 42.05266, 42.06078, 42.06298...

``` r
skim(data_train)
```

|                                                  |             |
| :----------------------------------------------- | :---------- |
| Name                                             | data\_train |
| Number of rows                                   | 2053        |
| Number of columns                                | 74          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |             |
| Column type frequency:                           |             |
| factor                                           | 40          |
| numeric                                          | 34          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |             |
| Group variables                                  | None        |

Data summary

**Variable type: factor**

| skim\_variable   | n\_missing | complete\_rate | ordered | n\_unique | top\_counts                              |
| :--------------- | ---------: | -------------: | :------ | --------: | :--------------------------------------- |
| MS\_SubClass     |          0 |              1 | FALSE   |        16 | One: 753, Two: 395, One: 211, One: 145   |
| MS\_Zoning       |          0 |              1 | FALSE   |         7 | Res: 1571, Res: 338, Flo: 102, Res: 19   |
| Street           |          0 |              1 | FALSE   |         2 | Pav: 2046, Grv: 7                        |
| Alley            |          0 |              1 | FALSE   |         3 | No\_: 1909, Gra: 93, Pav: 51             |
| Lot\_Shape       |          0 |              1 | FALSE   |         4 | Reg: 1321, Sli: 666, Mod: 54, Irr: 12    |
| Land\_Contour    |          0 |              1 | FALSE   |         4 | Lvl: 1850, HLS: 83, Bnk: 75, Low: 45     |
| Utilities        |          0 |              1 | FALSE   |         3 | All: 2050, NoS: 2, NoS: 1                |
| Lot\_Config      |          0 |              1 | FALSE   |         5 | Ins: 1494, Cor: 361, Cul: 127, FR2: 62   |
| Land\_Slope      |          0 |              1 | FALSE   |         3 | Gtl: 1958, Mod: 86, Sev: 9               |
| Neighborhood     |          0 |              1 | FALSE   |        27 | Nor: 298, Col: 187, Old: 171, Edw: 146   |
| Condition\_1     |          0 |              1 | FALSE   |         9 | Nor: 1768, Fee: 120, Art: 63, RRA: 35    |
| Condition\_2     |          0 |              1 | FALSE   |         7 | Nor: 2031, Fee: 10, Pos: 4, Art: 3       |
| Bldg\_Type       |          0 |              1 | FALSE   |         5 | One: 1692, Twn: 178, Twn: 74, Dup: 66    |
| House\_Style     |          0 |              1 | FALSE   |         8 | One: 1030, Two: 614, One: 233, SLv: 84   |
| Overall\_Cond    |          0 |              1 | FALSE   |         9 | Ave: 1168, Abo: 370, Goo: 280, Ver: 95   |
| Roof\_Style      |          0 |              1 | FALSE   |         6 | Gab: 1627, Hip: 387, Gam: 17, Fla: 14    |
| Roof\_Matl       |          0 |              1 | FALSE   |         5 | Com: 2024, Tar: 16, WdS: 7, WdS: 5       |
| Exterior\_1st    |          0 |              1 | FALSE   |        14 | Vin: 718, Met: 321, Wd : 301, HdB: 295   |
| Exterior\_2nd    |          0 |              1 | FALSE   |        16 | Vin: 710, Met: 321, Wd : 287, HdB: 267   |
| Mas\_Vnr\_Type   |          0 |              1 | FALSE   |         4 | Non: 1231, Brk: 624, Sto: 179, Brk: 19   |
| Exter\_Cond      |          0 |              1 | FALSE   |         5 | Typ: 1792, Goo: 204, Fai: 43, Exc: 11    |
| Foundation       |          0 |              1 | FALSE   |         6 | PCo: 916, CBl: 870, Brk: 226, Sla: 30    |
| Bsmt\_Cond       |          0 |              1 | FALSE   |         6 | Typ: 1834, Goo: 92, Fai: 71, No\_: 52    |
| Bsmt\_Exposure   |          0 |              1 | FALSE   |         5 | No: 1348, Av: 284, Gd: 205, Mn: 163      |
| BsmtFin\_Type\_1 |          0 |              1 | FALSE   |         7 | GLQ: 599, Unf: 588, ALQ: 313, Rec: 207   |
| BsmtFin\_Type\_2 |          0 |              1 | FALSE   |         7 | Unf: 1742, Rec: 81, LwQ: 66, No\_: 53    |
| Heating          |          0 |              1 | FALSE   |         5 | Gas: 2021, Gas: 20, Gra: 7, Wal: 3       |
| Heating\_QC      |          0 |              1 | FALSE   |         5 | Exc: 1052, Typ: 593, Goo: 340, Fai: 66   |
| Central\_Air     |          0 |              1 | FALSE   |         2 | Y: 1916, N: 137                          |
| Electrical       |          0 |              1 | FALSE   |         5 | SBr: 1873, Fus: 140, Fus: 32, Fus: 7     |
| Functional       |          0 |              1 | FALSE   |         7 | Typ: 1909, Min: 50, Min: 47, Mod: 25     |
| Garage\_Type     |          0 |              1 | FALSE   |         7 | Att: 1230, Det: 534, Bui: 124, No\_: 109 |
| Garage\_Finish   |          0 |              1 | FALSE   |         4 | Unf: 863, RFn: 567, Fin: 512, No\_: 111  |
| Garage\_Cond     |          0 |              1 | FALSE   |         6 | Typ: 1870, No\_: 111, Fai: 48, Goo: 12   |
| Paved\_Drive     |          0 |              1 | FALSE   |         3 | Pav: 1862, Dir: 140, Par: 51             |
| Pool\_QC         |          0 |              1 | FALSE   |         5 | No\_: 2045, Exc: 3, Typ: 3, Fai: 1       |
| Fence            |          0 |              1 | FALSE   |         5 | No\_: 1646, Min: 243, Goo: 80, Goo: 75   |
| Misc\_Feature    |          0 |              1 | FALSE   |         4 | Non: 1982, She: 65, Gar: 4, Oth: 2       |
| Sale\_Type       |          0 |              1 | FALSE   |        10 | WD : 1775, New: 159, COD: 65, Con: 22    |
| Sale\_Condition  |          0 |              1 | FALSE   |         6 | Nor: 1692, Par: 164, Abn: 139, Fam: 36   |

**Variable type: numeric**

| skim\_variable       | n\_missing | complete\_rate |      mean |       sd |       p0 |       p25 |       p50 |       p75 |      p100 | hist  |
| :------------------- | ---------: | -------------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: | :---- |
| Lot\_Frontage        |          0 |              1 |     57.00 |    33.69 |     0.00 |     41.00 |     62.00 |     78.00 |    313.00 | ▇▇▁▁▁ |
| Lot\_Area            |          0 |              1 |  10121.14 |  8120.98 |  1300.00 |   7313.00 |   9375.00 |  11512.00 | 215245.00 | ▇▁▁▁▁ |
| Year\_Built          |          0 |              1 |   1971.07 |    30.52 |  1872.00 |   1952.00 |   1973.00 |   2000.00 |   2010.00 | ▁▂▃▆▇ |
| Year\_Remod\_Add     |          0 |              1 |   1984.42 |    20.79 |  1950.00 |   1966.00 |   1993.00 |   2004.00 |   2010.00 | ▅▂▂▃▇ |
| Mas\_Vnr\_Area       |          0 |              1 |    102.59 |   179.08 |     0.00 |      0.00 |      0.00 |    164.00 |   1600.00 | ▇▁▁▁▁ |
| BsmtFin\_SF\_1       |          0 |              1 |      4.16 |     2.24 |     1.00 |      3.00 |      3.00 |      7.00 |      7.00 | ▅▆▁▁▇ |
| BsmtFin\_SF\_2       |          0 |              1 |     52.20 |   172.62 |     0.00 |      0.00 |      0.00 |      0.00 |   1474.00 | ▇▁▁▁▁ |
| Bsmt\_Unf\_SF        |          0 |              1 |    558.79 |   437.52 |     0.00 |    219.00 |    467.00 |    797.00 |   2336.00 | ▇▅▂▁▁ |
| Total\_Bsmt\_SF      |          0 |              1 |   1050.67 |   425.77 |     0.00 |    784.00 |    992.00 |   1298.00 |   3206.00 | ▂▇▃▁▁ |
| First\_Flr\_SF       |          0 |              1 |   1158.36 |   383.60 |   334.00 |    880.00 |   1083.00 |   1384.00 |   3820.00 | ▇▇▁▁▁ |
| Second\_Flr\_SF      |          0 |              1 |    336.30 |   426.47 |     0.00 |      0.00 |      0.00 |    702.00 |   2065.00 | ▇▃▂▁▁ |
| Gr\_Liv\_Area        |          0 |              1 |   1499.56 |   494.55 |   334.00 |   1136.00 |   1441.00 |   1743.00 |   4676.00 | ▅▇▂▁▁ |
| Bsmt\_Full\_Bath     |          0 |              1 |      0.43 |     0.53 |     0.00 |      0.00 |      0.00 |      1.00 |      3.00 | ▇▆▁▁▁ |
| Bsmt\_Half\_Bath     |          0 |              1 |      0.06 |     0.25 |     0.00 |      0.00 |      0.00 |      0.00 |      2.00 | ▇▁▁▁▁ |
| Full\_Bath           |          0 |              1 |      1.57 |     0.56 |     0.00 |      1.00 |      2.00 |      2.00 |      4.00 | ▁▇▇▁▁ |
| Half\_Bath           |          0 |              1 |      0.38 |     0.50 |     0.00 |      0.00 |      0.00 |      1.00 |      2.00 | ▇▁▅▁▁ |
| Bedroom\_AbvGr       |          0 |              1 |      2.84 |     0.83 |     0.00 |      2.00 |      3.00 |      3.00 |      8.00 | ▁▇▂▁▁ |
| Kitchen\_AbvGr       |          0 |              1 |      1.04 |     0.21 |     0.00 |      1.00 |      1.00 |      1.00 |      3.00 | ▁▇▁▁▁ |
| TotRms\_AbvGrd       |          0 |              1 |      6.42 |     1.54 |     2.00 |      5.00 |      6.00 |      7.00 |     14.00 | ▁▇▆▁▁ |
| Fireplaces           |          0 |              1 |      0.61 |     0.65 |     0.00 |      0.00 |      1.00 |      1.00 |      3.00 | ▇▇▁▁▁ |
| Garage\_Cars         |          0 |              1 |      1.77 |     0.77 |     0.00 |      1.00 |      2.00 |      2.00 |      5.00 | ▅▇▂▁▁ |
| Garage\_Area         |          0 |              1 |    474.14 |   215.76 |     0.00 |    325.00 |    480.00 |    576.00 |   1390.00 | ▂▇▃▁▁ |
| Wood\_Deck\_SF       |          0 |              1 |     92.98 |   123.01 |     0.00 |      0.00 |      0.00 |    168.00 |    857.00 | ▇▂▁▁▁ |
| Open\_Porch\_SF      |          0 |              1 |     47.42 |    68.23 |     0.00 |      0.00 |     26.00 |     70.00 |    742.00 | ▇▁▁▁▁ |
| Enclosed\_Porch      |          0 |              1 |     24.02 |    65.83 |     0.00 |      0.00 |      0.00 |      0.00 |   1012.00 | ▇▁▁▁▁ |
| Three\_season\_porch |          0 |              1 |      2.99 |    27.75 |     0.00 |      0.00 |      0.00 |      0.00 |    508.00 | ▇▁▁▁▁ |
| Screen\_Porch        |          0 |              1 |     15.36 |    54.61 |     0.00 |      0.00 |      0.00 |      0.00 |    576.00 | ▇▁▁▁▁ |
| Pool\_Area           |          0 |              1 |      1.90 |    31.33 |     0.00 |      0.00 |      0.00 |      0.00 |    648.00 | ▇▁▁▁▁ |
| Misc\_Val            |          0 |              1 |     44.77 |   509.47 |     0.00 |      0.00 |      0.00 |      0.00 |  15500.00 | ▇▁▁▁▁ |
| Mo\_Sold             |          0 |              1 |      6.20 |     2.71 |     1.00 |      4.00 |      6.00 |      8.00 |     12.00 | ▅▆▇▃▃ |
| Year\_Sold           |          0 |              1 |   2007.79 |     1.31 |  2006.00 |   2007.00 |   2008.00 |   2009.00 |   2010.00 | ▇▇▇▇▃ |
| Sale\_Price          |          0 |              1 | 180996.28 | 80258.90 | 13100.00 | 129500.00 | 160000.00 | 213500.00 | 755000.00 | ▇▇▁▁▁ |
| Longitude            |          0 |              1 |   \-93.64 |     0.03 |  \-93.69 |   \-93.66 |   \-93.64 |   \-93.62 |   \-93.58 | ▅▅▇▆▁ |
| Latitude             |          0 |              1 |     42.03 |     0.02 |    41.99 |     42.02 |     42.03 |     42.05 |     42.06 | ▂▂▇▇▇ |

``` r
plot_missing(data_train)
```

![](lm-regression_files/figure-gfm/missing-vis-1.png)<!-- -->

Berdasarkan ringkasan data yang dihasilkan, diketahui dimensi data
sebesar 2053 baris dan 74 kolom. Dengan rincian masing-masing kolom,
yaitu: 40 kolom dengan jenis data factor dan 34 kolom dengan jenis data
numeric. Informasi lain yang diketahui adalah seluruh kolom dalam data
tidak memiliki *missing value*.

## Variasi

Variasi dari tiap variabel dapat divisualisasikan dengan menggunakan
histogram (numerik) dan barplot (kategorikal).

``` r
plot_histogram(data_train, ncol = 2L, nrow = 2L)
```

![](lm-regression_files/figure-gfm/hist-1.png)<!-- -->![](lm-regression_files/figure-gfm/hist-2.png)<!-- -->![](lm-regression_files/figure-gfm/hist-3.png)<!-- -->![](lm-regression_files/figure-gfm/hist-4.png)<!-- -->![](lm-regression_files/figure-gfm/hist-5.png)<!-- -->![](lm-regression_files/figure-gfm/hist-6.png)<!-- -->![](lm-regression_files/figure-gfm/hist-7.png)<!-- -->![](lm-regression_files/figure-gfm/hist-8.png)<!-- -->![](lm-regression_files/figure-gfm/hist-9.png)<!-- -->

``` r
plot_bar(data_train, ncol = 2L, nrow = 2L)
```

![](lm-regression_files/figure-gfm/bar-1.png)<!-- -->![](lm-regression_files/figure-gfm/bar-2.png)<!-- -->![](lm-regression_files/figure-gfm/bar-3.png)<!-- -->![](lm-regression_files/figure-gfm/bar-4.png)<!-- -->![](lm-regression_files/figure-gfm/bar-5.png)<!-- -->![](lm-regression_files/figure-gfm/bar-6.png)<!-- -->![](lm-regression_files/figure-gfm/bar-7.png)<!-- -->![](lm-regression_files/figure-gfm/bar-8.png)<!-- -->![](lm-regression_files/figure-gfm/bar-9.png)<!-- -->![](lm-regression_files/figure-gfm/bar-10.png)<!-- -->

Berdasarkan hasil visualisasi diperoleh bahwa sebagian besar variabel
numerik memiliki distribusi yang tidak simetris. Sedangkan pada variabel
kategorikal diketahui bahwa terdapat beberapa variabel yang memiliki
variasi rendah atau mendekati nol. Untuk mengetahui variabel dengan
variabilitas mendekati nol atau nol, dapat menggunakan sintaks berikut:

``` r
nzvar <- nearZeroVar(data_train, saveMetrics = TRUE) %>% 
  rownames_to_column() %>% 
  filter(nzv)
nzvar
```

    ##               rowname  freqRatio percentUnique zeroVar  nzv
    ## 1              Street  292.28571    0.09741841   FALSE TRUE
    ## 2               Alley   20.52688    0.14612762   FALSE TRUE
    ## 3        Land_Contour   22.28916    0.19483682   FALSE TRUE
    ## 4           Utilities 1025.00000    0.14612762   FALSE TRUE
    ## 5          Land_Slope   22.76744    0.14612762   FALSE TRUE
    ## 6         Condition_2  203.10000    0.34096444   FALSE TRUE
    ## 7           Roof_Matl  126.50000    0.24354603   FALSE TRUE
    ## 8           Bsmt_Cond   19.93478    0.29225524   FALSE TRUE
    ## 9      BsmtFin_Type_2   21.50617    0.34096444   FALSE TRUE
    ## 10            Heating  101.05000    0.24354603   FALSE TRUE
    ## 11      Kitchen_AbvGr   23.68675    0.19483682   FALSE TRUE
    ## 12         Functional   38.18000    0.34096444   FALSE TRUE
    ## 13     Enclosed_Porch  100.94118    7.40379932   FALSE TRUE
    ## 14 Three_season_porch  674.66667    1.16902094   FALSE TRUE
    ## 15       Screen_Porch  234.87500    4.52995616   FALSE TRUE
    ## 16          Pool_Area 2045.00000    0.43838285   FALSE TRUE
    ## 17            Pool_QC  681.66667    0.24354603   FALSE TRUE
    ## 18       Misc_Feature   30.49231    0.19483682   FALSE TRUE
    ## 19           Misc_Val  165.33333    1.41256698   FALSE TRUE

Berikut adalah ringkasan data pada variabel yang tidak memiliki variasi
yang mendekati nol.

``` r
without_nzvar <- select(data_train, !nzvar$rowname)
skim(without_nzvar)
```

|                                                  |                |
| :----------------------------------------------- | :------------- |
| Name                                             | without\_nzvar |
| Number of rows                                   | 2053           |
| Number of columns                                | 55             |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |                |
| Column type frequency:                           |                |
| factor                                           | 27             |
| numeric                                          | 28             |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |                |
| Group variables                                  | None           |

Data summary

**Variable type: factor**

| skim\_variable   | n\_missing | complete\_rate | ordered | n\_unique | top\_counts                              |
| :--------------- | ---------: | -------------: | :------ | --------: | :--------------------------------------- |
| MS\_SubClass     |          0 |              1 | FALSE   |        16 | One: 753, Two: 395, One: 211, One: 145   |
| MS\_Zoning       |          0 |              1 | FALSE   |         7 | Res: 1571, Res: 338, Flo: 102, Res: 19   |
| Lot\_Shape       |          0 |              1 | FALSE   |         4 | Reg: 1321, Sli: 666, Mod: 54, Irr: 12    |
| Lot\_Config      |          0 |              1 | FALSE   |         5 | Ins: 1494, Cor: 361, Cul: 127, FR2: 62   |
| Neighborhood     |          0 |              1 | FALSE   |        27 | Nor: 298, Col: 187, Old: 171, Edw: 146   |
| Condition\_1     |          0 |              1 | FALSE   |         9 | Nor: 1768, Fee: 120, Art: 63, RRA: 35    |
| Bldg\_Type       |          0 |              1 | FALSE   |         5 | One: 1692, Twn: 178, Twn: 74, Dup: 66    |
| House\_Style     |          0 |              1 | FALSE   |         8 | One: 1030, Two: 614, One: 233, SLv: 84   |
| Overall\_Cond    |          0 |              1 | FALSE   |         9 | Ave: 1168, Abo: 370, Goo: 280, Ver: 95   |
| Roof\_Style      |          0 |              1 | FALSE   |         6 | Gab: 1627, Hip: 387, Gam: 17, Fla: 14    |
| Exterior\_1st    |          0 |              1 | FALSE   |        14 | Vin: 718, Met: 321, Wd : 301, HdB: 295   |
| Exterior\_2nd    |          0 |              1 | FALSE   |        16 | Vin: 710, Met: 321, Wd : 287, HdB: 267   |
| Mas\_Vnr\_Type   |          0 |              1 | FALSE   |         4 | Non: 1231, Brk: 624, Sto: 179, Brk: 19   |
| Exter\_Cond      |          0 |              1 | FALSE   |         5 | Typ: 1792, Goo: 204, Fai: 43, Exc: 11    |
| Foundation       |          0 |              1 | FALSE   |         6 | PCo: 916, CBl: 870, Brk: 226, Sla: 30    |
| Bsmt\_Exposure   |          0 |              1 | FALSE   |         5 | No: 1348, Av: 284, Gd: 205, Mn: 163      |
| BsmtFin\_Type\_1 |          0 |              1 | FALSE   |         7 | GLQ: 599, Unf: 588, ALQ: 313, Rec: 207   |
| Heating\_QC      |          0 |              1 | FALSE   |         5 | Exc: 1052, Typ: 593, Goo: 340, Fai: 66   |
| Central\_Air     |          0 |              1 | FALSE   |         2 | Y: 1916, N: 137                          |
| Electrical       |          0 |              1 | FALSE   |         5 | SBr: 1873, Fus: 140, Fus: 32, Fus: 7     |
| Garage\_Type     |          0 |              1 | FALSE   |         7 | Att: 1230, Det: 534, Bui: 124, No\_: 109 |
| Garage\_Finish   |          0 |              1 | FALSE   |         4 | Unf: 863, RFn: 567, Fin: 512, No\_: 111  |
| Garage\_Cond     |          0 |              1 | FALSE   |         6 | Typ: 1870, No\_: 111, Fai: 48, Goo: 12   |
| Paved\_Drive     |          0 |              1 | FALSE   |         3 | Pav: 1862, Dir: 140, Par: 51             |
| Fence            |          0 |              1 | FALSE   |         5 | No\_: 1646, Min: 243, Goo: 80, Goo: 75   |
| Sale\_Type       |          0 |              1 | FALSE   |        10 | WD : 1775, New: 159, COD: 65, Con: 22    |
| Sale\_Condition  |          0 |              1 | FALSE   |         6 | Nor: 1692, Par: 164, Abn: 139, Fam: 36   |

**Variable type: numeric**

| skim\_variable   | n\_missing | complete\_rate |      mean |       sd |       p0 |       p25 |       p50 |       p75 |      p100 | hist  |
| :--------------- | ---------: | -------------: | --------: | -------: | -------: | --------: | --------: | --------: | --------: | :---- |
| Lot\_Frontage    |          0 |              1 |     57.00 |    33.69 |     0.00 |     41.00 |     62.00 |     78.00 |    313.00 | ▇▇▁▁▁ |
| Lot\_Area        |          0 |              1 |  10121.14 |  8120.98 |  1300.00 |   7313.00 |   9375.00 |  11512.00 | 215245.00 | ▇▁▁▁▁ |
| Year\_Built      |          0 |              1 |   1971.07 |    30.52 |  1872.00 |   1952.00 |   1973.00 |   2000.00 |   2010.00 | ▁▂▃▆▇ |
| Year\_Remod\_Add |          0 |              1 |   1984.42 |    20.79 |  1950.00 |   1966.00 |   1993.00 |   2004.00 |   2010.00 | ▅▂▂▃▇ |
| Mas\_Vnr\_Area   |          0 |              1 |    102.59 |   179.08 |     0.00 |      0.00 |      0.00 |    164.00 |   1600.00 | ▇▁▁▁▁ |
| BsmtFin\_SF\_1   |          0 |              1 |      4.16 |     2.24 |     1.00 |      3.00 |      3.00 |      7.00 |      7.00 | ▅▆▁▁▇ |
| BsmtFin\_SF\_2   |          0 |              1 |     52.20 |   172.62 |     0.00 |      0.00 |      0.00 |      0.00 |   1474.00 | ▇▁▁▁▁ |
| Bsmt\_Unf\_SF    |          0 |              1 |    558.79 |   437.52 |     0.00 |    219.00 |    467.00 |    797.00 |   2336.00 | ▇▅▂▁▁ |
| Total\_Bsmt\_SF  |          0 |              1 |   1050.67 |   425.77 |     0.00 |    784.00 |    992.00 |   1298.00 |   3206.00 | ▂▇▃▁▁ |
| First\_Flr\_SF   |          0 |              1 |   1158.36 |   383.60 |   334.00 |    880.00 |   1083.00 |   1384.00 |   3820.00 | ▇▇▁▁▁ |
| Second\_Flr\_SF  |          0 |              1 |    336.30 |   426.47 |     0.00 |      0.00 |      0.00 |    702.00 |   2065.00 | ▇▃▂▁▁ |
| Gr\_Liv\_Area    |          0 |              1 |   1499.56 |   494.55 |   334.00 |   1136.00 |   1441.00 |   1743.00 |   4676.00 | ▅▇▂▁▁ |
| Bsmt\_Full\_Bath |          0 |              1 |      0.43 |     0.53 |     0.00 |      0.00 |      0.00 |      1.00 |      3.00 | ▇▆▁▁▁ |
| Bsmt\_Half\_Bath |          0 |              1 |      0.06 |     0.25 |     0.00 |      0.00 |      0.00 |      0.00 |      2.00 | ▇▁▁▁▁ |
| Full\_Bath       |          0 |              1 |      1.57 |     0.56 |     0.00 |      1.00 |      2.00 |      2.00 |      4.00 | ▁▇▇▁▁ |
| Half\_Bath       |          0 |              1 |      0.38 |     0.50 |     0.00 |      0.00 |      0.00 |      1.00 |      2.00 | ▇▁▅▁▁ |
| Bedroom\_AbvGr   |          0 |              1 |      2.84 |     0.83 |     0.00 |      2.00 |      3.00 |      3.00 |      8.00 | ▁▇▂▁▁ |
| TotRms\_AbvGrd   |          0 |              1 |      6.42 |     1.54 |     2.00 |      5.00 |      6.00 |      7.00 |     14.00 | ▁▇▆▁▁ |
| Fireplaces       |          0 |              1 |      0.61 |     0.65 |     0.00 |      0.00 |      1.00 |      1.00 |      3.00 | ▇▇▁▁▁ |
| Garage\_Cars     |          0 |              1 |      1.77 |     0.77 |     0.00 |      1.00 |      2.00 |      2.00 |      5.00 | ▅▇▂▁▁ |
| Garage\_Area     |          0 |              1 |    474.14 |   215.76 |     0.00 |    325.00 |    480.00 |    576.00 |   1390.00 | ▂▇▃▁▁ |
| Wood\_Deck\_SF   |          0 |              1 |     92.98 |   123.01 |     0.00 |      0.00 |      0.00 |    168.00 |    857.00 | ▇▂▁▁▁ |
| Open\_Porch\_SF  |          0 |              1 |     47.42 |    68.23 |     0.00 |      0.00 |     26.00 |     70.00 |    742.00 | ▇▁▁▁▁ |
| Mo\_Sold         |          0 |              1 |      6.20 |     2.71 |     1.00 |      4.00 |      6.00 |      8.00 |     12.00 | ▅▆▇▃▃ |
| Year\_Sold       |          0 |              1 |   2007.79 |     1.31 |  2006.00 |   2007.00 |   2008.00 |   2009.00 |   2010.00 | ▇▇▇▇▃ |
| Sale\_Price      |          0 |              1 | 180996.28 | 80258.90 | 13100.00 | 129500.00 | 160000.00 | 213500.00 | 755000.00 | ▇▇▁▁▁ |
| Longitude        |          0 |              1 |   \-93.64 |     0.03 |  \-93.69 |   \-93.66 |   \-93.64 |   \-93.62 |   \-93.58 | ▅▅▇▆▁ |
| Latitude         |          0 |              1 |     42.03 |     0.02 |    41.99 |     42.02 |     42.03 |     42.05 |     42.06 | ▂▂▇▇▇ |

Berikut adalah tabulasi observasi pada masing-masing variabel yang
memiliki jumlah kategori \>= 10.

``` r
# MS_SubClass 
count(data_train, MS_SubClass) %>% arrange(n)
```

    ## # A tibble: 16 x 2
    ##    MS_SubClass                                   n
    ##    <fct>                                     <int>
    ##  1 One_and_Half_Story_PUD_All_Ages               1
    ##  2 One_Story_with_Finished_Attic_All_Ages        5
    ##  3 One_and_Half_Story_Unfinished_All_Ages       11
    ##  4 PUD_Multilevel_Split_Level_Foyer             14
    ##  5 Two_and_Half_Story_All_Ages                  17
    ##  6 Split_Foyer                                  32
    ##  7 Two_Family_conversion_All_Styles_and_Ages    43
    ##  8 Duplex_All_Styles_and_Ages                   66
    ##  9 Split_or_Multilevel                          75
    ## 10 One_Story_1945_and_Older                     91
    ## 11 Two_Story_PUD_1946_and_Newer                 96
    ## 12 Two_Story_1945_and_Older                     98
    ## 13 One_Story_PUD_1946_and_Newer                145
    ## 14 One_and_Half_Story_Finished_All_Ages        211
    ## 15 Two_Story_1946_and_Newer                    395
    ## 16 One_Story_1946_and_Newer_All_Styles         753

``` r
# Neighborhood
count(data_train, Neighborhood) %>% arrange(n)
```

    ## # A tibble: 27 x 2
    ##    Neighborhood                                n
    ##    <fct>                                   <int>
    ##  1 Green_Hills                                 2
    ##  2 Greens                                      7
    ##  3 Blueste                                     8
    ##  4 Northpark_Villa                            17
    ##  5 Briardale                                  18
    ##  6 Veenker                                    20
    ##  7 Bloomington_Heights                        21
    ##  8 South_and_West_of_Iowa_State_University    27
    ##  9 Meadow_Village                             29
    ## 10 Clear_Creek                                31
    ## # ... with 17 more rows

``` r
# Neighborhood
count(data_train, Exterior_1st) %>% arrange(n)
```

    ## # A tibble: 14 x 2
    ##    Exterior_1st     n
    ##    <fct>        <int>
    ##  1 PreCast          1
    ##  2 Stone            1
    ##  3 CBlock           2
    ##  4 BrkComm          5
    ##  5 Stucco          30
    ##  6 WdShing         31
    ##  7 AsbShng         37
    ##  8 BrkFace         65
    ##  9 CemntBd         92
    ## 10 Plywood        154
    ## 11 HdBoard        295
    ## 12 Wd Sdng        301
    ## 13 MetalSd        321
    ## 14 VinylSd        718

``` r
# Exterior_2nd
count(data_train, Exterior_2nd) %>% arrange(n)
```

    ## # A tibble: 16 x 2
    ##    Exterior_2nd     n
    ##    <fct>        <int>
    ##  1 AsphShn          1
    ##  2 PreCast          1
    ##  3 CBlock           2
    ##  4 Stone            2
    ##  5 ImStucc         10
    ##  6 Brk Cmn         17
    ##  7 AsbShng         31
    ##  8 Stucco          32
    ##  9 BrkFace         35
    ## 10 Wd Shng         50
    ## 11 CmentBd         92
    ## 12 Plywood        195
    ## 13 HdBoard        267
    ## 14 Wd Sdng        287
    ## 15 MetalSd        321
    ## 16 VinylSd        710

## Kovarian

Kovarian dapat dicek melalui visualisasi *heatmap* koefisien korelasi
(numerik) atau menggunakan *boxplot* (kontinu vs kategorikal)

``` r
plot_correlation(data_train, type = "continuous", 
                 cor_args = list(method = "spearman"))
```

![](lm-regression_files/figure-gfm/heatmap-1.png)<!-- -->

``` r
plot_boxplot(data_train, by = "Sale_Price", ncol = 2, nrow = 1)
```

![](lm-regression_files/figure-gfm/boxplot-1.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-2.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-3.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-4.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-5.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-6.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-7.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-8.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-9.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-10.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-11.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-12.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-13.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-14.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-15.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-16.png)<!-- -->![](lm-regression_files/figure-gfm/boxplot-17.png)<!-- -->

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
    secara bersamaan, lakukan sebelum *one-hot/dummy encode*.
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

Paket `recipes` ini memungkinkan kita untuk mengembangkan *bluprint
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

Langkah pertama adalah di mana kita menentukan *blueprint*. Dengan
proses ini, Kita memberikan formula model yang ingin kita buat (variabel
target, fitur, dan data yang menjadi dasarnya) dengan fungsi `recipe()`
dan kemudian kita secara bertahap menambahkan langkah-langkah rekayasa
fitur dengan fungsi `step_xxx()`.

``` r
blueprint <- recipe(Sale_Price ~., data = data_train) %>%
  # feature filtering
  step_nzv(all_nominal()) %>%
  step_corr(all_numeric(), -all_outcomes(), 
            threshold = 0.6, method = "spearman") %>%
  # imputation
  step_knnimpute(all_predictors(), neighbors = 7) %>%
  # lumping
  step_other(all_nominal(), threshold = 0.05) %>%
  # standardization
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) # %>%
  # feature extraction
  # step_pca(all_numeric(), -all_outcomes(), threshold = 0.7) %>%
  # dummy encode
  # step_dummy(all_nominal())
  
  
blueprint
```

    ## Data Recipe
    ## 
    ## Inputs:
    ## 
    ##       role #variables
    ##    outcome          1
    ##  predictor         73
    ## 
    ## Operations:
    ## 
    ## Sparse, unbalanced variable filter on all_nominal()
    ## Correlation filter on all_numeric(), -all_outcomes()
    ## K-nearest neighbor imputation for all_predictors()
    ## Collapsing factor levels for all_nominal()
    ## Centering for all_numeric(), -all_outcomes()
    ## Scaling for all_numeric(), -all_outcomes()

Selanjutnya, *blueprint* yang telah dibuat dilakukan *training* pada
data *training*.

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
    ##  predictor         73
    ## 
    ## Training data contained 2053 data points and no missing data.
    ## 
    ## Operations:
    ## 
    ## Sparse, unbalanced variable filter removed Street, Alley, Land_Contour, ... [trained]
    ## Correlation filter removed First_Flr_SF, TotRms_AbvGrd, ... [trained]
    ## K-nearest neighbor imputation for MS_Zoning, Lot_Frontage, Lot_Area, ... [trained]
    ## Collapsing factor levels for MS_SubClass, MS_Zoning, Lot_Shape, ... [trained]
    ## Centering for Lot_Frontage, Lot_Area, ... [trained]
    ## Scaling for Lot_Frontage, Lot_Area, ... [trained]

Langkah terakhir adalah mengaplikasikan *blueprint* pada data *training*
dan *test* menggunakan fungsi `bake()`.

``` r
baked_train <- bake(prepare, new_data = data_train)
baked_test <- bake(prepare, new_data = data_test)
baked_train
```

    ## # A tibble: 2,053 x 56
    ##    MS_SubClass MS_Zoning Lot_Frontage Lot_Area Lot_Shape Lot_Config Neighborhood
    ##    <fct>       <fct>            <dbl>    <dbl> <fct>     <fct>      <fct>       
    ##  1 One_Story_~ Resident~       2.49    2.67    Slightly~ Corner     North_Ames  
    ##  2 One_Story_~ other           0.683   0.185   Regular   Inside     North_Ames  
    ##  3 One_Story_~ Resident~       0.712   0.511   Slightly~ Corner     North_Ames  
    ##  4 Two_Story_~ Resident~       0.623  -0.0176  Slightly~ Inside     Gilbert     
    ##  5 One_Story_~ Resident~      -0.475  -0.640   Regular   Inside     other       
    ##  6 One_Story_~ Resident~      -0.534  -0.583   Slightly~ Inside     other       
    ##  7 Two_Story_~ Resident~       0.0892 -0.323   Regular   Inside     Gilbert     
    ##  8 Two_Story_~ Resident~       0.534  -0.0149  Slightly~ Corner     Gilbert     
    ##  9 Two_Story_~ Resident~       0.178  -0.212   Slightly~ Inside     Gilbert     
    ## 10 One_Story_~ Resident~       0.831   0.00676 Regular   Inside     Gilbert     
    ## # ... with 2,043 more rows, and 49 more variables: Condition_1 <fct>,
    ## #   Bldg_Type <fct>, House_Style <fct>, Overall_Cond <fct>,
    ## #   Year_Remod_Add <dbl>, Roof_Style <fct>, Exterior_1st <fct>,
    ## #   Exterior_2nd <fct>, Mas_Vnr_Type <fct>, Mas_Vnr_Area <dbl>,
    ## #   Exter_Cond <fct>, Foundation <fct>, Bsmt_Exposure <fct>,
    ## #   BsmtFin_Type_1 <fct>, BsmtFin_SF_1 <dbl>, BsmtFin_SF_2 <dbl>,
    ## #   Bsmt_Unf_SF <dbl>, Total_Bsmt_SF <dbl>, Heating_QC <fct>,
    ## #   Central_Air <fct>, Electrical <fct>, Second_Flr_SF <dbl>,
    ## #   Bsmt_Full_Bath <dbl>, Bsmt_Half_Bath <dbl>, Full_Bath <dbl>,
    ## #   Half_Bath <dbl>, Bedroom_AbvGr <dbl>, Kitchen_AbvGr <dbl>,
    ## #   Fireplaces <dbl>, Garage_Type <fct>, Garage_Finish <fct>,
    ## #   Garage_Area <dbl>, Garage_Cond <fct>, Paved_Drive <fct>,
    ## #   Wood_Deck_SF <dbl>, Open_Porch_SF <dbl>, Enclosed_Porch <dbl>,
    ## #   Three_season_porch <dbl>, Screen_Porch <dbl>, Pool_Area <dbl>, Fence <fct>,
    ## #   Misc_Val <dbl>, Mo_Sold <dbl>, Year_Sold <dbl>, Sale_Type <fct>,
    ## #   Sale_Condition <fct>, Longitude <dbl>, Latitude <dbl>, Sale_Price <int>

``` r
skim(baked_train)
```

|                                                  |              |
| :----------------------------------------------- | :----------- |
| Name                                             | baked\_train |
| Number of rows                                   | 2053         |
| Number of columns                                | 56           |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |              |
| Column type frequency:                           |              |
| factor                                           | 27           |
| numeric                                          | 29           |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |              |
| Group variables                                  | None         |

Data summary

**Variable type: factor**

| skim\_variable   | n\_missing | complete\_rate | ordered | n\_unique | top\_counts                              |
| :--------------- | ---------: | -------------: | :------ | --------: | :--------------------------------------- |
| MS\_SubClass     |          0 |              1 | FALSE   |         5 | One: 753, oth: 549, Two: 395, One: 211   |
| MS\_Zoning       |          0 |              1 | FALSE   |         3 | Res: 1571, Res: 338, oth: 144            |
| Lot\_Shape       |          0 |              1 | FALSE   |         3 | Reg: 1321, Sli: 666, oth: 66             |
| Lot\_Config      |          0 |              1 | FALSE   |         4 | Ins: 1494, Cor: 361, Cul: 127, oth: 71   |
| Neighborhood     |          0 |              1 | FALSE   |         8 | oth: 892, Nor: 298, Col: 187, Old: 171   |
| Condition\_1     |          0 |              1 | FALSE   |         3 | Nor: 1768, oth: 165, Fee: 120            |
| Bldg\_Type       |          0 |              1 | FALSE   |         3 | One: 1692, oth: 183, Twn: 178            |
| House\_Style     |          0 |              1 | FALSE   |         4 | One: 1030, Two: 614, One: 233, oth: 176  |
| Overall\_Cond    |          0 |              1 | FALSE   |         4 | Ave: 1168, Abo: 370, Goo: 280, oth: 235  |
| Roof\_Style      |          0 |              1 | FALSE   |         3 | Gab: 1627, Hip: 387, oth: 39             |
| Exterior\_1st    |          0 |              1 | FALSE   |         6 | Vin: 718, Met: 321, Wd : 301, HdB: 295   |
| Exterior\_2nd    |          0 |              1 | FALSE   |         6 | Vin: 710, Met: 321, Wd : 287, oth: 273   |
| Mas\_Vnr\_Type   |          0 |              1 | FALSE   |         4 | Non: 1231, Brk: 624, Sto: 179, oth: 19   |
| Exter\_Cond      |          0 |              1 | FALSE   |         3 | Typ: 1792, Goo: 204, oth: 57             |
| Foundation       |          0 |              1 | FALSE   |         4 | PCo: 916, CBl: 870, Brk: 226, oth: 41    |
| Bsmt\_Exposure   |          0 |              1 | FALSE   |         5 | No: 1348, Av: 284, Gd: 205, Mn: 163      |
| BsmtFin\_Type\_1 |          0 |              1 | FALSE   |         7 | GLQ: 599, Unf: 588, ALQ: 313, Rec: 207   |
| Heating\_QC      |          0 |              1 | FALSE   |         4 | Exc: 1052, Typ: 593, Goo: 340, oth: 68   |
| Central\_Air     |          0 |              1 | FALSE   |         2 | Y: 1916, N: 137                          |
| Electrical       |          0 |              1 | FALSE   |         3 | SBr: 1873, Fus: 140, oth: 40             |
| Garage\_Type     |          0 |              1 | FALSE   |         5 | Att: 1230, Det: 534, Bui: 124, No\_: 109 |
| Garage\_Finish   |          0 |              1 | FALSE   |         4 | Unf: 863, RFn: 567, Fin: 512, No\_: 111  |
| Garage\_Cond     |          0 |              1 | FALSE   |         3 | Typ: 1870, No\_: 111, oth: 72            |
| Paved\_Drive     |          0 |              1 | FALSE   |         3 | Pav: 1862, Dir: 140, oth: 51             |
| Fence            |          0 |              1 | FALSE   |         3 | No\_: 1646, Min: 243, oth: 164           |
| Sale\_Type       |          0 |              1 | FALSE   |         3 | WD : 1775, New: 159, oth: 119            |
| Sale\_Condition  |          0 |              1 | FALSE   |         4 | Nor: 1692, Par: 164, Abn: 139, oth: 58   |

**Variable type: numeric**

| skim\_variable       | n\_missing | complete\_rate |     mean |      sd |       p0 |       p25 |       p50 |       p75 |      p100 | hist  |
| :------------------- | ---------: | -------------: | -------: | ------: | -------: | --------: | --------: | --------: | --------: | :---- |
| Lot\_Frontage        |          0 |              1 |      0.0 |     1.0 |   \-1.69 |    \-0.47 |   1.5e-01 |      0.62 |      7.60 | ▇▇▁▁▁ |
| Lot\_Area            |          0 |              1 |      0.0 |     1.0 |   \-1.09 |    \-0.35 | \-9.0e-02 |      0.17 |     25.26 | ▇▁▁▁▁ |
| Year\_Remod\_Add     |          0 |              1 |      0.0 |     1.0 |   \-1.66 |    \-0.89 |   4.1e-01 |      0.94 |      1.23 | ▅▂▂▃▇ |
| Mas\_Vnr\_Area       |          0 |              1 |      0.0 |     1.0 |   \-0.57 |    \-0.57 | \-5.7e-01 |      0.34 |      8.36 | ▇▁▁▁▁ |
| BsmtFin\_SF\_1       |          0 |              1 |      0.0 |     1.0 |   \-1.41 |    \-0.52 | \-5.2e-01 |      1.27 |      1.27 | ▅▆▁▁▇ |
| BsmtFin\_SF\_2       |          0 |              1 |      0.0 |     1.0 |   \-0.30 |    \-0.30 | \-3.0e-01 |    \-0.30 |      8.24 | ▇▁▁▁▁ |
| Bsmt\_Unf\_SF        |          0 |              1 |      0.0 |     1.0 |   \-1.28 |    \-0.78 | \-2.1e-01 |      0.54 |      4.06 | ▇▅▂▁▁ |
| Total\_Bsmt\_SF      |          0 |              1 |      0.0 |     1.0 |   \-2.47 |    \-0.63 | \-1.4e-01 |      0.58 |      5.06 | ▂▇▃▁▁ |
| Second\_Flr\_SF      |          0 |              1 |      0.0 |     1.0 |   \-0.79 |    \-0.79 | \-7.9e-01 |      0.86 |      4.05 | ▇▃▂▁▁ |
| Bsmt\_Full\_Bath     |          0 |              1 |      0.0 |     1.0 |   \-0.82 |    \-0.82 | \-8.2e-01 |      1.09 |      4.89 | ▇▆▁▁▁ |
| Bsmt\_Half\_Bath     |          0 |              1 |      0.0 |     1.0 |   \-0.25 |    \-0.25 | \-2.5e-01 |    \-0.25 |      7.73 | ▇▁▁▁▁ |
| Full\_Bath           |          0 |              1 |      0.0 |     1.0 |   \-2.81 |    \-1.01 |   7.8e-01 |      0.78 |      4.36 | ▁▇▇▁▁ |
| Half\_Bath           |          0 |              1 |      0.0 |     1.0 |   \-0.75 |    \-0.75 | \-7.5e-01 |      1.24 |      3.23 | ▇▁▅▁▁ |
| Bedroom\_AbvGr       |          0 |              1 |      0.0 |     1.0 |   \-3.44 |    \-1.02 |   1.9e-01 |      0.19 |      6.24 | ▁▇▂▁▁ |
| Kitchen\_AbvGr       |          0 |              1 |      0.0 |     1.0 |   \-5.06 |    \-0.19 | \-1.9e-01 |    \-0.19 |      9.53 | ▁▇▁▁▁ |
| Fireplaces           |          0 |              1 |      0.0 |     1.0 |   \-0.94 |    \-0.94 |   6.1e-01 |      0.61 |      3.71 | ▇▇▁▁▁ |
| Garage\_Area         |          0 |              1 |      0.0 |     1.0 |   \-2.20 |    \-0.69 |   3.0e-02 |      0.47 |      4.24 | ▂▇▃▁▁ |
| Wood\_Deck\_SF       |          0 |              1 |      0.0 |     1.0 |   \-0.76 |    \-0.76 | \-7.6e-01 |      0.61 |      6.21 | ▇▂▁▁▁ |
| Open\_Porch\_SF      |          0 |              1 |      0.0 |     1.0 |   \-0.70 |    \-0.70 | \-3.1e-01 |      0.33 |     10.18 | ▇▁▁▁▁ |
| Enclosed\_Porch      |          0 |              1 |      0.0 |     1.0 |   \-0.36 |    \-0.36 | \-3.6e-01 |    \-0.36 |     15.01 | ▇▁▁▁▁ |
| Three\_season\_porch |          0 |              1 |      0.0 |     1.0 |   \-0.11 |    \-0.11 | \-1.1e-01 |    \-0.11 |     18.20 | ▇▁▁▁▁ |
| Screen\_Porch        |          0 |              1 |      0.0 |     1.0 |   \-0.28 |    \-0.28 | \-2.8e-01 |    \-0.28 |     10.27 | ▇▁▁▁▁ |
| Pool\_Area           |          0 |              1 |      0.0 |     1.0 |   \-0.06 |    \-0.06 | \-6.0e-02 |    \-0.06 |     20.62 | ▇▁▁▁▁ |
| Misc\_Val            |          0 |              1 |      0.0 |     1.0 |   \-0.09 |    \-0.09 | \-9.0e-02 |    \-0.09 |     30.34 | ▇▁▁▁▁ |
| Mo\_Sold             |          0 |              1 |      0.0 |     1.0 |   \-1.92 |    \-0.81 | \-7.0e-02 |      0.67 |      2.14 | ▅▆▇▃▃ |
| Year\_Sold           |          0 |              1 |      0.0 |     1.0 |   \-1.37 |    \-0.61 |   1.6e-01 |      0.92 |      1.69 | ▇▇▇▇▃ |
| Longitude            |          0 |              1 |      0.0 |     1.0 |   \-1.96 |    \-0.68 |   4.0e-02 |      0.80 |      2.57 | ▅▅▇▆▁ |
| Latitude             |          0 |              1 |      0.0 |     1.0 |   \-2.61 |    \-0.67 |   1.0e-02 |      0.83 |      1.58 | ▂▂▇▇▇ |
| Sale\_Price          |          0 |              1 | 180996.3 | 80258.9 | 13100.00 | 129500.00 |   1.6e+05 | 213500.00 | 755000.00 | ▇▇▁▁▁ |

# Regresi Linier

Regresi linier adalah pendekatan yang sangat sederhana untuk
pembelajaran yang diawasi. Secara khusus, regresi linier adalah alat
yang berguna untuk memprediksi respons kuantitatif. Regresi linier telah
ada sejak lama dan merupakan topik buku teks yang tak terhitung
banyaknya. Meskipun mungkin tampak agak membosankan dibandingkan dengan
beberapa pendekatan pembelajaran statistik yang lebih modern yang
dijelaskan dalam tutorial selanjutnya, regresi linier masih merupakan
metode pembelajaran statistik yang bermanfaat dan banyak digunakan.
Selain itu, ini berfungsi sebagai titik awal yang baik untuk pendekatan
yang lebih baru: seperti yang akan kita lihat dalam tutorial
selanjutnya, banyak pendekatan pembelajaran statistik mewah dapat
dilihat sebagai generalisasi atau perpanjangan dari regresi linier.
Akibatnya, pentingnya memiliki pemahaman yang baik tentang regresi
linier sebelum mempelajari metode pembelajaran yang lebih kompleks tidak
dapat dilebih-lebihkan.

## Regresi Linier

``` r
model_lm <- lm(Sale_Price~., data = baked_train)
model_glm <- glm(Sale_Price~., data = baked_train, family = "gaussian")
model_caret <- train(Sale_Price~., data = baked_train, method = 'lm')
```

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
  savePredictions = TRUE,
  allowParallel = TRUE
)
```

Setelah parameter *tuning* dan validasi silang dispesifikasikan, proses
training dilakukan menggunakan fungsi `train()`.

``` r
system.time(
model_fit_cv <- train(
  blueprint, 
  data = data_train, 
  method = "lm", 
  trControl = cv, 
  metric = "RMSE"
  )
)
```

    ##    user  system elapsed 
    ##   20.60    0.20   30.41

``` r
model_fit_cv
```

    ## Linear Regression 
    ## 
    ## 2053 samples
    ##   73 predictor
    ## 
    ## Recipe steps: nzv, corr, knnimpute, other, center, scale 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1848, 1847, 1847, 1847, 1847, 1848, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   31061.91  0.8515971  20838.55
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

Proses *training* berlangsung selama 16.579 detik. Model terbaik dipilih
berdasarkan nilai **RMSE** terbesar. Nilai **RMSE** rata-rata model
terbaik adalah sebagai berikut:

``` r
lm_rmse <- model_fit_cv$results %>%
  arrange(RMSE) %>%
  slice(1) %>%
  dplyr::select(RMSE) %>%
  pull()
lm_rmse
```

    ## [1] 31061.91

Berdasarkan hasil yang diperoleh, nilai **RMSE** rata-rata model sebesar
3.106190810^{4}.

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

Untuk melihat performa sebuah model regresi adalah dengan melihat
visualisasi nilai residunya. Berikut adalah sintaks yang digunakan:

``` r
plot(model_fit)
```

![](lm-regression_files/figure-gfm/lm-res-vis-1.png)<!-- -->![](lm-regression_files/figure-gfm/lm-res-vis-2.png)<!-- -->![](lm-regression_files/figure-gfm/lm-res-vis-3.png)<!-- -->![](lm-regression_files/figure-gfm/lm-res-vis-4.png)<!-- -->

Ringkasan model ditampilkan sebagai berikut:

``` r
summary(model_fit)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -390628  -14385    -458   12836  231932 
    ## 
    ## Coefficients: (2 not defined because of singularities)
    ##                                                  Estimate Std. Error t value
    ## (Intercept)                                     135470.21   18454.57   7.341
    ## MS_SubClassOne_and_Half_Story_Finished_All_Ages -13370.09    6737.56  -1.984
    ## MS_SubClassTwo_Story_1946_and_Newer              -7606.63    4175.76  -1.822
    ## MS_SubClassOne_Story_PUD_1946_and_Newer            332.07    5209.74   0.064
    ## MS_SubClassother                                 -4969.69    3620.66  -1.373
    ## MS_ZoningResidential_Medium_Density              -5797.36    2806.00  -2.066
    ## MS_Zoningother                                  -13241.80    4246.32  -3.118
    ## Lot_Frontage                                      1953.62     794.01   2.460
    ## Lot_Area                                          3106.17     810.92   3.830
    ## Lot_ShapeSlightly_Irregular                       5916.73    1710.33   3.459
    ## Lot_Shapeother                                    7988.56    4262.61   1.874
    ## Lot_ConfigCulDSac                                 4835.73    3420.04   1.414
    ## Lot_ConfigInside                                 -2373.83    1852.29  -1.282
    ## Lot_Configother                                  -7730.04    4082.62  -1.893
    ## NeighborhoodCollege_Creek                        11902.93    5106.04   2.331
    ## NeighborhoodOld_Town                                37.88    4128.78   0.009
    ## NeighborhoodEdwards                               6193.32    4441.55   1.394
    ## NeighborhoodSomerset                             23555.46    5391.22   4.369
    ## NeighborhoodNorthridge_Heights                   36009.18    4581.16   7.860
    ## NeighborhoodGilbert                             -17059.74    4394.05  -3.882
    ## Neighborhoodother                                15988.02    2988.81   5.349
    ## Condition_1Norm                                  11418.32    3046.00   3.749
    ## Condition_1other                                  4163.21    3774.51   1.103
    ## Bldg_TypeTwnhsE                                 -14449.95    4798.28  -3.011
    ## Bldg_Typeother                                  -20943.07    4005.25  -5.229
    ## House_StyleOne_Story                             -3853.20    6489.19  -0.594
    ## House_StyleTwo_Story                             -6538.12    5974.84  -1.094
    ## House_Styleother                                 -3646.96    6428.86  -0.567
    ## Overall_CondAbove_Average                         1597.67    2094.85   0.763
    ## Overall_CondGood                                  7542.42    2457.93   3.069
    ## Overall_Condother                                -1515.76    2691.38  -0.563
    ## Year_Remod_Add                                    7926.54    1064.31   7.448
    ## Roof_StyleHip                                     9895.95    1943.53   5.092
    ## Roof_Styleother                                  -5911.30    5133.22  -1.152
    ## Exterior_1stMetalSd                              -8371.50    8757.61  -0.956
    ## Exterior_1stPlywood                                371.80    4692.56   0.079
    ## Exterior_1stVinylSd                              -3981.76    7486.65  -0.532
    ## Exterior_1stWd Sdng                                952.72    5409.46   0.176
    ## Exterior_1stother                                17759.41    5027.02   3.533
    ## Exterior_2ndMetalSd                              15252.07    8727.55   1.748
    ## Exterior_2ndPlywood                              -2055.31    4463.79  -0.460
    ## Exterior_2ndVinylSd                               9100.91    7544.18   1.206
    ## Exterior_2ndWd Sdng                               6145.14    5543.01   1.109
    ## Exterior_2ndother                                -3113.01    5004.82  -0.622
    ## Mas_Vnr_TypeNone                                 10960.80    2224.39   4.928
    ## Mas_Vnr_TypeStone                                 7623.52    2858.03   2.667
    ## Mas_Vnr_Typeother                                -7475.02    7235.18  -1.033
    ## Mas_Vnr_Area                                      8617.81    1035.68   8.321
    ## Exter_CondTypical                                 2521.67    2432.83   1.037
    ## Exter_Condother                                  -5911.76    4847.95  -1.219
    ## FoundationCBlock                                 -2228.94    2918.11  -0.764
    ## FoundationPConc                                   4555.98    3181.99   1.432
    ## Foundationother                                   2371.60    6709.73   0.353
    ## Bsmt_ExposureGd                                  17135.23    2966.88   5.776
    ## Bsmt_ExposureMn                                 -10072.53    3191.76  -3.156
    ## Bsmt_ExposureNo                                  -9298.15    2337.27  -3.978
    ## Bsmt_Exposureother                              -12032.70   30454.47  -0.395
    ## BsmtFin_Type_1BLQ                                 2058.74    2907.12   0.708
    ## BsmtFin_Type_1GLQ                                14640.42    2532.56   5.781
    ## BsmtFin_Type_1LwQ                                  466.66    3559.35   0.131
    ## BsmtFin_Type_1Rec                                 -112.73    2889.45  -0.039
    ## BsmtFin_Type_1Unf                                11682.50    2913.92   4.009
    ## BsmtFin_Type_1other                              61895.60   31165.43   1.986
    ## BsmtFin_SF_1                                           NA         NA      NA
    ## BsmtFin_SF_2                                     -2533.15     783.53  -3.233
    ## Bsmt_Unf_SF                                     -10753.64    1343.70  -8.003
    ## Total_Bsmt_SF                                    36762.15    1621.53  22.671
    ## Heating_QCGood                                   -3257.59    2117.32  -1.539
    ## Heating_QCTypical                                -6862.11    2070.99  -3.313
    ## Heating_QCother                                 -13138.35    4255.44  -3.087
    ## Central_AirY                                     -1469.58    3424.94  -0.429
    ## ElectricalSBrkr                                    524.82    2959.85   0.177
    ## Electricalother                                    813.81    5652.57   0.144
    ## Second_Flr_SF                                    23654.11    1963.87  12.045
    ## Bsmt_Full_Bath                                     215.42    1032.16   0.209
    ## Bsmt_Half_Bath                                    -916.16     736.67  -1.244
    ## Full_Bath                                         5121.31    1118.42   4.579
    ## Half_Bath                                         3201.32     985.97   3.247
    ## Bedroom_AbvGr                                    -1409.77     977.07  -1.443
    ## Kitchen_AbvGr                                    -2002.92     947.82  -2.113
    ## Fireplaces                                        6588.96     832.84   7.911
    ## Garage_TypeBuiltIn                                4341.10    3339.77   1.300
    ## Garage_TypeDetchd                                -1440.25    2311.85  -0.623
    ## Garage_TypeNo_Garage                              1593.61   22395.10   0.071
    ## Garage_Typeother                                  -765.94    4537.42  -0.169
    ## Garage_FinishNo_Garage                            7068.90   22580.96   0.313
    ## Garage_FinishRFn                                 -8584.77    1993.12  -4.307
    ## Garage_FinishUnf                                 -3334.87    2390.23  -1.395
    ## Garage_Area                                       8939.79    1105.63   8.086
    ## Garage_CondTypical                                5068.17    3949.87   1.283
    ## Garage_Condother                                       NA         NA      NA
    ## Paved_DrivePaved                                  7191.96    3354.29   2.144
    ## Paved_Driveother                                  8387.24    5168.31   1.623
    ## Wood_Deck_SF                                      1201.84     751.84   1.599
    ## Open_Porch_SF                                     -272.54     751.33  -0.363
    ## Enclosed_Porch                                    -189.78     754.07  -0.252
    ## Three_season_porch                                 643.36     679.05   0.947
    ## Screen_Porch                                      1939.64     708.28   2.739
    ## Pool_Area                                         3207.12     698.74   4.590
    ## FenceNo_Fence                                    -2734.42    2234.39  -1.224
    ## Fenceother                                       -2571.23    3129.91  -0.822
    ## Misc_Val                                           341.88     674.63   0.507
    ## Mo_Sold                                           -135.98     686.58  -0.198
    ## Year_Sold                                         -950.39     701.87  -1.354
    ## Sale_TypeWD                                       2153.89   14082.66   0.153
    ## Sale_Typeother                                    2569.31   14164.57   0.181
    ## Sale_ConditionNormal                              9929.85    2853.89   3.479
    ## Sale_ConditionPartial                            31646.15   14049.63   2.252
    ## Sale_Conditionother                               6399.76    4905.22   1.305
    ## Longitude                                         2375.16    1240.71   1.914
    ## Latitude                                          6836.99    1010.34   6.767
    ##                                                 Pr(>|t|)    
    ## (Intercept)                                     3.11e-13 ***
    ## MS_SubClassOne_and_Half_Story_Finished_All_Ages 0.047350 *  
    ## MS_SubClassTwo_Story_1946_and_Newer             0.068667 .  
    ## MS_SubClassOne_Story_PUD_1946_and_Newer         0.949184    
    ## MS_SubClassother                                0.170038    
    ## MS_ZoningResidential_Medium_Density             0.038955 *  
    ## MS_Zoningother                                  0.001845 ** 
    ## Lot_Frontage                                    0.013963 *  
    ## Lot_Area                                        0.000132 ***
    ## Lot_ShapeSlightly_Irregular                     0.000553 ***
    ## Lot_Shapeother                                  0.061067 .  
    ## Lot_ConfigCulDSac                               0.157539    
    ## Lot_ConfigInside                                0.200148    
    ## Lot_Configother                                 0.058453 .  
    ## NeighborhoodCollege_Creek                       0.019847 *  
    ## NeighborhoodOld_Town                            0.992681    
    ## NeighborhoodEdwards                             0.163354    
    ## NeighborhoodSomerset                            1.31e-05 ***
    ## NeighborhoodNorthridge_Heights                  6.30e-15 ***
    ## NeighborhoodGilbert                             0.000107 ***
    ## Neighborhoodother                               9.87e-08 ***
    ## Condition_1Norm                                 0.000183 ***
    ## Condition_1other                                0.270172    
    ## Bldg_TypeTwnhsE                                 0.002633 ** 
    ## Bldg_Typeother                                  1.89e-07 ***
    ## House_StyleOne_Story                            0.552724    
    ## House_StyleTwo_Story                            0.273970    
    ## House_Styleother                                0.570589    
    ## Overall_CondAbove_Average                       0.445757    
    ## Overall_CondGood                                0.002180 ** 
    ## Overall_Condother                               0.573371    
    ## Year_Remod_Add                                  1.42e-13 ***
    ## Roof_StyleHip                                   3.89e-07 ***
    ## Roof_Styleother                                 0.249636    
    ## Exterior_1stMetalSd                             0.339236    
    ## Exterior_1stPlywood                             0.936857    
    ## Exterior_1stVinylSd                             0.594892    
    ## Exterior_1stWd Sdng                             0.860217    
    ## Exterior_1stother                               0.000421 ***
    ## Exterior_2ndMetalSd                             0.080695 .  
    ## Exterior_2ndPlywood                             0.645252    
    ## Exterior_2ndVinylSd                             0.227830    
    ## Exterior_2ndWd Sdng                             0.267727    
    ## Exterior_2ndother                               0.534014    
    ## Mas_Vnr_TypeNone                                9.03e-07 ***
    ## Mas_Vnr_TypeStone                               0.007707 ** 
    ## Mas_Vnr_Typeother                               0.301662    
    ## Mas_Vnr_Area                                     < 2e-16 ***
    ## Exter_CondTypical                               0.300090    
    ## Exter_Condother                                 0.222827    
    ## FoundationCBlock                                0.445061    
    ## FoundationPConc                                 0.152362    
    ## Foundationother                                 0.723784    
    ## Bsmt_ExposureGd                                 8.91e-09 ***
    ## Bsmt_ExposureMn                                 0.001625 ** 
    ## Bsmt_ExposureNo                                 7.20e-05 ***
    ## Bsmt_Exposureother                              0.692809    
    ## BsmtFin_Type_1BLQ                               0.478924    
    ## BsmtFin_Type_1GLQ                               8.64e-09 ***
    ## BsmtFin_Type_1LwQ                               0.895704    
    ## BsmtFin_Type_1Rec                               0.968882    
    ## BsmtFin_Type_1Unf                               6.32e-05 ***
    ## BsmtFin_Type_1other                             0.047170 *  
    ## BsmtFin_SF_1                                          NA    
    ## BsmtFin_SF_2                                    0.001246 ** 
    ## Bsmt_Unf_SF                                     2.07e-15 ***
    ## Total_Bsmt_SF                                    < 2e-16 ***
    ## Heating_QCGood                                  0.124079    
    ## Heating_QCTypical                               0.000938 ***
    ## Heating_QCother                                 0.002048 ** 
    ## Central_AirY                                    0.667910    
    ## ElectricalSBrkr                                 0.859281    
    ## Electricalother                                 0.885537    
    ## Second_Flr_SF                                    < 2e-16 ***
    ## Bsmt_Full_Bath                                  0.834699    
    ## Bsmt_Half_Bath                                  0.213780    
    ## Full_Bath                                       4.97e-06 ***
    ## Half_Bath                                       0.001187 ** 
    ## Bedroom_AbvGr                                   0.149220    
    ## Kitchen_AbvGr                                   0.034712 *  
    ## Fireplaces                                      4.23e-15 ***
    ## Garage_TypeBuiltIn                              0.193817    
    ## Garage_TypeDetchd                               0.533367    
    ## Garage_TypeNo_Garage                            0.943279    
    ## Garage_Typeother                                0.865968    
    ## Garage_FinishNo_Garage                          0.754279    
    ## Garage_FinishRFn                                1.74e-05 ***
    ## Garage_FinishUnf                                0.163112    
    ## Garage_Area                                     1.08e-15 ***
    ## Garage_CondTypical                              0.199601    
    ## Garage_Condother                                      NA    
    ## Paved_DrivePaved                                0.032148 *  
    ## Paved_Driveother                                0.104790    
    ## Wood_Deck_SF                                    0.110087    
    ## Open_Porch_SF                                   0.716832    
    ## Enclosed_Porch                                  0.801318    
    ## Three_season_porch                              0.343531    
    ## Screen_Porch                                    0.006228 ** 
    ## Pool_Area                                       4.72e-06 ***
    ## FenceNo_Fence                                   0.221181    
    ## Fenceother                                      0.411461    
    ## Misc_Val                                        0.612372    
    ## Mo_Sold                                         0.843024    
    ## Year_Sold                                       0.175865    
    ## Sale_TypeWD                                     0.878457    
    ## Sale_Typeother                                  0.856081    
    ## Sale_ConditionNormal                            0.000514 ***
    ## Sale_ConditionPartial                           0.024405 *  
    ## Sale_Conditionother                             0.192155    
    ## Longitude                                       0.055722 .  
    ## Latitude                                        1.73e-11 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 29760 on 1944 degrees of freedom
    ## Multiple R-squared:  0.8697, Adjusted R-squared:  0.8625 
    ## F-statistic: 120.2 on 108 and 1944 DF,  p-value: < 2.2e-16

Model yang dihasilkan selanjutnya dapat kita uji lagi menggunakan data
baru. Berikut adalah perhitungan nilai **RMSE** model pada data *test*.

``` r
pred_test <- predict(model_fit, baked_test)

## RMSE
rmse <- RMSE(pred_test, baked_test$Sale_Price, na.rm = TRUE)
rmse
```

    ## [1] 41783.26

Berdasarkan hasil evaluasi diperoleh nilai akurasi sebesar
4.178325810^{4}

## Interpretasi Fitur

Untuk mengetahui variabel yang paling berpengaruh secara global terhadap
hasil prediksi model, kita dapat menggunakan plot *variable importance*.

``` r
vi <- vip(model_fit_cv, num_features = 10)
vi
```

![](lm-regression_files/figure-gfm/lm-vip-1.png)<!-- -->

Berdasarkan terdapat 4 buah variabel yang berpengaruh besar terhadap
prediksi yang dihasilkan oleh model, antara lain: Total\_Bsmt\_SF,
Second\_Flr\_SF, Mas\_Vnr\_Area, Garage\_Area. Untuk melihat efek
masing-masing variabel tersebut, jalankan perintah berikut:

``` r
p1 <- pdp::partial(model_fit_cv, pred.var = as.character(vi$data[1,1] %>% pull())) %>% 
  autoplot() 

p2 <- pdp::partial(model_fit_cv, pred.var = "Neighborhood") %>% 
  autoplot()

p3 <- pdp::partial(model_fit_cv, pred.var = as.character(vi$data[3,1] %>% pull())) %>% 
  autoplot()
  

p4 <- pdp::partial(model_fit_cv, pred.var = as.character(vi$data[4,1] %>% pull())) %>% 
  autoplot()

grid.arrange(p1, p2, p3, p4, nrow = 2)
```

![](lm-regression_files/figure-gfm/lm-pdp-1.png)<!-- -->
