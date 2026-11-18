# ============================================================
# EDA DATASET NYC AIRBNB OPEN DATA
# Dataset : AB_NYC_2019.csv
# Jumlah  : 48.895 baris x 16 kolom
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 0. KONFIGURASI
# ============================================================

FILE_PATH = "AB_NYC_2019 - Asli.csv"
OUTPUT_DIR = "hasil_eda"

# Membuat folder hasil jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pengaturan tampilan
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 100)

sns.set_theme(style="whitegrid")


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("NYC AIRBNB OPEN DATA - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print("\nERROR: File dataset tidak ditemukan!")
    print(f"Pastikan file '{FILE_PATH}' berada satu folder dengan program.")
    exit()

print("\nDataset berhasil dibaca.")
print(f"Nama file      : {FILE_PATH}")
print(f"Jumlah baris   : {df.shape[0]:,}")
print(f"Jumlah kolom   : {df.shape[1]}")


# ============================================================
# 1. DATA UNDERSTANDING
# ============================================================

print("\n\n" + "=" * 70)
print("1. DATA UNDERSTANDING")
print("=" * 70)

# ------------------------------------------------------------
# 1.1 Jumlah baris dan kolom
# ------------------------------------------------------------

jumlah_baris, jumlah_kolom = df.shape

print("\n1.1 JUMLAH BARIS DAN KOLOM")
print("-" * 70)
print(f"Jumlah baris  : {jumlah_baris:,}")
print(f"Jumlah kolom  : {jumlah_kolom}")

# ------------------------------------------------------------
# 1.2 Lima data pertama
# ------------------------------------------------------------

print("\n1.2 LIMA DATA PERTAMA")
print("-" * 70)
print(df.head())

# ------------------------------------------------------------
# 1.3 Lima data terakhir
# ------------------------------------------------------------

print("\n1.3 LIMA DATA TERAKHIR")
print("-" * 70)
print(df.tail())

# ------------------------------------------------------------
# 1.4 Informasi dataset
# ------------------------------------------------------------

print("\n1.4 INFORMASI DATASET")
print("-" * 70)
print(df.info())

# ------------------------------------------------------------
# 1.5 Tipe data
# ------------------------------------------------------------

print("\n1.5 TIPE DATA SETIAP VARIABEL")
print("-" * 70)

dtype_table = pd.DataFrame({
    "Kolom": df.columns,
    "Tipe Data": df.dtypes.astype(str).values,
    "Jumlah Nilai": df.count().values,
    "Missing": df.isnull().sum().values
})

print(dtype_table.to_string(index=False))

# ------------------------------------------------------------
# 1.6 Deskripsi variabel
# ------------------------------------------------------------

print("\n1.6 DESKRIPSI VARIABEL")
print("-" * 70)

deskripsi_variabel = {
    "id": "ID unik untuk setiap listing Airbnb.",
    "name": "Nama atau judul listing Airbnb.",
    "host_id": "ID unik pemilik/host.",
    "host_name": "Nama host.",
    "neighbourhood_group": "Kelompok wilayah/bagian kota tempat listing berada.",
    "neighbourhood": "Nama lingkungan tempat listing berada.",
    "latitude": "Koordinat lintang lokasi listing.",
    "longitude": "Koordinat bujur lokasi listing.",
    "room_type": "Jenis tempat yang disewakan.",
    "price": "Harga sewa per malam dalam USD.",
    "minimum_nights": "Jumlah minimum malam yang harus dipesan.",
    "number_of_reviews": "Jumlah review yang diterima listing.",
    "last_review": "Tanggal review terakhir.",
    "reviews_per_month": "Rata-rata jumlah review per bulan.",
    "calculated_host_listings_count": "Jumlah listing yang dimiliki host.",
    "availability_365": "Jumlah hari dalam satu tahun ketika listing tersedia."
}

for kolom in df.columns:
    if kolom in deskripsi_variabel:
        print(f"{kolom:35} : {deskripsi_variabel[kolom]}")


# ============================================================
# 2. DATA QUALITY
# ============================================================

print("\n\n" + "=" * 70)
print("2. DATA QUALITY")
print("=" * 70)

# ------------------------------------------------------------
# 2.1 Missing Value
# ------------------------------------------------------------

print("\n2.1 MISSING VALUE")
print("-" * 70)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Kolom": df.columns,
    "Missing": missing_count.values,
    "Persentase (%)": missing_percent.round(2).values
})

missing_table = missing_table.sort_values(
    by="Missing",
    ascending=False
)

print(missing_table.to_string(index=False))

total_missing = df.isnull().sum().sum()

print(f"\nTotal missing value : {total_missing:,}")

# ------------------------------------------------------------
# 2.2 Duplicate
# ------------------------------------------------------------

print("\n2.2 DUPLICATE")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print(f"Jumlah data duplikat : {duplicate_count:,}")

if duplicate_count == 0:
    print("Tidak ditemukan data duplikat.")
else:
    print("Ditemukan data duplikat.")

# ------------------------------------------------------------
# 2.3 Invalid Value
# ------------------------------------------------------------

print("\n2.3 INVALID VALUE")
print("-" * 70)

# Harga tidak boleh negatif
invalid_price = (df["price"] < 0).sum()

# Minimum nights tidak boleh 0 atau negatif
invalid_minimum_nights = (df["minimum_nights"] <= 0).sum()

# Latitude NYC secara umum berada sekitar 40-41
invalid_latitude = (
    (df["latitude"] < 40) |
    (df["latitude"] > 41)
).sum()

# Longitude NYC secara umum berada sekitar -75 sampai -73
invalid_longitude = (
    (df["longitude"] < -75) |
    (df["longitude"] > -73)
).sum()

# Availability seharusnya 0-365
invalid_availability = (
    (df["availability_365"] < 0) |
    (df["availability_365"] > 365)
).sum()

# Number of reviews tidak boleh negatif
invalid_reviews = (df["number_of_reviews"] < 0).sum()

# Reviews per month tidak boleh negatif
invalid_reviews_per_month = (
    df["reviews_per_month"].dropna() < 0
).sum()

print(f"Harga negatif                    : {invalid_price:,}")
print(f"Minimum nights <= 0              : {invalid_minimum_nights:,}")
print(f"Latitude di luar 40-41           : {invalid_latitude:,}")
print(f"Longitude di luar -75 sampai -73 : {invalid_longitude:,}")
print(f"Availability di luar 0-365       : {invalid_availability:,}")
print(f"Jumlah review negatif             : {invalid_reviews:,}")
print(f"Review per bulan negatif          : {invalid_reviews_per_month:,}")

# ------------------------------------------------------------
# 2.4 Data Type
# ------------------------------------------------------------

print("\n2.4 DATA TYPE")
print("-" * 70)

print(df.dtypes)

# Mengecek apakah kolom tanggal bisa dikonversi
print("\nKonversi kolom 'last_review' menjadi datetime...")

df["last_review"] = pd.to_datetime(
    df["last_review"],
    errors="coerce"
)

print(f"Tipe data last_review setelah konversi: {df['last_review'].dtype}")


# ============================================================
# 3. EDA
# ============================================================

print("\n\n" + "=" * 70)
print("3. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)


# ------------------------------------------------------------
# 3.1 Statistik Deskriptif
# ------------------------------------------------------------

print("\n3.1 STATISTIK DESKRIPTIF")
print("-" * 70)

numeric_columns = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365"
]

descriptive_stats = df[numeric_columns].describe().T

descriptive_stats["median"] = df[numeric_columns].median()

print(descriptive_stats)

# Simpan statistik
descriptive_stats.to_csv(
    os.path.join(OUTPUT_DIR, "statistik_deskriptif.csv")
)


# ------------------------------------------------------------
# 3.2 Histogram Price
# ------------------------------------------------------------

print("\n3.2 MEMBUAT HISTOGRAM PRICE...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["price"],
    bins=50,
    edgecolor="black"
)

plt.title("Distribusi Harga Airbnb di NYC")
plt.xlabel("Harga per Malam (USD)")
plt.ylabel("Jumlah Listing")
plt.xlim(0, df["price"].quantile(0.99))

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_histogram_price.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.3 Histogram Minimum Nights
# ------------------------------------------------------------

print("3.3 MEMBUAT HISTOGRAM MINIMUM NIGHTS...")

plt.figure(figsize=(10, 6))

plt.hist(
    df["minimum_nights"],
    bins=50,
    edgecolor="black"
)

plt.title("Distribusi Minimum Nights")
plt.xlabel("Minimum Nights")
plt.ylabel("Jumlah Listing")

# Membatasi visualisasi agar nilai ekstrem tidak mendominasi
plt.xlim(0, df["minimum_nights"].quantile(0.99))

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_histogram_minimum_nights.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.4 Bar Chart Neighbourhood Group
# ------------------------------------------------------------

print("3.4 MEMBUAT BAR CHART NEIGHBOURHOOD GROUP...")

neighbourhood_counts = (
    df["neighbourhood_group"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nJumlah listing berdasarkan neighbourhood group:")
print(neighbourhood_counts)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=neighbourhood_counts.index,
    y=neighbourhood_counts.values
)

plt.title("Jumlah Listing Berdasarkan Neighbourhood Group")
plt.xlabel("Neighbourhood Group")
plt.ylabel("Jumlah Listing")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_bar_neighbourhood_group.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.5 Bar Chart Room Type
# ------------------------------------------------------------

print("\n3.5 MEMBUAT BAR CHART ROOM TYPE...")

room_counts = (
    df["room_type"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nJumlah listing berdasarkan room type:")
print(room_counts)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=room_counts.index,
    y=room_counts.values
)

plt.title("Jumlah Listing Berdasarkan Room Type")
plt.xlabel("Room Type")
plt.ylabel("Jumlah Listing")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_bar_room_type.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.6 Boxplot Price
# ------------------------------------------------------------

print("\n3.6 MEMBUAT BOXPLOT PRICE...")

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=df["price"]
)

plt.title("Boxplot Harga Airbnb")
plt.xlabel("Harga per Malam (USD)")

# Agar boxplot lebih mudah dibaca
plt.xlim(
    0,
    df["price"].quantile(0.99)
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_boxplot_price.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.7 Boxplot Minimum Nights
# ------------------------------------------------------------

print("3.7 MEMBUAT BOXPLOT MINIMUM NIGHTS...")

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=df["minimum_nights"]
)

plt.title("Boxplot Minimum Nights")
plt.xlabel("Minimum Nights")

plt.xlim(
    0,
    df["minimum_nights"].quantile(0.99)
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "06_boxplot_minimum_nights.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.8 Scatter Plot Price vs Number of Reviews
# ------------------------------------------------------------

print("\n3.8 MEMBUAT SCATTER PLOT PRICE VS NUMBER OF REVIEWS...")

plt.figure(figsize=(10, 6))

plt.scatter(
    df["number_of_reviews"],
    df["price"],
    alpha=0.3
)

plt.title("Hubungan Number of Reviews dengan Price")
plt.xlabel("Number of Reviews")
plt.ylabel("Price (USD)")

plt.ylim(
    0,
    df["price"].quantile(0.99)
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "07_scatter_price_reviews.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.9 Scatter Plot Price vs Minimum Nights
# ------------------------------------------------------------

print("3.9 MEMBUAT SCATTER PLOT PRICE VS MINIMUM NIGHTS...")

plt.figure(figsize=(10, 6))

plt.scatter(
    df["minimum_nights"],
    df["price"],
    alpha=0.3
)

plt.title("Hubungan Minimum Nights dengan Price")
plt.xlabel("Minimum Nights")
plt.ylabel("Price (USD)")

plt.xlim(
    0,
    df["minimum_nights"].quantile(0.99)
)

plt.ylim(
    0,
    df["price"].quantile(0.99)
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "08_scatter_price_minimum_nights.png"),
    dpi=300
)

plt.show()
plt.close()


# ------------------------------------------------------------
# 3.10 Correlation Heatmap
# ------------------------------------------------------------

print("\n3.10 MEMBUAT CORRELATION HEATMAP...")

correlation_matrix = df[numeric_columns].corr()

print("\nMatriks korelasi:")
print(correlation_matrix.round(3))

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap Variabel Numerik")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "09_correlation_heatmap.png"),
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================

print("\n\n" + "=" * 70)
print("4. ANOMALY DETECTION")
print("=" * 70)


# ------------------------------------------------------------
# 4.1 Outlier Harga dengan IQR
# ------------------------------------------------------------

print("\n4.1 OUTLIER HARGA")
print("-" * 70)

Q1_price = df["price"].quantile(0.25)
Q3_price = df["price"].quantile(0.75)

IQR_price = Q3_price - Q1_price

lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

price_outliers = df[
    (df["price"] < lower_price) |
    (df["price"] > upper_price)
]

print(f"Q1 harga       : ${Q1_price:,.2f}")
print(f"Q3 harga       : ${Q3_price:,.2f}")
print(f"IQR            : ${IQR_price:,.2f}")
print(f"Batas bawah    : ${lower_price:,.2f}")
print(f"Batas atas     : ${upper_price:,.2f}")

print(f"\nJumlah outlier harga : {len(price_outliers):,}")

print("\nContoh outlier harga:")
print(
    price_outliers[
        [
            "id",
            "name",
            "neighbourhood_group",
            "neighbourhood",
            "room_type",
            "price"
        ]
    ]
    .sort_values("price", ascending=False)
    .head(10)
)


# ------------------------------------------------------------
# 4.2 Minimum Nights Ekstrem
# ------------------------------------------------------------

print("\n4.2 MINIMUM NIGHTS EKSTREM")
print("-" * 70)

Q1_nights = df["minimum_nights"].quantile(0.25)
Q3_nights = df["minimum_nights"].quantile(0.75)

IQR_nights = Q3_nights - Q1_nights

lower_nights = Q1_nights - 1.5 * IQR_nights
upper_nights = Q3_nights + 1.5 * IQR_nights

minimum_nights_extreme = df[
    (df["minimum_nights"] < lower_nights) |
    (df["minimum_nights"] > upper_nights)
]

print(f"Q1 minimum nights       : {Q1_nights:.2f}")
print(f"Q3 minimum nights       : {Q3_nights:.2f}")
print(f"IQR                     : {IQR_nights:.2f}")
print(f"Batas bawah             : {lower_nights:.2f}")
print(f"Batas atas              : {upper_nights:.2f}")

print(
    f"\nJumlah minimum nights ekstrem : "
    f"{len(minimum_nights_extreme):,}"
)

print("\nContoh minimum nights ekstrem:")

print(
    minimum_nights_extreme[
        [
            "id",
            "name",
            "neighbourhood_group",
            "room_type",
            "minimum_nights",
            "price"
        ]
    ]
    .sort_values("minimum_nights", ascending=False)
    .head(10)
)


# ------------------------------------------------------------
# 4.3 Harga Tidak Wajar
# ------------------------------------------------------------

print("\n4.3 HARGA TIDAK WAJAR")
print("-" * 70)

zero_price = (df["price"] == 0).sum()
negative_price = (df["price"] < 0).sum()

# Harga sangat tinggi menggunakan threshold eksploratif
# 1000 USD digunakan sebagai penanda harga yang perlu diperiksa,
# bukan berarti seluruh data di atas angka ini pasti salah.
very_high_price = df[df["price"] > 1000]

print(f"Listing dengan harga = $0       : {zero_price:,}")
print(f"Listing dengan harga negatif    : {negative_price:,}")
print(f"Listing dengan harga > $1,000   : {len(very_high_price):,}")

if len(very_high_price) > 0:

    print("\nContoh listing dengan harga > $1,000:")

    print(
        very_high_price[
            [
                "id",
                "name",
                "neighbourhood_group",
                "neighbourhood",
                "room_type",
                "price"
            ]
        ]
        .sort_values("price", ascending=False)
        .head(10)
    )


# ------------------------------------------------------------
# 4.4 Missing Value
# ------------------------------------------------------------

print("\n4.4 MISSING VALUE")
print("-" * 70)

missing_columns = missing_table[
    missing_table["Missing"] > 0
]

if len(missing_columns) == 0:
    print("Tidak terdapat missing value.")
else:
    print("Kolom yang memiliki missing value:")
    print(missing_columns.to_string(index=False))


# ------------------------------------------------------------
# 4.5 Anomali Availability
# ------------------------------------------------------------

print("\n4.5 ANOMALI AVAILABILITY")
print("-" * 70)

availability_anomaly = df[
    (df["availability_365"] < 0) |
    (df["availability_365"] > 365)
]

print(
    f"Jumlah data dengan availability di luar 0-365: "
    f"{len(availability_anomaly):,}"
)


# ------------------------------------------------------------
# 4.6 Menampilkan 10 Harga Tertinggi
# ------------------------------------------------------------

print("\n4.6 10 HARGA TERTINGGI")
print("-" * 70)

top_price = df[
    [
        "id",
        "name",
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
        "price",
        "minimum_nights"
    ]
].sort_values(
    "price",
    ascending=False
).head(10)

print(top_price.to_string(index=False))


# ============================================================
# 5. RINGKASAN HASIL ANALISIS
# ============================================================

print("\n\n" + "=" * 70)
print("5. KESIMPULAN")
print("=" * 70)

# Statistik utama
median_price = df["price"].median()
mean_price = df["price"].mean()

median_minimum_nights = df["minimum_nights"].median()
mean_minimum_nights = df["minimum_nights"].mean()

most_common_room = df["room_type"].mode()[0]
most_common_group = df["neighbourhood_group"].mode()[0]

highest_corr_pair = None
highest_corr_value = 0

corr_abs = correlation_matrix.abs().copy()

# Hilangkan korelasi diagonal
for i in range(len(corr_abs)):
    corr_abs.iloc[i, i] = 0

max_position = np.unravel_index(
    corr_abs.to_numpy().argmax(),
    corr_abs.shape
)

col1 = corr_abs.index[max_position[0]]
col2 = corr_abs.columns[max_position[1]]

highest_corr_value = correlation_matrix.loc[col1, col2]

highest_corr_pair = f"{col1} dengan {col2}"

print("\n5.1 KUALITAS DATASET")
print("-" * 70)

print(f"Jumlah data              : {len(df):,}")
print(f"Jumlah variabel          : {len(df.columns)}")
print(f"Total missing value      : {total_missing:,}")
print(f"Total duplicate          : {duplicate_count:,}")
print(f"Harga negatif            : {invalid_price:,}")
print(f"Minimum nights tidak valid: {invalid_minimum_nights:,}")

print("\n5.2 POLA YANG DITEMUKAN")
print("-" * 70)

print(
    f"Neighbourhood group dengan jumlah listing terbanyak: "
    f"{most_common_group}"
)

print(
    f"Room type yang paling banyak ditemukan: "
    f"{most_common_room}"
)

print(
    f"Rata-rata harga per malam: "
    f"${mean_price:,.2f}"
)

print(
    f"Median harga per malam: "
    f"${median_price:,.2f}"
)

print(
    f"Rata-rata minimum nights: "
    f"{mean_minimum_nights:.2f}"
)

print(
    f"Median minimum nights: "
    f"{median_minimum_nights:.2f}"
)

print(
    f"Korelasi numerik terbesar secara absolut: "
    f"{highest_corr_pair} = {highest_corr_value:.3f}"
)

print("\n5.3 ANOMALI")
print("-" * 70)

print(
    f"Outlier harga berdasarkan metode IQR: "
    f"{len(price_outliers):,}"
)

print(
    f"Minimum nights ekstrem berdasarkan metode IQR: "
    f"{len(minimum_nights_extreme):,}"
)

print(
    f"Harga > $1,000: "
    f"{len(very_high_price):,}"
)

print(
    f"Availability di luar 0-365: "
    f"{len(availability_anomaly):,}"
)

print("\n5.4 REKOMENDASI")
print("-" * 70)

print("1. Missing value perlu ditangani sebelum digunakan untuk pemodelan.")
print("2. Outlier harga perlu diperiksa sebelum analisis statistik lebih lanjut.")
print("3. Minimum nights yang sangat tinggi perlu diperiksa karena dapat memengaruhi analisis.")
print("4. Harga ekstrem tidak langsung dianggap sebagai kesalahan karena bisa saja merupakan listing dengan karakteristik khusus.")
print("5. Kolom kategori seperti room_type dan neighbourhood_group dapat digunakan untuk analisis perbandingan.")
print("6. Korelasi antarvariabel numerik dapat digunakan sebagai dasar untuk analisis lanjutan.")


# ============================================================
# 6. MENYIMPAN LAPORAN RINGKAS
# ============================================================

print("\n\n" + "=" * 70)
print("MENYIMPAN LAPORAN")
print("=" * 70)

report_path = os.path.join(
    OUTPUT_DIR,
    "laporan_eda.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("LAPORAN EXPLORATORY DATA ANALYSIS\n")
    file.write("NYC AIRBNB OPEN DATA\n")
    file.write("=" * 70 + "\n\n")

    file.write("1. DATA UNDERSTANDING\n")
    file.write("-" * 70 + "\n")
    file.write(f"Jumlah baris : {len(df):,}\n")
    file.write(f"Jumlah kolom : {len(df.columns)}\n\n")

    file.write("Kolom dataset:\n")

    for kolom in df.columns:
        file.write(f"- {kolom}: {deskripsi_variabel.get(kolom, '-')}\n")

    file.write("\n\n2. DATA QUALITY\n")
    file.write("-" * 70 + "\n")
    file.write(f"Total missing value : {total_missing:,}\n")
    file.write(f"Total duplicate     : {duplicate_count:,}\n")
    file.write(f"Harga negatif       : {invalid_price:,}\n")
    file.write(f"Minimum nights <= 0 : {invalid_minimum_nights:,}\n")
    file.write(f"Latitude invalid     : {invalid_latitude:,}\n")
    file.write(f"Longitude invalid    : {invalid_longitude:,}\n")
    file.write(f"Availability invalid : {invalid_availability:,}\n")

    file.write("\n\n3. EDA\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Rata-rata harga : ${mean_price:,.2f}\n"
    )
    file.write(
        f"Median harga    : ${median_price:,.2f}\n"
    )
    file.write(
        f"Rata-rata minimum nights : "
        f"{mean_minimum_nights:.2f}\n"
    )
    file.write(
        f"Median minimum nights : "
        f"{median_minimum_nights:.2f}\n"
    )
    file.write(
        f"Room type terbanyak : {most_common_room}\n"
    )
    file.write(
        f"Neighbourhood group terbanyak : {most_common_group}\n"
    )
    file.write(
        f"Korelasi terbesar : "
        f"{highest_corr_pair} = {highest_corr_value:.3f}\n"
    )

    file.write("\n\n4. ANOMALY DETECTION\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Outlier harga : {len(price_outliers):,}\n"
    )
    file.write(
        f"Minimum nights ekstrem : "
        f"{len(minimum_nights_extreme):,}\n"
    )
    file.write(
        f"Harga > $1,000 : {len(very_high_price):,}\n"
    )
    file.write(
        f"Availability invalid : "
        f"{len(availability_anomaly):,}\n"
    )

    file.write("\n\n5. REKOMENDASI\n")
    file.write("-" * 70 + "\n")
    file.write(
        "1. Missing value perlu ditangani sebelum analisis lanjutan.\n"
    )
    file.write(
        "2. Outlier harga perlu diperiksa sebelum pemodelan.\n"
    )
    file.write(
        "3. Minimum nights ekstrem perlu diperiksa.\n"
    )
    file.write(
        "4. Harga ekstrem tidak otomatis berarti data salah.\n"
    )
    file.write(
        "5. Variabel kategori dapat digunakan untuk analisis perbandingan.\n"
    )
    file.write(
        "6. Korelasi dapat digunakan sebagai dasar analisis lanjutan.\n"
    )


# ============================================================
# SELESAI
# ============================================================

print(f"\nLaporan berhasil disimpan:")
print(f"  {report_path}")

print("\nSemua grafik berhasil disimpan di folder:")
print(f"  {OUTPUT_DIR}/")

print("\n" + "=" * 70)
print("EDA SELESAI")
print("=" * 70)