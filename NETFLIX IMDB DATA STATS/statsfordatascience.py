import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import kurtosistest
from scipy.stats import ttest_1samp
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr


# CSV dosyasını oku
df = pd.read_excel("C:/Users/glsmn/OneDrive/Masaüstü/netflix.xlsx")

print('İlk 5 sütunu ekrana yazdır:')
print(df.head())
print()


print('İstatistikleri özetle:')
print(df.describe())
print()


print('Sütun ve veri hakkında bilgi:')
print(df.info())
print()


#Gereksiz sütunları çıkardıktan sonra
df = df.drop(columns=['index'])

print(df.head())
print()

print('Null değerleri yaz:')
print(df.isnull().sum())
print()

sayısal_sutun= df.select_dtypes(include=['float64','int64'])


print()
print('Sayısal Sütun medyan değerleri:')
medyan = sayısal_sutun.median()
print(medyan)


print()
print('Varyans:')
varyans = sayısal_sutun.var()
print(varyans)


print()
mod = sayısal_sutun.mode().iloc[0]
print("Sayısal Sütunların Modları:")
print(mod)
print()

def harmonik_ortalama(veri):
    if len(veri) == 0:
        return 0
    return len(veri) / sum(1 / x for x in veri)

harmonik_ortalama_sayisal = sayısal_sutun.apply(harmonik_ortalama, axis=0)

print("Sayısal Sutunların Harmonik Ortalaması:")

print(harmonik_ortalama_sayisal)
def geometric_mean(nums):
    log_sum = np.sum(np.log(nums))
    return np.exp(log_sum / len(nums))


geometric_means = sayısal_sutun.apply(geometric_mean, axis=0)
print()
print("Sayısal sütunların geometrik ortalaması:")
print(geometric_means)
print()



def weighted_mean(nums, weights):
    return sum(nums * weights) / sum(weights)

# Her bir sütun için ağırlıklı ortalama hesapla
weighted_means = {}
for column in sayısal_sutun:
    weights = df[column].notna().astype(int)  # NaN olmayan değerlere ağırlık olarak 1, NaN olanlara 0 veriyoruz
    weighted_means[column] = weighted_mean(df[column], weights)

print("Sayısal sütunların ağırlıklı ortalaması:")
for column, mean in weighted_means.items():
    print(f"{column}: {mean}")

print()

duplicated_rows = df[df.duplicated()]
if duplicated_rows.empty:
    print("Yinelenen satır yok")
else:
    print("Yinelenen satırlar:\n", duplicated_rows)
    
    print()


# 'type' sütununu gruplayıp sayılarını alın
type_counts = df['type'].value_counts()

# Pasta dilimi grafiğini çizin
colors = ['lightblue', 'orange']
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=colors)
plt.axis('equal')  # Daireyi dairesel yapmak için
plt.title('Dizi-Film Dağılımı')
plt.show()


top_rated = df.nlargest(10, 'imdb_score') 

plt.figure(figsize=(10, 8))
sns.barplot(x='imdb_score', y='title', data=top_rated, palette='icefire')
plt.title('Skorda İlk 10')
plt.show()


sns.countplot(x='age_certification', data=df)
plt.xlabel('yaş sınır')
plt.ylabel('Rakam')
plt.title('Yaş sınırı dağılımı')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(x='release_year', bins=20, kde=True, data=df, color='purple', edgecolor='black')
plt.xlabel('çıkış yılı')
plt.ylabel('Rakam')
plt.title('Dizi-Filmlerin Çıkış Yılı Histogram Grafiği')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(x='imdb_votes', bins=20, kde=True, data=df, color='green', edgecolor='black')
plt.xlabel('Imdb Oylar')
plt.ylabel('Rakam')
plt.title('Kullanılan Oy Sayısı')
plt.show()


plt.figure(figsize=(6,5))

for i in ["runtime"]:
    sns.boxplot(df[i], color='bisque')
    plt.title("Dizi/Film Süreleri İçin Kutu Grafiği")
    plt.show()


plt.figure(figsize=(8, 6))
stats.probplot(df["imdb_score"], dist="norm", plot=plt)
plt.title('Imdb Skorları için Q-Q Grafiği')
plt.xlabel('Teorik Normal Dağılım Değerleri')
plt.ylabel('Gözlemlenen Değerler')
plt.show()


print()
print("Shapiro-Wilk Testi- Imdb skorları 6-8 aralığında olanlar için")
print()
#Örneklem almak için gerekli kodu ekleme
def sample_from_excel(file_path, n, column_name):

    # Sadece sayısal verilere sahip olan sütunu seçme
    column = pd.to_numeric(df[column_name], errors='coerce')

#Sadece belirli aralıktaki sayısal verilere sahip olan sütunu seçme
    filtered_data = column[(column >= 6) & (column <= 8)]

#Dosyadan n adet örneklem alma
    sample = filtered_data.sample(n=min(n, len(filtered_data)))

    return sample

excel_file_path = "netflix.xlsx"
n = 20
column_name = "imdb_score"

#Örneklemi alma
sample_data = sample_from_excel(excel_file_path, n, column_name)

#Shapiro-Wilk testini uygulama
statistic, p_value = shapiro(sample_data)

#Sonuçları yazdır
print("Shapiro-Wilk Test İstatistiği:", statistic)
print("p-değeri:", p_value)

#p-değeri 0.05'ten küçükse, null hipotezini (verilerin normal dağılıma sahip olduğu) reddedebiliriz.
if p_value > 0.05:
    print("Veriler normal dağılıma sahiptir.")
else:
    print("Veriler normal dağılıma sahip değildir.")

#hipotez testi için 6-8 skora sahip dizi-filmlerin yarısından fazlası 2010 ve sonrası çıkmıştır
# ya da yarısından fazlası filmdir denilebilir


print()
print("Anderson testi çıkış yılları 2014-2021 için")
def sample_from_excel(file_path, n, column_name):

    # Sadece sayısal verilere sahip olan sütunu seçme
    column = pd.to_numeric(df[column_name], errors='coerce')

#Sadece belirli aralıktaki sayısal verilere sahip olan sütunu seçme
    filtered_data = column[(column >= 2014) & (column <= 2021)]

#Dosyadan n adet örneklem alma
    sample = filtered_data.sample(n=min(n, len(filtered_data)))

    return sample

excel_file_path = "netflix.xlsx"

column_name = "release_year"

#Örneklemi alma
sample_data = sample_from_excel(excel_file_path, n, column_name)

#Anderson-Darling testini uygulama
result = anderson(sample_data, dist='norm')

#İstatistik ve elemanları alıp listeye dönüştürme
statistic = result.statistic
critical_values = result.critical_values
significance_levels = result.significance_level

#Sonuçları yazdırma
print("Anderson-Darling Test İstatistiği:", statistic)

#Kritik değerleri ve karşılık gelen anlamlılık düzeylerini yazdırma
for i in range(len(critical_values)):
    if statistic < critical_values[i]:
        print(f"Veri normal dağılıma sahiptir ({significance_levels[i]*100}% anlamlılık düzeyinde)")
        break
    elif i == len(critical_values) - 1:
        print("Veri normal dağılıma sahip değildir.")


# z testi tek örneklem: 2014-2021 yılları arasında çıkan filmler ana kütle, buradan seçilen 60 tane random film/dizi örneklem
#örneklem ve anakütle ortalamaları aynı mı?
print()
print()
# Ana kütle olarak 2014-2021 yılları arasındaki yapımları filtreleme
filtered_df = df[(df['release_year'] >= 2014) & (df['release_year'] <= 2021)]

# 60 random örneklem seçimi
sample = filtered_df.sample(n=120, random_state=42)['release_year']

# Örneklem ortalaması ve standart sapması
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)

# Anakütle ortalaması ve standart sapması
population_mean = filtered_df['release_year'].mean()
population_std = filtered_df['release_year'].std(ddof=1)

# Z test istatistiği hesaplama
z_score = (sample_mean - population_mean) / (population_std / np.sqrt(60))

# P-değeri hesaplama
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# Sonuçları yazdırma
print("Z-Score:", z_score)
print("P-değeri:", p_value)

print("Ana Kütle Ortalaması:", population_mean)
print("Örneklem Ortalaması:", sample_mean)

# Anlamlılık düzeyi belirleme
alpha = 0.05


# Karar verme
if p_value < alpha:
    print("Null hipotezi reddedilir: Örneklem, anakütle ortalamasından farklıdır.")
else:
    print("Null hipotezi reddedilemez: Örneklem, anakütle ortalaması ile aynıdır.")



print()

print()
print()
imdb_scores = df['imdb_score']

#6 ile 8 arasındaki imdb skorlarını filtreler
imdb_scores_filtered = imdb_scores[(imdb_scores >= 6) & (imdb_scores <= 8)]

#Ortalama hesaplar
ortalama_imdb = imdb_scores_filtered.mean()

print("6-8 aralığındaki IMDb skorlarının ortalaması:", ortalama_imdb)

print()
print()


imdb_scores = df['imdb_score']

#6 ile 8 arasındaki imdb skorlarını filtreler
imdb_scores_filtered = imdb_scores[(imdb_scores >= 6) & (imdb_scores <= 8)]

#Standart sapma hesaplar
standart_sapma_imdb = imdb_scores_filtered.std()

print("6-8 aralığındaki IMDb skorlarının standart sapması:", standart_sapma_imdb)
print()
print()

imdb_scores = df['imdb_score']

#6 ile 8 arasındaki imdb skorlarını filtreler
imdb_scores_filtered = imdb_scores[(imdb_scores >= 6) & (imdb_scores <= 8)]

#Ortalama hesaplar
ortalama_imdb = imdb_scores_filtered.mean()

print("6-8 aralığındaki IMDb skorlarının ortalaması:", ortalama_imdb)

print()
print()


imdb_scores = df['imdb_score']

#6 ile 8 arasındaki imdb skorlarını filtreler
imdb_scores_filtered = imdb_scores[(imdb_scores >= 6) & (imdb_scores <= 8)]

#Standart sapma hesaplar
standart_sapma_imdb = imdb_scores_filtered.std()

print("6-8 aralığındaki IMDb skorlarının standart sapması:", standart_sapma_imdb)
print()
print()

#Ana kütle parametreleri
population_mean = 7.04
population_std = 0.57

#Ana kütle oluşturma
np.random.seed(41)  # Tekrarlanabilirlik için seed ayarı
population = np.random.normal(population_mean, population_std, 400)

#Örneklem çekimi
sample = np.random.choice(population, 10)

#Örneklem ortalaması
sample_mean = np.mean(sample)

#T testi yapma
t_stat, p_value = stats.ttest_1samp(sample, population_mean)

#Anlamlılık seviyesi
alpha = 0.05

#Hipotezi kontrol etme
if p_value < alpha:
    hypothesis_result = "H0 reddedildi: 6-8 aralığındaki imdb_score'ların ortalaması, alınan örneklemin ortalamasından daha küçüktür."
else:
    hypothesis_result = "H0 kabul edildi: 6-8 aralığındaki imdb_score'ların ortalaması, alınan örneklemin ortalamasından daha büyüktür."

#Sonuçları yazdırma
print("Anakütle Ortalaması:", population_mean)
print("Örneklem Ortalaması:", sample_mean)
print("T istatistiği:", t_stat)
print("p-değeri:", p_value)
print(hypothesis_result)



numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix_numeric = df[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_numeric, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Sayısal Sütunlar için korelasyon')
plt.show()



    


print()
print()

#IMDb oyları ve film süreleri sütunlarını seç
imdb_votes = df['imdb_votes']
runtime = df['runtime']

#Spearman korelasyon katsayısını hesapla
corr, p_value = spearmanr(imdb_votes, runtime)

print("Spearman Korelasyon Katsayısı:", corr)
print("P-değeri:", p_value)

print()
print()

#IMDb oyları ve film süreleri sütunlarını seç
imdb_votes = df['imdb_votes']
runtime = df['runtime']

#Spearman korelasyon katsayısını hesapla
corr, p_value = spearmanr(imdb_votes, runtime)

print("Spearman Korelasyon Katsayısı:", corr)
print("P-değeri:", p_value)

print()
print()


#IMDb oyları ve film süreleri sütunlarını seç
imdb_votes = df['imdb_votes']
runtime = df['runtime']

#Mann-Whitney U testini uygula
stat, p_value = mannwhitneyu(imdb_votes, runtime)

print("Mann-Whitney U Test İstatistiği:", stat)
print("P-değeri:", p_value) #0.00000000000000000000000042951577980675895




    
    
    
    
    

    
    
    


























