
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("PROJECTS/Measurement Problems PROJECT/amazon_review.csv")
df.head(20)

df.sort_values("helpful", ascending=False).head(20)
###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

df["overall"].mean()
# 4.587589013224822

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################
df.sort_values("day_diff", ascending=False).head(20)
#max = 1064

df.sort_values("day_diff", ascending=True).head()
#min = 1

df.loc[df["day_diff"] <= 50, "overall"].mean() * 26/100 + \
df.loc[(df["day_diff"] > 50) & (df["day_diff"] <= 100), "overall"].mean() * 24/100 + \
df.loc[(df["day_diff"] > 100) & (df["day_diff"] <= 300), "overall"].mean() * 20/100 + \
df.loc[(df["day_diff"] > 300) & (df["day_diff"] <= 600), "overall"].mean() * 18/100 + \
df.loc[(df["day_diff"] > 600) & (df["day_diff"] <= 1200), "overall"].mean() * 12/100

#fonksiyon hali:
def time_based_weighted_average(dataframe, w1=26, w2=24, w3=20, w4=18, w5=12):
    return df.loc[df["day_diff"] <= 50, "overall"].mean() * 26/100 + \
           df.loc[(df["day_diff"] > 50) & (df["day_diff"] <= 100), "overall"].mean() * 24/100 + \
           df.loc[(df["day_diff"] > 100) & (df["day_diff"] <= 300), "overall"].mean() * 20/100 + \
           df.loc[(df["day_diff"] > 300) & (df["day_diff"] <= 600), "overall"].mean() * 18/100 + \
           df.loc[(df["day_diff"] > 600) & (df["day_diff"] <= 1200), "overall"].mean() * 12/100

time_based_weighted_average(df)

# tarih ağırlıklı puan ortalaması = 4.673992725919244
# normal ortalamaya nazaran daha yüksek çıktı.


###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################

###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df.sort_values("helpful_no", ascending=False).head()


###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

def score_pos_neg_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df.sort_values("score_pos_neg_diff", ascending=False).head(20)
##
def score_average_rating(helpful_yes, helpful_no, total_vote):
    if helpful_yes + helpful_no == 0:
        return 0
    else:
        return helpful_yes / total_vote

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"],
                                                                     x["total_vote"]), axis=1)
df.sort_values("score_average_rating", ascending=False).head(20)

##
def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)


















