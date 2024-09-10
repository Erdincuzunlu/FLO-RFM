from crypt import methods
from operator import index
##########################################################
# GÖREVLER
###############################################################
import pandas as pd
import numpy as np
from dateutil.utils import today
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from RMF.rfm import rfm_new
from pandas_alistirmalar import df_tips

pd.set_option('display.width', 500)

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
df = pd.read_csv("/Users/erdinc/Desktop/flo_data_20k.csv")
df

           # 2. Veri setinde
                     # a. İlk 10 gözlem,
df.head(10)
                     # b. Değişken isimleri,
df.info()
                     # c. Betimsel istatistik,
df.describe().T
                     # d. Boş değer,
df.isnull().sum()
                     # e. Değişken tipleri, incelemesi yapınız.
df.dtypes
                    # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["total_order_count"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
### müşteri toplam sayısı
df["total_spent"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
### Toplam harcama
df.head()
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes
dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
for column in dates:
    df[column] = df[column].apply(pd.to_datetime)
           # 5.  Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız
df.columns
df.head()
average_order_count = df.groupby('order_channel').agg({"master_id": "count",
                                                       "total_order_count": "sum",
                                                        "total_spent": "sum"})
           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df["total_spent"].sort_values(ascending=False).head(10)
           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df["total_order_count"].sort_values(ascending=False).head(10)
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def onhazırlık(df):
    df["total_order_count"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_spent"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for column in dates:
        df[column] = df[column].apply(pd.to_datetime)
        average_order_count = df.groupby('order_channel').agg({"master_id": "count",
                                                               "total_order_count": "sum",
                                                               "total_spent": "sum"})
    df["total_spent"].sort_values(ascending=False).head(10)
    df["total_order_count"].sort_values(ascending=False).head(10)
    return df




# GÖREV 2: RFM Metriklerinin Hesaplanması
today_date = dt.datetime(2021, 6, 2)
rfm = df.groupby("master_id").agg({"last_order_date": lambda order_date: (today_date-order_date).dt.days,
                             "total_order_count": lambda order: order.sum(),
                             "total_spent": lambda customer_value: customer_value.sum()})
rfm


# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
rfm.columns
rfm.columns = ["recency", "frequency", "monetary"]
rfm
rfm["recency_Score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2 ,1])
rfm["frequency_Score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4 ,5])
rfm["monetary_Score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4 ,5])
rfm["rfm_score"] = (rfm["recency_Score"].astype(str) +
                    rfm["frequency_Score"].astype(str))
rfm


# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

seg_map = {
    r'[1-2][1-2]': "hibernating",
    r'[1-2][3-4]':  "at_Risk",
    r'[1-2]5' :     "can't_loose",
    r'3[1-2]' :     "about_to_sleep",
    r'33':          "Need_attention",
    r'[3-4][4-5]':  "loyal_customers",
    r'41':          "promosing",
    r'51':          "new_customers",
    r'[4-5][2-3]':  "potantinal_loyalist",
    r'5[4-5]':      "champions"
}
rfm["segment"] = rfm["rfm_score"].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
rfm[rfm["segment"] == "Need_attention"].head()
rfm[rfm["segment"] == "Need_attention"].index
rfm

rfm

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})
rfm
           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

rfm_1 = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")]
rfm_1.head(100)
rfm["segment"].value_counts()
rfm_1
df
# Kadın kategorisinden alışveriş yapan müşterileri filtreleme
df_women = df[df["interested_in_categories_12"].str.contains("KADIN", na=False)]
# Toplam harcaması 250 TL'den fazla olanları filtreleme
target_customers = df_women[df_women["master_id"].isin(rfm_1.index)]
target_customers = target_customers[target_customers["total_spent"] > 250]
rfm
df
# Hedef müşteri ID'lerini CSV dosyasına kaydetme
target_customers[["master_id"]].to_csv("/Users/erdinc/Desktop/yeni_marka_hedef_müşteri_id.csv", index=False)
                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.
df_men_kids = df[df["interested_in_categories_12"].str.contains("ERKEK|COCUK", na=False)]
target_segments = [["hibernating", "about_to_sleep", "Need_attention"]]
rfm_target = rfm[rfm["segment"].isin(target_segments)]
target_customers = df_men_kids[df_men_kids["master_id"].isin(rfm_target.index)]
target_customers[["master_id"]].to_csv("/Users/erdinc/Desktop/indirim_hedef_müşteri_ids.csv", index=False)
# GÖREV 6: Tüm süreci fonksiyonlaştırınız.


def create_rfm1(dataframe, today_date, new_brand_file_path, discount_file_path):

    df = dataframe.copy()

    # Yeni değişkenler oluştur
    df["total_order_count"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_spent"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


    dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for column in dates:
        df[column] = pd.to_datetime(df[column])


    rfm = df.groupby("master_id").agg({
        "last_order_date": lambda order_date: (today_date - order_date).dt.days,
        "total_order_count": "sum",
        "total_spent": "sum"
    })
    rfm.columns = ["recency", "frequency", "monetary"]


    rfm["recency_Score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_Score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_Score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["rfm_score"] = (rfm["recency_Score"].astype(str) +
                        rfm["frequency_Score"].astype(str))


    seg_map = {
        r'[1-2][1-2]': "hibernating",
        r'[1-2][3-4]': "at_Risk",
        r'[1-2]5': "can't_loose",
        r'3[1-2]': "about_to_sleep",
        r'33': "Need_attention",
        r'[3-4][4-5]': "loyal_customers",
        r'41': "promosing",
        r'51': "new_customers",
        r'[4-5][2-3]': "potantinal_loyalist",
        r'5[4-5]': "champions"
    }
    rfm["segment"] = rfm["rfm_score"].replace(seg_map, regex=True)


    df_women = df[df["interested_in_categories_12"].str.contains("KADIN", na=False)]
    rfm_new_brand = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]
    target_customers_new_brand = df_women[df_women["master_id"].isin(rfm_new_brand.index)]
    target_customers_new_brand = target_customers_new_brand[target_customers_new_brand["total_spent"] > 250]
    target_customers_new_brand[["master_id"]].to_csv(new_brand_file_path, index=False)


    df_men_kids = df[df["interested_in_categories_12"].str.contains("ERKEK|COCUK", na=False)]
    target_segments = ["hibernating", "about_to_sleep", "Need_attention", "new_customers"]
    rfm_discount = rfm[rfm["segment"].isin(target_segments)]
    target_customers_discount = df_men_kids[df_men_kids["master_id"].isin(rfm_discount.index)]
    target_customers_discount[["master_id"]].to_csv(discount_file_path, index=False)

df2 = rfm.copy()

dataframe = pd.read_csv("/Users/erdinc/Desktop/flo_data_20k.csv")
create_rfm1(
    dataframe=dataframe,
    today_date=dt(2021, 6, 2),
    new_brand_file_path="/Users/erdinc/Desktop/yeni_marka_hedef_müşteri_id.csv",
    discount_file_path="/Users/erdinc/Desktop/indirim_hedef_müşteri_ids.csv"
)



###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################
df = pd.read_csv("/Users/erdinc/Desktop/flo_data_20k.csv")
df
# 2. Veri setinde
        # a. İlk 10 gözlem,
df.head(10)
df.tail(15)
        # b. Değişken isimleri,
df.info()
        # c. Boyut,
df.shape
        # d. Betimsel istatistik,
df.describe()
df.describe().T
        # e. Boş değer,
df.isnull().sum()
        # f. Değişken tipleri, incelemesi yapınız.
df.dtypes



# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["Total_Order"] = df["order_num_total_ever_online"] + df ["order_num_total_ever_offline"]
df["Total_Order"]

df["Total_Spent"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["Total_Spent"]
# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
from datetime import datetime
df.dtypes
dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
for column in dates:
    df[column] = pd.to_datetime(df[column])
df[dates] = df[dates].apply(pd.to_datetime)
df.info()
df.columns[df.columns.str.contains("date")]



# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız. 
df.groupby(["order_channel", "master_id"]).agg({"Total_Order": "sum",
                                 "Total_Spent": "sum"})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df["Total_Spent"].sort_values(ascending=False).head(10)
#df.sort_values("Total_Spent", ascending=False)[:10] 2nd solution


# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df[["Total_Order", "master_id"]].sort_values(by= "Total_Order", ascending=False).head(10)



# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def create_new(df):
    df["Total_Order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["Total_Spent"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for column in dates:
        df[column] = pd.to_datetime(df[column])

    return df

create_new(df)
df.info()
###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################
# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi
df[["last_order_date"]].max()
today_date = dt.datetime(2021, 6, 1)

# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe
rfm = pd.DataFrame()

rfm[["recency", "frequency", "monetary"]] = df.groupby("master_id").agg({"last_order_date": lambda x : (today_date - x).dt.days,
                             "Total_Order": lambda x : sum(x),
                             "Total_Spent": lambda x : sum(x)})



#### Second solution
#rfm = pd.DataFrame()

#rfm["master_id"] = df["master_id"]
#rfm["recency"] = (today_date- df["last_order_date"]).astype('timedelta64[ns]').dt.days
#rfm[["frequency","monetary"]] = df[["Total_Order", "Total_Spent"]]

###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_Score"] = pd.qcut(rfm["recency"], 5, labels = (5, 4, 3, 2, 1))
rfm["frequency_Score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels = (1, 2, 3, 4, 5))
rfm["monetary_Score"] = pd.qcut(rfm["monetary"], 5, labels = (1, 2, 3, 4, 5))


# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["RF_Score"] = rfm["recency_Score"].astype(str) + rfm["frequency_Score"].astype(str)

rfm
###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

seg_map = {
    r'[1-2][1-2]': "hibernating",
    r'[1-2][3-4]': "at_Risk",
    r'[1-2]5' : "can't_loose",
    r'3[1-2]' : "about_to_sleep",
    r'33': "need_attention",
    r'[3-4][4-5]': "loyal_customers",
    r'41': "promising",
    r'51': "new_customers",
    r'[4-5][2-3]': "potential_loyalist",
    r'5[4-5]': "champions"
}




# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme

rfm["segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

rfm.head(10)
###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.


# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.
### .isin filtrelemek için kullanıyorduk
target = rfm[rfm["segment"].isin(["loyal_customers", "champions"])]["master_id"]
df.info()
df.head()
target.tail(20)

rfm[rfm["segment"].str.contains("champions|loyal_customers")]["master_id"]


df_women_target = df_women[df_women["master_id"].isin(target)]

df_women.head()
rfm[target]

df_women_target["master_id"].to_csv("hedef_müşteri.csv", index=False)





# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

target_b = rfm[rfm["segment"].isin(["hibernating", "at risk", "new customers"])]["master_id"]


df_men_kids_target = df_men_kids[df_men_kids["master_id"].isin(target_b)]

df_men_kids_target["master_id"].to_csv("indirim_hedef_müsteri.csv", index=False)