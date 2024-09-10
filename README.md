# FLO-RFM
FLO Müşteri Segmentasyonu ve RFM Analizi

Bu proje, FLO’nun müşteri verilerini analiz ederek çeşitli müşteri segmentleri oluşturmayı ve bu segmentlere göre hedef müşteri listeleri hazırlamayı amaçlamaktadır. Veriler, FLO’nun online ve offline platformlarından toplanmıştır ve çeşitli analizler kullanılarak müşteri segmentleri oluşturulmuştur.

İçindekiler

	•	Veri Hazırlığı
	•	RFM Metriklerinin Hesaplanması
	•	RF ve RFM Skorlarının Hesaplanması
	•	RF Skorlarının Segment Olarak Tanımlanması
	•	Aksiyon Zamanı
	•	Fonksiyonlar
	•	Kullanım
	•	Gereksinimler

 Veri Hazırlığı

	1.	Veri Setini Okuma: flo_data_20k.csv dosyası okunur ve veri çerçevesi oluşturulur.
	2.	Yeni Değişkenler: Her müşterinin toplam alışveriş sayısı (total_order_count) ve toplam harcaması (total_spent) hesaplanır.
	3.	Tarih Değişkenleri: Tarih ifadeleri datetime formatına dönüştürülür.
	4.	Veri Analizi: Alışveriş kanallarındaki müşteri sayısı, toplam alınan ürün sayısı ve toplam harcamaların dağılımı analiz edilir.

RFM Metriklerinin Hesaplanması

	•	Recency: Müşterinin son alışverişinden itibaren geçen gün sayısı.
	•	Frequency: Müşterinin toplam alışveriş sayısı.
	•	Monetary: Müşterinin toplam harcaması.

RF ve RFM Skorlarının Hesaplanması

Recency, Frequency ve Monetary metrikleri, qcut fonksiyonu ile 1-5 arasında skorlara dönüştürülür. Bu skorlar, müşteri segmentlerini tanımlamak için kullanılır.

RF Skorlarının Segment Olarak Tanımlanması

RF skorları, belirli segmentlere dönüştürülür. Segmentler:

	•	hibernating
	•	at_risk
	•	can't_loose
	•	about_to_sleep
	•	need_attention
	•	loyal_customers
	•	promising
	•	new_customers
	•	potential_loyalist
	•	champions

Aksiyon Zamanı

	1.	Yeni Marka Hedef Müşterileri: Kadın kategorisinden alışveriş yapan ve toplam harcaması 250 TL üzeri olan sadık müşteriler CSV dosyasına kaydedilir.
	2.	İndirim Hedef Müşterileri: Erkek ve çocuk ürünlerinde indirim yapılacak müşteriler belirlenir ve uygun profildeki müşterilerin ID’leri CSV dosyasına kaydedilir.
Fonksiyonlar

	•	create_rfm1(dataframe, today_date, new_brand_file_path, discount_file_path): Veriyi işler, RFM analizini yapar ve hedef müşteri listelerini CSV dosyalarına kaydeder.

 kullanım.

 import pandas as pd
from datetime import datetime as dt
from your_module import create_rfm1

# Veri okuma
dataframe = pd.read_csv("/Users/erdinc/Desktop/flo_data_20k.csv")

# Fonksiyonu çağırma
create_rfm1(
    dataframe=dataframe,
    today_date=dt(2021, 6, 2),
    new_brand_file_path="/Users/erdinc/Desktop/yeni_marka_hedef_müşteri_id.csv",
    discount_file_path="/Users/erdinc/Desktop/indirim_hedef_müşteri_ids.csv"
)

2.	CSV Dosyalarının İncelenmesi:
	•	yeni_marka_hedef_müşteri_id.csv: Kadın kategorisindeki hedef müşterilerin ID’leri.
	•	indirim_hedef_müşteri_ids.csv: Erkek ve çocuk kategorisindeki indirim hedef müşterilerin ID’leri.

Gereksinimler

	•	Python 3.x
	•	pandas
	•	numpy
	•	scikit-learn
	•	matplotlib
	•	seaborn
Yazar

Erdinç Uzunlu
