#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:31:25 2018

@author: sait
"""
#Çalışmamıza başlarken kullanacağımız kütüphaneleri çalışma alanına taşıyoruz
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Çalışmamızda kullanacağımız veri setimizi yüklüyoruz
diabet = datasets.load_diabetes()

#Bu çalışmada sadece bir özellik kullanacağız
diabetes_X = diabet.data[:, np.newaxis, 2]

#Veri Setimizdeki verileri iki parçaya Eğitim/Test(%80 /%20) verisi olarak ayırıyoruz
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#Çıktı verileriniz ikiye (Eğitim/Test) verisi olarak ayırıyoruz
diabetes_y_train = diabet.target[:-20]
diabetes_y_test = diabet.target[-20:]

#Bir lineer regresyon nesnesi oluşturuyoruz
regr = linear_model.LinearRegression()

#Eğitim veri setini kullanarak modelimizi eğitiyoruz
regr.fit(diabetes_X_train, diabetes_y_train)

#Şimdi de eğittiğimiz modelimizi test edelim
diabetes_y_pred = regr.predict(diabetes_X_test)

#Katsayıları Yazdıralım
print('Katsayılar: \n', regr.coef_)

#Ortalama karesel toplamayı da yazdıralım (MSE)
print("Ortalama Karesel Hata: %2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))

#Varyans puanını yazdıralım: 1 En iyi varyans değeridir.
print('Varyans Puanı: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

#Artık Çıktıları çizdirme zamanı geldi, şimdi de çıktılarımızı çizdirelim.
plt.scatter(diabetes_X_test, diabetes_y_test, color = 'yellow')
plt.plot(diabetes_X_test, diabetes_y_pred, color = 'blue', linewidth = 1)

plt.xticks(())
plt.yticks(())

plt.show()
