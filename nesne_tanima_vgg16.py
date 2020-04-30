# -*- coding: utf-8 -*- 

"""
Python ile Derin Öğrenme Dersi sunumu için Begençh Hemrakuliyev ve Ali Tunacan Onar tarafından hazırlanmıştır.
Danışman: Doç. Dr. Ercan Buluş
"""

# Öncelikle Keras’a vgg16 modülünü ekliyoruz.
from keras.applications.vgg16 import VGG16

# Sonrasında görüntümüz üzerinde ön işlemlerimizi gerçekleştiriyoruz.
from keras.preprocessing import image

# tahminlerini basitçe yazdırmamız için gerekli fonksiyonları çağırıyoruz. 
from keras.applications.vgg16 import preprocess_input, decode_predictions

# Giriş verimiz üzerine matris işlemleri yapmak için NumPy modülünü de çağırıyoruz.
import numpy as np

# Daha öncesinde eğitilmiş olan modelin ağırlıklarını alıyoruz.
model = VGG16(weights ='imagenet')

#Görüntünün yolunu veriyoruz.
img_path = 'images/kitap.jpg'

# Görüntünün boyutlarını VGG16 modeline uygun hale getiriyoruz.
img = image.load_img(img_path, target_size=(224,224))
    
# Boyutları ayarlanan görüntüyü matris dizisine çeviriyoruz.
x = image.img_to_array(img)

# Matrise çevirdiğimiz görüntünün eksenlerini ayarlıyoruz.
x = np.expand_dims(x, axis = 0)

#Tahmine hazır hale getiriyoruz.
x = preprocess_input(x)

# Giriş örnekleri için çıktı tahminleri üretiyoruz.
preds = model.predict(x)

# En son görüntümüze en yakın 3 tahmini bize göstermesini istiyoruz.
print('Predicted:', decode_predictions(preds, top = 3)[0])
