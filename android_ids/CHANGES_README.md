# Anpassung des Flower-Android-Beispiels an das IDS-Autoencoder-Modell

Original: https://github.com/flwrlabs/flower/tree/main/examples/android


Änderungen:
tflite_convertor/convert_to_tflite.py | Head-Modell: CNN -> Autoencoder

export_partitions.py schreibt die Daten der 10 Geraete aus load_device_dataset nach assets/data/ 

python test_tflite.py Testdatei

getestet mit Python 3.9 mit tensorflow-cpu==2.10.1

Ablauf:
1. PC-Test (ohne Android und ohne Server): python tflite_convertor/test_tflite.py
2. Optional, Modelle neu erzeugen: in tflite_convertor/ python convert_to_tflite.py
   (erzeugt tflite_model)
3. Optional, Daten neu erzeugen (IDS-Projekt, Python 3.11): export_partitions.py
   (Zielordner client/app/src/main/assets/data).
