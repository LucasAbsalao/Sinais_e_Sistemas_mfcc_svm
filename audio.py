import pyaudio
import keyboard
import time
import wave
import pickle
import numpy as np
import librosa
import sklearn as sk
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

translate = {0: 'down', 1: 'left', 2: 'no', 3: 'right', 4: 'up', 5: 'yes', 6: 'zero'}

def get_features(mfcc,mel_spect,y):
  featstack = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1), scipy.stats.skew(mfcc, axis = 1), np.max(mfcc, axis = 1), np.median(mfcc, axis = 1), np.min(mfcc, axis = 1)))
  feat2 = librosa.feature.zero_crossing_rate(y)[0]
  feat2stack = np.hstack((np.mean(feat2), np.std(feat2), scipy.stats.skew(feat2, bias=False), np.max(feat2), np.median(feat2), np.min(feat2)))
  feat3 = librosa.feature.spectral_rolloff(y=y)[0] #a frequência abaixo da qual se encontra um certo percentual da energia espectral cumulativa (por padrão, 85%)
  feat3stack = np.hstack((np.mean(feat3), np.std(feat3), scipy.stats.skew(feat3,bias=False), np.max(feat3), np.median(feat3), np.min(feat3)))
  feat4 = librosa.feature.spectral_centroid(y=y)[0]
  feat4stack = np.hstack((np.mean(feat4), np.std(feat4), scipy.stats.skew(feat4, bias=False), np.max(feat4), np.median(feat4), np.min(feat4)))
  feat5 = librosa.feature.spectral_contrast(y=y)[0]
  feat5stack = np.hstack((np.mean(feat5), np.std(feat5), scipy.stats.skew(feat5, bias=False), np.max(feat5), np.median(feat5), np.min(feat5)))
  feat6 = librosa.feature.spectral_bandwidth(y=y)[0]
  feat6stack = np.hstack((np.min(feat6), np.std(feat6), scipy.stats.skew(feat6, bias=False), np.max(feat6), np.median(feat6), np.min(feat6)))
  mfcc_data = pd.Series(np.hstack((featstack, feat2stack, feat3stack, feat4stack, feat5stack, feat6stack)))
  return mfcc_data

def values_predicted(cont):
    file_path = 'audios_gravados/**/*.wav'

    audiosList = glob.glob(file_path, recursive=True)  # carrega todos os arquivos .wav que estão na pasta audios
    if not audiosList:
        print("Nenhum arquivo de áudio encontrado. Verifique o caminho.")

    print(audiosList)
    all_mfccs = []  # lista de mfccs
    all_mfcc_datas = []

    for audio in audiosList:  # percorre todos os arquivos de audio
        y, sr = librosa.load(audio, sr=None)  # carrega o audio

        spec = np.abs(librosa.stft(y, hop_length=512))
        spec = librosa.amplitude_to_db(spec, ref=np.max)

        # calcula o mel spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        #calcula os coeficientes mfcc
        mfccs = librosa.feature.mfcc(S=mel_spect, sr=sr, n_mfcc=13)  # n_mfcc é a quantidade de coeficientesr
        all_mfccs.append(mfccs)
        all_mfcc_datas.append(get_features(mfccs, mel_spect, y))

    all_mfcc_datas = np.array(all_mfcc_datas)
    matrix = all_mfcc_datas

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if np.isnan(matrix[i][j]):
                print(i, j)
                matrix[i][j] = 0

    #aplicando pca
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix)
    with open('pca.pkl', 'rb') as pickle_file:
        pca = pickle.load(pickle_file)
    X_pca = pca.transform(X_scaled)

    svc = joblib.load('svm_treinado.pkl')
    Y_predicao = svc.predict(X_pca[:cont])

    return Y_predicao

chunks = 1024
formato = pyaudio.paInt16
canais = 2
freq = 44100
filename = "audio_gravado.wav"

comando = 3
cont=0

while comando!=0 and cont<10:
    #print(comando!=0)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=formato,
                     channels=canais,
                     rate=freq,
                     input=True,
                     frames_per_buffer=chunks)

    frames = []
    print("Aperte espaço para começar a gravar, fale a palavra e aperte espaço para parar (rapidamente)")

    keyboard.wait('space')
    print("gravando")
    time.sleep(0.2)

    while True:
        data = stream.read(chunks)
        frames.append(data)
        if keyboard.is_pressed('space'):
            print("gravação parou")
            time.sleep(0.2)
            break
    print("digite um valor diferente de 0 para continuar enviando dados")
    stream.stop_stream()
    stream.close()
    pa.terminate()

    filename = f'audios_gravados/audio_gravado{cont}.wav'
    cont+=1

    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(canais)
    wavefile.setsampwidth(pa.get_sample_size(formato))
    wavefile.setframerate(freq)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


    comando = int(input())

print("realizando predicoes")
idx_predicoes = values_predicted(cont)
print(idx_predicoes)
for i,idx in enumerate(idx_predicoes):
    print(i,': ',translate[idx])