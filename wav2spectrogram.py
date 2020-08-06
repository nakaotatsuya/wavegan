import function
import numpy as np
from matplotlib import pyplot as plt
import os

#path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/drums/test/Cowbell_00010.wav")
path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "output_data/Cowbell/001.wav")

#path = 'ff_A.wav'                           #ファイルパスを指定
data, samplerate = function.wavload(path)   #wavファイルを読み込む
x = np.arange(0, len(data)) / samplerate    #波形生成のための時間軸の作成
 
# Fsとoverlapでスペクトログラムの分解能を調整する。
Fs = 4096                                   # フレームサイズ
overlap = 90                                # オーバーラップ率
 
# オーバーラップ抽出された時間波形配列
time_array, N_ave, final_time = function.ov(data, samplerate, Fs, overlap)
 
# ハニング窓関数をかける
time_array, acf = function.hanning(time_array, Fs, N_ave)
 
# FFTをかける
fft_array, fft_mean, fft_axis = function.fft_ave(time_array, samplerate, Fs, N_ave, acf)
 
# スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
fft_array = fft_array.T
 
# ここからグラフ描画
# グラフをオブジェクト指向で作成する。
fig = plt.figure()
ax1 = fig.add_subplot(111)
 
# データをプロットする。
im = ax1.imshow(fft_array, \
                vmin = 0, vmax = 60,
                extent = [0, final_time, 0, samplerate], \
                aspect = 'auto',\
                cmap = 'jet')
 
# カラーバーを設定する。
cbar = fig.colorbar(im)
cbar.set_label('SPL [dBA]')
 
# 軸設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')
 
# スケールの設定をする。
ax1.set_xticks(np.arange(0, 50, 5))
ax1.set_yticks(np.arange(0, 20000, 2000))
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 10000)
 
# グラフを表示する。
plt.show()
plt.close()
