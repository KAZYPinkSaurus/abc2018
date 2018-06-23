import pandas as pd
import numpy as np
import sys
import os
from prepro import handle_lon_lat as hll
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 順にデータロードして前処理して保存

# rowdata path
in_path = './data/train/'
label_path = './data/label/'
#for saving transeformed path
tran_path = './data_add_theta_speed/'
#for saving sequence path
seq_path = './sequence/'
# saving name
seq_name = 'abc.txt'
# for saving figure 
fig_path = './fig/'


# 処理するファイル名リスト
file_list = os.listdir(in_path)
label = os.listdir(label_path)

pre_pro = hll.HandleLonLat()

# お試し
# df = pd.DataFrame(np.array([[38.56632,139.2922], [38.56763,139.293], [4, 5]]), columns=['longitude', 'latitude'])

# ラベル
label_df = pd.read_csv(label_path+label[0],names=['y'])

delta_theta = pd.DataFrame()
speed = pd.DataFrame()

representation = []


for (i,j) in zip(file_list,label_df['y']):
    df = pd.read_csv(in_path+i, names=['longitude', 'latitude','a','b','c','d','time','days'])
    
    # 角度(一点前からの)
    df_theta = pd.DataFrame(pre_pro.cal_direction(df),columns=['theta'])
    df = pd.concat([df,pd.DataFrame(df_theta.values,columns=['theta'])],axis=1)
    
    # Δ角度(一点前からの)
    df_theta_shift = df_theta.shift(1)
    df_delta_theta = pre_pro.cal_delta_direction(df_theta, df_theta_shift)

    df = pd.concat([df, pd.DataFrame(df_delta_theta.values, columns=['delta_theta'])], axis=1)
    # plot用
    delta_theta = pd.concat([delta_theta,df_delta_theta])
    # print(df_delta_theta)
    
    # 速度(m/s)
    df_speed = pre_pro.cal_speed(df, 5)
    df = pd.concat([df, pd.DataFrame(df_speed.values, columns=['speed'])], axis=1)
    # label付け
    df = pd.concat([pd.DataFrame(np.ones(df.shape[0])*int(j),columns=['y']),df], axis=1)

    # plot用
    speed = pd.concat([speed, df_speed])
    df.to_csv(tran_path + i, index=False)
    
    # スピードと向きからシンボル作成
    # speed は1m/s以下をslow(1)とそれ以上をFast(2)と定義
    # Δthetaは±45をFoward(1), -180~-45をLeft(2), 45~180をRight(3)と定義
    sequence = str(j)
    for (k, l) in zip(df['speed'],df['delta_theta']):
        if k < 1:
            # slow
            sequence += ' 1'
        else:
            # fast
            sequence += ' 2'

        if l < -45:
            # left
            sequence += ':2'
        elif l > 45:
            # right
            sequence += ':3'
        else:
            # forward
            sequence += ':1'
    
    representation.append(sequence)

f = open(seq_path + seq_name, 'w')  # 書き込みモードで開く
for m in representation:
    f.write(m+'\n') # 引数の文字列をファイルに書き込む
f.close() # ファイルを閉じる



# histgram
plt.hist(speed.fillna(0).values,bins=100)
plt.title("speed(m/s)")
plt.savefig(fig_path+'speed.png')

plt.hist(delta_theta.fillna(0).values,bins=100)
plt.title("delta_theta(dgree)")
plt.savefig(fig_path+'delta_theta.png')





# もう一つのdfに積む

# ラベル付け

# 保存










