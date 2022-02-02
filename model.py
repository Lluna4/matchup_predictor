import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
from tensorflow.python.ops.numpy_ops import np_config
import shutil
#shutil.make_archive("model", 'zip', "./my_model/")
np_config.enable_numpy_behavior()
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
n = 0
n2 = -1
n3 = -1
n4 = -1
n5 = -1
n6 = -1
n7 = -1
print("si")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
cols = ["CHAMPION", "WIN"]
new_cols = ['vs_champ', 'win2']
data = pd.read_csv("../input/league-of-legends-1v1-matchups-results/matchups.csv", usecols=[3, 7])
d2 = pd.read_csv("../input/league-of-legends-1v1-matchups-results/matchups.csv", usecols=[0])
d2 = d2[:50000]
data = data[:100000]
print(data)
d = dict()
for col, ncol in zip(cols, new_cols):
    d[col] = data[col].iloc[::2].values
    d[ncol] = data[col].iloc[1::2].values


    
print(d)

d = pd.DataFrame(d)

a2= {}
t = "Aatrox,Ahri,Akali,Akshan,Alistar,Amumu,Anivia,Annie,Aphelios,Ashe,AurelionSol,Azir,Bard,Blitzcrank,Brand,Braum,Caitlyn,Camille,Cassiopeia,Chogath,Corki,Darius,Diana,DrMundo,Draven,Ekko,Elise,Evelynn,Ezreal,FiddleSticks,Fiora,Fizz,Galio,Gangplank,Garen,Gnar,Gragas,Graves,Gwen,Hecarim,Heimerdinger,Illaoi,Irelia,Ivern,Janna,JarvanIV,Jax,Jayce,Jhin,Jinx,Kai'Sa,Kalista,Karma,Karthus,Kassadin,Katarina,Kayle,Kayn,Kennen,Khazix,Kindred,Kled,KogMaw,Leblanc,LeeSin,Leona,Lillia,Lissandra,Lucian,Lulu,Lux,Malphite,Malzahar,Maokai,MasterYi,MissFortune,Mordekaiser,Morgana,Nami,Nasus,Nautilus,Neeko,Nidalee,Nocturne,Nunu,Olaf,Orianna,Ornn,Pantheon,Poppy,Pyke,Qiyana,Quinn,Rakan,Rammus,RekSai,Rell,Renekton,Rengar,Riven,Rumble,Ryze,Samira,Sejuani,Senna,Seraphine,Sett,Shaco,Shen,Shyvana,Singed,Sion,Sivir,Skarner,Sona,Soraka,Swain,Sylas,Syndra,TahmKench,Taliyah,Talon,Taric,Teemo,Thresh,Tristana,Trundle,Tryndamere,TwistedFate,Twitch,Udyr,Urgot,Varus,Vayne,Veigar,Velkoz,Vex,Vi,Viego,Viktor,Vladimir,Volibear,Warwick,MonkeyKing,Xayah,Xerath,Xin Zhao,Yasuo,Yone,Yorick,Yuumi,Zac,Zed,Zeri,Ziggs,Zilean,Zoe,Zyra"
t = t.split(",")
an = {True: 1, False: 0}
add = {"bottom": 1, "utility": 2, "middle": 3, "jungle": 4, "top": 5}
for a in t:
    n += 1
    a2.update({a:n})
print(d)
for ia in d2["P_MATCH_ID"]:
    n6 += 1
    ia = ia[15: ]
    d2.loc[n6, "P_MATCH_ID"] = ia
print(d2)
    
    
    
for ii in d["CHAMPION"]:
    if ii == "Kaisa":
        ii = "Kai'Sa"
    if ii == "XinZhao":
        ii = "Xin Zhao"
    try:
        n2 += 1
        #print (ii)
        oo = a2[ii]
        d.loc[n2, "CHAMPION"] = oo
    except KeyError:
        pass
for ii in d["vs_champ"]:
    if ii == "Kaisa":
        ii = "Kai'Sa"
    if ii == "XinZhao":
        ii = "Xin Zhao"
    try:
        n3 += 1
        #print (ii)
        oo = a2[ii]
        d.loc[n3, "vs_champ"] = oo
    except KeyError:
        pass

for ii in d["win2"]:
    if ii == "Kaisa":
        ii = "Kai'Sa"
    if ii == "XinZhao":
        ii = "Xin Zhao"
    try:
        n4 += 1
        #print (ii)
        oo = an[ii]
        d.loc[n4, "win2"] = oo
    except KeyError:
        pass

for ii in d["WIN"]:
    if ii == "Kaisa":
        ii = "Kai'Sa"
    if ii == "XinZhao":
        ii = "Xin Zhao"
    try:
        n5 += 1
        #print (ii)
        oo = an[ii]
        d.loc[n5, "WIN"] = oo
    except KeyError:
        pass
for iii in d2["P_MATCH_ID"]:
    if iii == "_top":
        iii = "top"
    if iii == "_middle":
        iii = "middle"
    if iii == "_bottom":
        iii = "bottom"
    if iii ==  "_utility":
        iii = "utility"
    if iii == "_jungle":
        iii = "jungle"
    if iii == "op":
        iii = "top"
    if iii == "iddle":
        iii = "middle"
    if iii == "ottom":
        iii = "bottom"
    if iii ==  "tility":
        iii = "utility"
    if iii == "ungle":
        iii = "jungle"
    if iii == "p":
        iii = "top"
    if iii == "ddle":
        iii = "middle"
    if iii == "ttom":
        iii = "bottom"
    if iii ==  "ility":
        iii = "utility"
    if iii == "ngle":
        iii = "jungle"
    n7 += 1
    #print (ii)
    oo = int(add[iii])
    d.loc[n7, "P_MATCH_ID"] = oo

d2 = d2.iloc[::2]
d2 = pd.DataFrame(d2)
#d.merge(d2)
print(d)
target = d.pop("WIN")
target2 = d.pop("win2")
target = pd.DataFrame(target2)
target = np.asarray(target).reshape(50000,1).astype(int)
print(target)


d = pd.DataFrame(d)

d = np.asarray(d).reshape(50000,3).astype(int)
print(d)

 



#normalizer(d.iloc[:3])





def get_basic_model():
  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,3)),
    tf.keras.layers.Dense(24, activation='sigmoid'),
    tf.keras.layers.Dense(12, activation='sigmoid'),
   
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])
  return model

model = get_basic_model()
print(d[0])
model.fit(d, target, epochs=5, batch_size=BATCH_SIZE)
model.save('my_model')

