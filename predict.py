import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt

dataset = pd.read_csv('data.csv')
# ternary
MoWS_features = pd.read_csv('MoWS.csv')
MoWSe_features = pd.read_csv('MoWSe.csv')
MoWTe_features = pd.read_csv('MoWTe.csv')
MoSSe_features = pd.read_csv('MoSSe.csv')
MoSTe_features = pd.read_csv('MoSTe.csv')
MoSeTe_features = pd.read_csv('MoSeTe.csv')
WSSe_features = pd.read_csv('WSSe.csv')
WSTe_features = pd.read_csv('WSTe.csv')
WSeTe_features = pd.read_csv('WSeTe.csv')
# quaternary
MoxWS2xSe_features = pd.read_csv('MoxWS2xSe.csv')
MoxWS2xTe_features = pd.read_csv('MoxWS2xTe.csv')
MoxWSe2xTe_features = pd.read_csv('MoxWSe2xTe.csv')
MoxWSSe2x_features = pd.read_csv('MoxWSSe2x.csv')
MoxWSTe2x_features = pd.read_csv('MoxWSTe2x.csv')
MoxWSeTe2x_features = pd.read_csv('MoxWSeTe2x.csv')
MoS2xSe2xTe_features = pd.read_csv('MoS2xSe2xTe.csv')
MoS2xSeTe2x_features = pd.read_csv('MoS2xSeTe2x.csv')
MoSSe2xTe2x_features = pd.read_csv('MoSSe2xTe2x.csv')
WS2xSe2xTe_features = pd.read_csv('WS2xSe2xTe.csv')
WS2xSeTe2x_features = pd.read_csv('WS2xSeTe2x.csv')
WSSe2xTe2x_features = pd.read_csv('WSSe2xTe2x.csv')
# quinary
MoWxS2xSe2xTe_features = pd.read_csv('MoWxS2xSe2xTe.csv')
MoWxS2xSeTe2x_features = pd.read_csv('MoWxS2xSeTe2x.csv')
MoWxSSe2xTe2x_features = pd.read_csv('MoWxSSe2xTe2x.csv')
MoxWS2xSe2xTe_features = pd.read_csv('MoxWS2xSe2xTe.csv')
MoxWS2xSeTe2x_features = pd.read_csv('MoxWS2xSeTe2x.csv')
MoxWSSe2xTe2x_features = pd.read_csv('MoxWSSe2xTe2x.csv')

dataset1 = dataset.sample(frac=0.2, random_state=1)
remain1 = dataset.drop(dataset1.index)

dataset2 = remain1.sample(frac=0.25, random_state=1)
remain2 = remain1.drop(dataset2.index)

dataset3 = remain2.sample(frac=0.333, random_state=1)
remain3 = remain2.drop(dataset3.index)

dataset4 = remain3.sample(frac=0.5, random_state=1)
dataset5 = remain3.drop(dataset4.index)

# 1
train_dataset1 = pd.concat([dataset1, dataset2, dataset3])
validat_dataset1 = dataset4
test_dataset1 = dataset5

train_features1 = train_dataset1.copy()
validat_features1 = validat_dataset1.copy()
test_features1 = test_dataset1.copy()

train_labels1 = train_features1.pop('bandgap')
validat_labels1 = validat_features1.pop('bandgap')
test_labels1 = test_features1.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_1 = MyModel()

model_1.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_1 = model_1.fit(train_features1, train_labels1,
                    validation_data=(validat_features1, validat_labels1), 
                    epochs=3000, verbose=0)

test_results_1 = model_1.evaluate(test_features1, test_labels1, verbose=0)

model_1.summary()

MoWS_predictions1 = model_1.predict(MoWS_features).flatten()
MoWSe_predictions1 = model_1.predict(MoWSe_features).flatten()
MoWTe_predictions1 = model_1.predict(MoWTe_features).flatten()
MoSSe_predictions1 = model_1.predict(MoSSe_features).flatten()
MoSTe_predictions1 = model_1.predict(MoSTe_features).flatten()
MoSeTe_predictions1 = model_1.predict(MoSeTe_features).flatten()
WSSe_predictions1 = model_1.predict(WSSe_features).flatten()
WSTe_predictions1 = model_1.predict(WSTe_features).flatten()
WSeTe_predictions1 = model_1.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions1 = model_1.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions1 = model_1.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions1 = model_1.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions1 = model_1.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions1 = model_1.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions1 = model_1.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions1 = model_1.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions1 = model_1.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions1 = model_1.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions1 = model_1.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions1 = model_1.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions1 = model_1.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions1 = model_1.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions1 = model_1.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions1 = model_1.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions1 = model_1.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions1 = model_1.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions1 = model_1.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p1 = {}
MoWSe_p1 = {}
MoWTe_p1 = {}
MoSSe_p1 = {}
MoSTe_p1 = {}
MoSeTe_p1 = {}
WSSe_p1 = {}
WSTe_p1 = {}
WSeTe_p1 = {}

MoxWS2xSe_p1 = {}
MoxWS2xTe_p1 = {}
MoxWSe2xTe_p1 = {}
MoxWSSe2x_p1 = {}
MoxWSTe2x_p1 = {}
MoxWSeTe2x_p1 = {}
MoS2xSe2xTe_p1 = {}
MoS2xSeTe2x_p1 = {}
MoSSe2xTe2x_p1 = {}
WS2xSe2xTe_p1 = {}
WS2xSeTe2x_p1 = {}
WSSe2xTe2x_p1 = {}

MoWxS2xSe2xTe_p1 = {}
MoWxS2xSeTe2x_p1 = {}
MoWxSSe2xTe2x_p1 = {}
MoxWS2xSe2xTe_p1 = {}
MoxWS2xSeTe2x_p1 = {}
MoxWSSe2xTe2x_p1 = {}

MoWS_p1['gap1'] = MoWS_predictions1
MoWSe_p1['gap1'] = MoWSe_predictions1
MoWTe_p1['gap1'] = MoWTe_predictions1
MoSSe_p1['gap1'] = MoSSe_predictions1
MoSTe_p1['gap1'] = MoSTe_predictions1
MoSeTe_p1['gap1'] = MoSeTe_predictions1
WSSe_p1['gap1'] = WSSe_predictions1
WSTe_p1['gap1'] = WSTe_predictions1
WSeTe_p1['gap1'] = WSeTe_predictions1

MoxWS2xSe_p1['gap1'] = MoxWS2xSe_predictions1
MoxWS2xTe_p1['gap1'] = MoxWS2xTe_predictions1
MoxWSe2xTe_p1['gap1'] = MoxWSe2xTe_predictions1
MoxWSSe2x_p1['gap1'] = MoxWSSe2x_predictions1
MoxWSTe2x_p1['gap1'] = MoxWSTe2x_predictions1
MoxWSeTe2x_p1['gap1'] = MoxWSeTe2x_predictions1
MoS2xSe2xTe_p1['gap1'] = MoS2xSe2xTe_predictions1
MoS2xSeTe2x_p1['gap1'] = MoS2xSeTe2x_predictions1
MoSSe2xTe2x_p1['gap1'] = MoSSe2xTe2x_predictions1
WS2xSe2xTe_p1['gap1'] = WS2xSe2xTe_predictions1
WS2xSeTe2x_p1['gap1'] = WS2xSeTe2x_predictions1
WSSe2xTe2x_p1['gap1'] = WSSe2xTe2x_predictions1

MoWxS2xSe2xTe_p1['gap1'] = MoWxS2xSe2xTe_predictions1
MoWxS2xSeTe2x_p1['gap1'] = MoWxS2xSeTe2x_predictions1
MoWxSSe2xTe2x_p1['gap1'] = MoWxSSe2xTe2x_predictions1
MoxWS2xSe2xTe_p1['gap1'] = MoxWS2xSe2xTe_predictions1
MoxWS2xSeTe2x_p1['gap1'] = MoxWS2xSeTe2x_predictions1
MoxWSSe2xTe2x_p1['gap1'] = MoxWSSe2xTe2x_predictions1

MoWS_pre1 = pd.DataFrame(MoWS_p1)
MoWSe_pre1 = pd.DataFrame(MoWSe_p1)
MoWTe_pre1 = pd.DataFrame(MoWTe_p1)
MoSSe_pre1 = pd.DataFrame(MoSSe_p1)
MoSTe_pre1 = pd.DataFrame(MoSTe_p1)
MoSeTe_pre1 = pd.DataFrame(MoSeTe_p1)
WSSe_pre1 = pd.DataFrame(WSSe_p1)
WSTe_pre1 = pd.DataFrame(WSTe_p1)
WSeTe_pre1 = pd.DataFrame(WSeTe_p1)

MoxWS2xSe_pre1 = pd.DataFrame(MoxWS2xSe_p1)
MoxWS2xTe_pre1 = pd.DataFrame(MoxWS2xTe_p1)
MoxWSe2xTe_pre1 = pd.DataFrame(MoxWSe2xTe_p1)
MoxWSSe2x_pre1 = pd.DataFrame(MoxWSSe2x_p1)
MoxWSTe2x_pre1 = pd.DataFrame(MoxWSTe2x_p1)
MoxWSeTe2x_pre1 = pd.DataFrame(MoxWSeTe2x_p1)
MoS2xSe2xTe_pre1 = pd.DataFrame(MoS2xSe2xTe_p1)
MoS2xSeTe2x_pre1 = pd.DataFrame(MoS2xSeTe2x_p1)
MoSSe2xTe2x_pre1 = pd.DataFrame(MoSSe2xTe2x_p1)
WS2xSe2xTe_pre1 = pd.DataFrame(WS2xSe2xTe_p1)
WS2xSeTe2x_pre1 = pd.DataFrame(WS2xSeTe2x_p1)
WSSe2xTe2x_pre1 = pd.DataFrame(WSSe2xTe2x_p1)

MoWxS2xSe2xTe_pre1 = pd.DataFrame(MoWxS2xSe2xTe_p1)
MoWxS2xSeTe2x_pre1 = pd.DataFrame(MoWxS2xSeTe2x_p1)
MoWxSSe2xTe2x_pre1 = pd.DataFrame(MoWxSSe2xTe2x_p1)
MoxWS2xSe2xTe_pre1 = pd.DataFrame(MoxWS2xSe2xTe_p1)
MoxWS2xSeTe2x_pre1 = pd.DataFrame(MoxWS2xSeTe2x_p1)
MoxWSSe2xTe2x_pre1 = pd.DataFrame(MoxWSSe2xTe2x_p1)

# 2
train_dataset2 = pd.concat([dataset1, dataset2, dataset3])
validat_dataset2 = dataset5
test_dataset2 = dataset4

train_features2 = train_dataset2.copy()
validat_features2 = validat_dataset2.copy()
test_features2 = test_dataset2.copy()

train_labels2 = train_features2.pop('bandgap')
validat_labels2 = validat_features2.pop('bandgap')
test_labels2 = test_features2.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_2 = MyModel()

model_2.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_2 = model_2.fit(train_features2, train_labels2,
                    validation_data=(validat_features2, validat_labels2), 
                    epochs=3000, verbose=0)

test_results_2 = model_2.evaluate(test_features2, test_labels2, verbose=0)

model_2.summary()

MoWS_predictions2 = model_2.predict(MoWS_features).flatten()
MoWSe_predictions2 = model_2.predict(MoWSe_features).flatten()
MoWTe_predictions2 = model_2.predict(MoWTe_features).flatten()
MoSSe_predictions2 = model_2.predict(MoSSe_features).flatten()
MoSTe_predictions2 = model_2.predict(MoSTe_features).flatten()
MoSeTe_predictions2 = model_2.predict(MoSeTe_features).flatten()
WSSe_predictions2 = model_2.predict(WSSe_features).flatten()
WSTe_predictions2 = model_2.predict(WSTe_features).flatten()
WSeTe_predictions2 = model_2.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions2 = model_2.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions2 = model_2.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions2 = model_2.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions2 = model_2.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions2 = model_2.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions2 = model_2.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions2 = model_2.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions2 = model_2.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions2 = model_2.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions2 = model_2.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions2 = model_2.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions2 = model_2.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions2 = model_2.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions2 = model_2.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions2 = model_2.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions2 = model_2.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions2 = model_2.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions2 = model_2.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p2 = {}
MoWSe_p2 = {}
MoWTe_p2 = {}
MoSSe_p2 = {}
MoSTe_p2 = {}
MoSeTe_p2 = {}
WSSe_p2 = {}
WSTe_p2 = {}
WSeTe_p2 = {}

MoxWS2xSe_p2 = {}
MoxWS2xTe_p2 = {}
MoxWSe2xTe_p2 = {}
MoxWSSe2x_p2 = {}
MoxWSTe2x_p2 = {}
MoxWSeTe2x_p2 = {}
MoS2xSe2xTe_p2 = {}
MoS2xSeTe2x_p2 = {}
MoSSe2xTe2x_p2 = {}
WS2xSe2xTe_p2 = {}
WS2xSeTe2x_p2 = {}
WSSe2xTe2x_p2 = {}

MoWxS2xSe2xTe_p2 = {}
MoWxS2xSeTe2x_p2 = {}
MoWxSSe2xTe2x_p2 = {}
MoxWS2xSe2xTe_p2 = {}
MoxWS2xSeTe2x_p2 = {}
MoxWSSe2xTe2x_p2 = {}

MoWS_p2['gap2'] = MoWS_predictions2
MoWSe_p2['gap2'] = MoWSe_predictions2
MoWTe_p2['gap2'] = MoWTe_predictions2
MoSSe_p2['gap2'] = MoSSe_predictions2
MoSTe_p2['gap2'] = MoSTe_predictions2
MoSeTe_p2['gap2'] = MoSeTe_predictions2
WSSe_p2['gap2'] = WSSe_predictions2
WSTe_p2['gap2'] = WSTe_predictions2
WSeTe_p2['gap2'] = WSeTe_predictions2

MoxWS2xSe_p2['gap2'] = MoxWS2xSe_predictions2
MoxWS2xTe_p2['gap2'] = MoxWS2xTe_predictions2
MoxWSe2xTe_p2['gap2'] = MoxWSe2xTe_predictions2
MoxWSSe2x_p2['gap2'] = MoxWSSe2x_predictions2
MoxWSTe2x_p2['gap2'] = MoxWSTe2x_predictions2
MoxWSeTe2x_p2['gap2'] = MoxWSeTe2x_predictions2
MoS2xSe2xTe_p2['gap2'] = MoS2xSe2xTe_predictions2
MoS2xSeTe2x_p2['gap2'] = MoS2xSeTe2x_predictions2
MoSSe2xTe2x_p2['gap2'] = MoSSe2xTe2x_predictions2
WS2xSe2xTe_p2['gap2'] = WS2xSe2xTe_predictions2
WS2xSeTe2x_p2['gap2'] = WS2xSeTe2x_predictions2
WSSe2xTe2x_p2['gap2'] = WSSe2xTe2x_predictions2

MoWxS2xSe2xTe_p2['gap2'] = MoWxS2xSe2xTe_predictions2
MoWxS2xSeTe2x_p2['gap2'] = MoWxS2xSeTe2x_predictions2
MoWxSSe2xTe2x_p2['gap2'] = MoWxSSe2xTe2x_predictions2
MoxWS2xSe2xTe_p2['gap2'] = MoxWS2xSe2xTe_predictions2
MoxWS2xSeTe2x_p2['gap2'] = MoxWS2xSeTe2x_predictions2
MoxWSSe2xTe2x_p2['gap2'] = MoxWSSe2xTe2x_predictions2

MoWS_pre2 = pd.DataFrame(MoWS_p2)
MoWSe_pre2 = pd.DataFrame(MoWSe_p2)
MoWTe_pre2 = pd.DataFrame(MoWTe_p2)
MoSSe_pre2 = pd.DataFrame(MoSSe_p2)
MoSTe_pre2 = pd.DataFrame(MoSTe_p2)
MoSeTe_pre2 = pd.DataFrame(MoSeTe_p2)
WSSe_pre2 = pd.DataFrame(WSSe_p2)
WSTe_pre2 = pd.DataFrame(WSTe_p2)
WSeTe_pre2 = pd.DataFrame(WSeTe_p2)

MoxWS2xSe_pre2 = pd.DataFrame(MoxWS2xSe_p2)
MoxWS2xTe_pre2 = pd.DataFrame(MoxWS2xTe_p2)
MoxWSe2xTe_pre2 = pd.DataFrame(MoxWSe2xTe_p2)
MoxWSSe2x_pre2 = pd.DataFrame(MoxWSSe2x_p2)
MoxWSTe2x_pre2 = pd.DataFrame(MoxWSTe2x_p2)
MoxWSeTe2x_pre2 = pd.DataFrame(MoxWSeTe2x_p2)
MoS2xSe2xTe_pre2 = pd.DataFrame(MoS2xSe2xTe_p2)
MoS2xSeTe2x_pre2 = pd.DataFrame(MoS2xSeTe2x_p2)
MoSSe2xTe2x_pre2 = pd.DataFrame(MoSSe2xTe2x_p2)
WS2xSe2xTe_pre2 = pd.DataFrame(WS2xSe2xTe_p2)
WS2xSeTe2x_pre2 = pd.DataFrame(WS2xSeTe2x_p2)
WSSe2xTe2x_pre2 = pd.DataFrame(WSSe2xTe2x_p2)

MoWxS2xSe2xTe_pre2 = pd.DataFrame(MoWxS2xSe2xTe_p2)
MoWxS2xSeTe2x_pre2 = pd.DataFrame(MoWxS2xSeTe2x_p2)
MoWxSSe2xTe2x_pre2 = pd.DataFrame(MoWxSSe2xTe2x_p2)
MoxWS2xSe2xTe_pre2 = pd.DataFrame(MoxWS2xSe2xTe_p2)
MoxWS2xSeTe2x_pre2 = pd.DataFrame(MoxWS2xSeTe2x_p2)
MoxWSSe2xTe2x_pre2 = pd.DataFrame(MoxWSSe2xTe2x_p2)

# 3
train_dataset3 = pd.concat([dataset1, dataset2, dataset4])
validat_dataset3 = dataset3
test_dataset3 = dataset5

train_features3 = train_dataset3.copy()
validat_features3 = validat_dataset3.copy()
test_features3 = test_dataset3.copy()

train_labels3 = train_features3.pop('bandgap')
validat_labels3 = validat_features3.pop('bandgap')
test_labels3 = test_features3.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_3 = MyModel()

model_3.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_3 = model_3.fit(train_features3, train_labels3,
                    validation_data=(validat_features3, validat_labels3), 
                    epochs=3000, verbose=0)

test_results_3 = model_3.evaluate(test_features3, test_labels3, verbose=0)

model_3.summary()

MoWS_predictions3 = model_3.predict(MoWS_features).flatten()
MoWSe_predictions3 = model_3.predict(MoWSe_features).flatten()
MoWTe_predictions3 = model_3.predict(MoWTe_features).flatten()
MoSSe_predictions3 = model_3.predict(MoSSe_features).flatten()
MoSTe_predictions3 = model_3.predict(MoSTe_features).flatten()
MoSeTe_predictions3 = model_3.predict(MoSeTe_features).flatten()
WSSe_predictions3 = model_3.predict(WSSe_features).flatten()
WSTe_predictions3 = model_3.predict(WSTe_features).flatten()
WSeTe_predictions3 = model_3.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions3 = model_3.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions3 = model_3.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions3 = model_3.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions3 = model_3.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions3 = model_3.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions3 = model_3.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions3 = model_3.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions3 = model_3.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions3 = model_3.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions3 = model_3.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions3 = model_3.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions3 = model_3.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions3 = model_3.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions3 = model_3.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions3 = model_3.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions3 = model_3.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions3 = model_3.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions3 = model_3.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p3 = {}
MoWSe_p3 = {}
MoWTe_p3 = {}
MoSSe_p3 = {}
MoSTe_p3 = {}
MoSeTe_p3 = {}
WSSe_p3 = {}
WSTe_p3 = {}
WSeTe_p3 = {}

MoxWS2xSe_p3 = {}
MoxWS2xTe_p3 = {}
MoxWSe2xTe_p3 = {}
MoxWSSe2x_p3 = {}
MoxWSTe2x_p3 = {}
MoxWSeTe2x_p3 = {}
MoS2xSe2xTe_p3 = {}
MoS2xSeTe2x_p3 = {}
MoSSe2xTe2x_p3 = {}
WS2xSe2xTe_p3 = {}
WS2xSeTe2x_p3 = {}
WSSe2xTe2x_p3 = {}

MoWxS2xSe2xTe_p3 = {}
MoWxS2xSeTe2x_p3 = {}
MoWxSSe2xTe2x_p3 = {}
MoxWS2xSe2xTe_p3 = {}
MoxWS2xSeTe2x_p3 = {}
MoxWSSe2xTe2x_p3 = {}

MoWS_p3['gap3'] = MoWS_predictions3
MoWSe_p3['gap3'] = MoWSe_predictions3
MoWTe_p3['gap3'] = MoWTe_predictions3
MoSSe_p3['gap3'] = MoSSe_predictions3
MoSTe_p3['gap3'] = MoSTe_predictions3
MoSeTe_p3['gap3'] = MoSeTe_predictions3
WSSe_p3['gap3'] = WSSe_predictions3
WSTe_p3['gap3'] = WSTe_predictions3
WSeTe_p3['gap3'] = WSeTe_predictions3

MoxWS2xSe_p3['gap3'] = MoxWS2xSe_predictions3
MoxWS2xTe_p3['gap3'] = MoxWS2xTe_predictions3
MoxWSe2xTe_p3['gap3'] = MoxWSe2xTe_predictions3
MoxWSSe2x_p3['gap3'] = MoxWSSe2x_predictions3
MoxWSTe2x_p3['gap3'] = MoxWSTe2x_predictions3
MoxWSeTe2x_p3['gap3'] = MoxWSeTe2x_predictions3
MoS2xSe2xTe_p3['gap3'] = MoS2xSe2xTe_predictions3
MoS2xSeTe2x_p3['gap3'] = MoS2xSeTe2x_predictions3
MoSSe2xTe2x_p3['gap3'] = MoSSe2xTe2x_predictions3
WS2xSe2xTe_p3['gap3'] = WS2xSe2xTe_predictions3
WS2xSeTe2x_p3['gap3'] = WS2xSeTe2x_predictions3
WSSe2xTe2x_p3['gap3'] = WSSe2xTe2x_predictions3

MoWxS2xSe2xTe_p3['gap3'] = MoWxS2xSe2xTe_predictions3
MoWxS2xSeTe2x_p3['gap3'] = MoWxS2xSeTe2x_predictions3
MoWxSSe2xTe2x_p3['gap3'] = MoWxSSe2xTe2x_predictions3
MoxWS2xSe2xTe_p3['gap3'] = MoxWS2xSe2xTe_predictions3
MoxWS2xSeTe2x_p3['gap3'] = MoxWS2xSeTe2x_predictions3
MoxWSSe2xTe2x_p3['gap3'] = MoxWSSe2xTe2x_predictions3

MoWS_pre3 = pd.DataFrame(MoWS_p3)
MoWSe_pre3 = pd.DataFrame(MoWSe_p3)
MoWTe_pre3 = pd.DataFrame(MoWTe_p3)
MoSSe_pre3 = pd.DataFrame(MoSSe_p3)
MoSTe_pre3 = pd.DataFrame(MoSTe_p3)
MoSeTe_pre3 = pd.DataFrame(MoSeTe_p3)
WSSe_pre3 = pd.DataFrame(WSSe_p3)
WSTe_pre3 = pd.DataFrame(WSTe_p3)
WSeTe_pre3 = pd.DataFrame(WSeTe_p3)

MoxWS2xSe_pre3 = pd.DataFrame(MoxWS2xSe_p3)
MoxWS2xTe_pre3 = pd.DataFrame(MoxWS2xTe_p3)
MoxWSe2xTe_pre3 = pd.DataFrame(MoxWSe2xTe_p3)
MoxWSSe2x_pre3 = pd.DataFrame(MoxWSSe2x_p3)
MoxWSTe2x_pre3 = pd.DataFrame(MoxWSTe2x_p3)
MoxWSeTe2x_pre3 = pd.DataFrame(MoxWSeTe2x_p3)
MoS2xSe2xTe_pre3 = pd.DataFrame(MoS2xSe2xTe_p3)
MoS2xSeTe2x_pre3 = pd.DataFrame(MoS2xSeTe2x_p3)
MoSSe2xTe2x_pre3 = pd.DataFrame(MoSSe2xTe2x_p3)
WS2xSe2xTe_pre3 = pd.DataFrame(WS2xSe2xTe_p3)
WS2xSeTe2x_pre3 = pd.DataFrame(WS2xSeTe2x_p3)
WSSe2xTe2x_pre3 = pd.DataFrame(WSSe2xTe2x_p3)

MoWxS2xSe2xTe_pre3 = pd.DataFrame(MoWxS2xSe2xTe_p3)
MoWxS2xSeTe2x_pre3 = pd.DataFrame(MoWxS2xSeTe2x_p3)
MoWxSSe2xTe2x_pre3 = pd.DataFrame(MoWxSSe2xTe2x_p3)
MoxWS2xSe2xTe_pre3 = pd.DataFrame(MoxWS2xSe2xTe_p3)
MoxWS2xSeTe2x_pre3 = pd.DataFrame(MoxWS2xSeTe2x_p3)
MoxWSSe2xTe2x_pre3 = pd.DataFrame(MoxWSSe2xTe2x_p3)

# 4
train_dataset4 = pd.concat([dataset1, dataset2, dataset4])
validat_dataset4 = dataset5
test_dataset4 = dataset3

train_features4 = train_dataset4.copy()
validat_features4 = validat_dataset4.copy()
test_features4 = test_dataset4.copy()

train_labels4 = train_features4.pop('bandgap')
validat_labels4 = validat_features4.pop('bandgap')
test_labels4 = test_features4.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_4 = MyModel()

model_4.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_4 = model_4.fit(train_features4, train_labels4,
                    validation_data=(validat_features4, validat_labels4), 
                    epochs=3000, verbose=0)

test_results_4 = model_4.evaluate(test_features4, test_labels4, verbose=0)

model_4.summary()

MoWS_predictions4 = model_4.predict(MoWS_features).flatten()
MoWSe_predictions4 = model_4.predict(MoWSe_features).flatten()
MoWTe_predictions4 = model_4.predict(MoWTe_features).flatten()
MoSSe_predictions4 = model_4.predict(MoSSe_features).flatten()
MoSTe_predictions4 = model_4.predict(MoSTe_features).flatten()
MoSeTe_predictions4 = model_4.predict(MoSeTe_features).flatten()
WSSe_predictions4 = model_4.predict(WSSe_features).flatten()
WSTe_predictions4 = model_4.predict(WSTe_features).flatten()
WSeTe_predictions4 = model_4.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions4 = model_4.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions4 = model_4.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions4 = model_4.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions4 = model_4.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions4 = model_4.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions4 = model_4.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions4 = model_4.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions4 = model_4.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions4 = model_4.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions4 = model_4.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions4 = model_4.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions4 = model_4.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions4 = model_4.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions4 = model_4.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions4 = model_4.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions4 = model_4.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions4 = model_4.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions4 = model_4.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p4 = {}
MoWSe_p4 = {}
MoWTe_p4 = {}
MoSSe_p4 = {}
MoSTe_p4 = {}
MoSeTe_p4 = {}
WSSe_p4 = {}
WSTe_p4 = {}
WSeTe_p4 = {}

MoxWS2xSe_p4 = {}
MoxWS2xTe_p4 = {}
MoxWSe2xTe_p4 = {}
MoxWSSe2x_p4 = {}
MoxWSTe2x_p4 = {}
MoxWSeTe2x_p4 = {}
MoS2xSe2xTe_p4 = {}
MoS2xSeTe2x_p4 = {}
MoSSe2xTe2x_p4 = {}
WS2xSe2xTe_p4 = {}
WS2xSeTe2x_p4 = {}
WSSe2xTe2x_p4 = {}

MoWxS2xSe2xTe_p4 = {}
MoWxS2xSeTe2x_p4 = {}
MoWxSSe2xTe2x_p4 = {}
MoxWS2xSe2xTe_p4 = {}
MoxWS2xSeTe2x_p4 = {}
MoxWSSe2xTe2x_p4 = {}

MoWS_p4['gap4'] = MoWS_predictions4
MoWSe_p4['gap4'] = MoWSe_predictions4
MoWTe_p4['gap4'] = MoWTe_predictions4
MoSSe_p4['gap4'] = MoSSe_predictions4
MoSTe_p4['gap4'] = MoSTe_predictions4
MoSeTe_p4['gap4'] = MoSeTe_predictions4
WSSe_p4['gap4'] = WSSe_predictions4
WSTe_p4['gap4'] = WSTe_predictions4
WSeTe_p4['gap4'] = WSeTe_predictions4

MoxWS2xSe_p4['gap4'] = MoxWS2xSe_predictions4
MoxWS2xTe_p4['gap4'] = MoxWS2xTe_predictions4
MoxWSe2xTe_p4['gap4'] = MoxWSe2xTe_predictions4
MoxWSSe2x_p4['gap4'] = MoxWSSe2x_predictions4
MoxWSTe2x_p4['gap4'] = MoxWSTe2x_predictions4
MoxWSeTe2x_p4['gap4'] = MoxWSeTe2x_predictions4
MoS2xSe2xTe_p4['gap4'] = MoS2xSe2xTe_predictions4
MoS2xSeTe2x_p4['gap4'] = MoS2xSeTe2x_predictions4
MoSSe2xTe2x_p4['gap4'] = MoSSe2xTe2x_predictions4
WS2xSe2xTe_p4['gap4'] = WS2xSe2xTe_predictions4
WS2xSeTe2x_p4['gap4'] = WS2xSeTe2x_predictions4
WSSe2xTe2x_p4['gap4'] = WSSe2xTe2x_predictions4

MoWxS2xSe2xTe_p4['gap4'] = MoWxS2xSe2xTe_predictions4
MoWxS2xSeTe2x_p4['gap4'] = MoWxS2xSeTe2x_predictions4
MoWxSSe2xTe2x_p4['gap4'] = MoWxSSe2xTe2x_predictions4
MoxWS2xSe2xTe_p4['gap4'] = MoxWS2xSe2xTe_predictions4
MoxWS2xSeTe2x_p4['gap4'] = MoxWS2xSeTe2x_predictions4
MoxWSSe2xTe2x_p4['gap4'] = MoxWSSe2xTe2x_predictions4

MoWS_pre4 = pd.DataFrame(MoWS_p4)
MoWSe_pre4 = pd.DataFrame(MoWSe_p4)
MoWTe_pre4 = pd.DataFrame(MoWTe_p4)
MoSSe_pre4 = pd.DataFrame(MoSSe_p4)
MoSTe_pre4 = pd.DataFrame(MoSTe_p4)
MoSeTe_pre4 = pd.DataFrame(MoSeTe_p4)
WSSe_pre4 = pd.DataFrame(WSSe_p4)
WSTe_pre4 = pd.DataFrame(WSTe_p4)
WSeTe_pre4 = pd.DataFrame(WSeTe_p4)

MoxWS2xSe_pre4 = pd.DataFrame(MoxWS2xSe_p4)
MoxWS2xTe_pre4 = pd.DataFrame(MoxWS2xTe_p4)
MoxWSe2xTe_pre4 = pd.DataFrame(MoxWSe2xTe_p4)
MoxWSSe2x_pre4 = pd.DataFrame(MoxWSSe2x_p4)
MoxWSTe2x_pre4 = pd.DataFrame(MoxWSTe2x_p4)
MoxWSeTe2x_pre4 = pd.DataFrame(MoxWSeTe2x_p4)
MoS2xSe2xTe_pre4 = pd.DataFrame(MoS2xSe2xTe_p4)
MoS2xSeTe2x_pre4 = pd.DataFrame(MoS2xSeTe2x_p4)
MoSSe2xTe2x_pre4 = pd.DataFrame(MoSSe2xTe2x_p4)
WS2xSe2xTe_pre4 = pd.DataFrame(WS2xSe2xTe_p4)
WS2xSeTe2x_pre4 = pd.DataFrame(WS2xSeTe2x_p4)
WSSe2xTe2x_pre4 = pd.DataFrame(WSSe2xTe2x_p4)

MoWxS2xSe2xTe_pre4 = pd.DataFrame(MoWxS2xSe2xTe_p4)
MoWxS2xSeTe2x_pre4 = pd.DataFrame(MoWxS2xSeTe2x_p4)
MoWxSSe2xTe2x_pre4 = pd.DataFrame(MoWxSSe2xTe2x_p4)
MoxWS2xSe2xTe_pre4 = pd.DataFrame(MoxWS2xSe2xTe_p4)
MoxWS2xSeTe2x_pre4 = pd.DataFrame(MoxWS2xSeTe2x_p4)
MoxWSSe2xTe2x_pre4 = pd.DataFrame(MoxWSSe2xTe2x_p4)

# 5
train_dataset5 = pd.concat([dataset1, dataset2, dataset5])
validat_dataset5 = dataset3
test_dataset5 = dataset4

train_features5 = train_dataset5.copy()
validat_features5 = validat_dataset5.copy()
test_features5 = test_dataset5.copy()

train_labels5 = train_features5.pop('bandgap')
validat_labels5 = validat_features5.pop('bandgap')
test_labels5 = test_features5.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_5 = MyModel()

model_5.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_5 = model_5.fit(train_features5, train_labels5,
                    validation_data=(validat_features5, validat_labels5), 
                    epochs=3000, verbose=0)

test_results_5 = model_5.evaluate(test_features5, test_labels5, verbose=0)

model_5.summary()

MoWS_predictions5 = model_5.predict(MoWS_features).flatten()
MoWSe_predictions5 = model_5.predict(MoWSe_features).flatten()
MoWTe_predictions5 = model_5.predict(MoWTe_features).flatten()
MoSSe_predictions5 = model_5.predict(MoSSe_features).flatten()
MoSTe_predictions5 = model_5.predict(MoSTe_features).flatten()
MoSeTe_predictions5 = model_5.predict(MoSeTe_features).flatten()
WSSe_predictions5 = model_5.predict(WSSe_features).flatten()
WSTe_predictions5 = model_5.predict(WSTe_features).flatten()
WSeTe_predictions5 = model_5.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions5 = model_5.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions5 = model_5.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions5 = model_5.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions5 = model_5.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions5 = model_5.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions5 = model_5.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions5 = model_5.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions5 = model_5.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions5 = model_5.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions5 = model_5.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions5 = model_5.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions5 = model_5.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions5 = model_5.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions5 = model_5.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions5 = model_5.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions5 = model_5.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions5 = model_5.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions5 = model_5.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p5 = {}
MoWSe_p5 = {}
MoWTe_p5 = {}
MoSSe_p5 = {}
MoSTe_p5 = {}
MoSeTe_p5 = {}
WSSe_p5 = {}
WSTe_p5 = {}
WSeTe_p5 = {}

MoxWS2xSe_p5 = {}
MoxWS2xTe_p5 = {}
MoxWSe2xTe_p5 = {}
MoxWSSe2x_p5 = {}
MoxWSTe2x_p5 = {}
MoxWSeTe2x_p5 = {}
MoS2xSe2xTe_p5 = {}
MoS2xSeTe2x_p5 = {}
MoSSe2xTe2x_p5 = {}
WS2xSe2xTe_p5 = {}
WS2xSeTe2x_p5 = {}
WSSe2xTe2x_p5 = {}

MoWxS2xSe2xTe_p5 = {}
MoWxS2xSeTe2x_p5 = {}
MoWxSSe2xTe2x_p5 = {}
MoxWS2xSe2xTe_p5 = {}
MoxWS2xSeTe2x_p5 = {}
MoxWSSe2xTe2x_p5 = {}

MoWS_p5['gap5'] = MoWS_predictions5
MoWSe_p5['gap5'] = MoWSe_predictions5
MoWTe_p5['gap5'] = MoWTe_predictions5
MoSSe_p5['gap5'] = MoSSe_predictions5
MoSTe_p5['gap5'] = MoSTe_predictions5
MoSeTe_p5['gap5'] = MoSeTe_predictions5
WSSe_p5['gap5'] = WSSe_predictions5
WSTe_p5['gap5'] = WSTe_predictions5
WSeTe_p5['gap5'] = WSeTe_predictions5

MoxWS2xSe_p5['gap5'] = MoxWS2xSe_predictions5
MoxWS2xTe_p5['gap5'] = MoxWS2xTe_predictions5
MoxWSe2xTe_p5['gap5'] = MoxWSe2xTe_predictions5
MoxWSSe2x_p5['gap5'] = MoxWSSe2x_predictions5
MoxWSTe2x_p5['gap5'] = MoxWSTe2x_predictions5
MoxWSeTe2x_p5['gap5'] = MoxWSeTe2x_predictions5
MoS2xSe2xTe_p5['gap5'] = MoS2xSe2xTe_predictions5
MoS2xSeTe2x_p5['gap5'] = MoS2xSeTe2x_predictions5
MoSSe2xTe2x_p5['gap5'] = MoSSe2xTe2x_predictions5
WS2xSe2xTe_p5['gap5'] = WS2xSe2xTe_predictions5
WS2xSeTe2x_p5['gap5'] = WS2xSeTe2x_predictions5
WSSe2xTe2x_p5['gap5'] = WSSe2xTe2x_predictions5

MoWxS2xSe2xTe_p5['gap5'] = MoWxS2xSe2xTe_predictions5
MoWxS2xSeTe2x_p5['gap5'] = MoWxS2xSeTe2x_predictions5
MoWxSSe2xTe2x_p5['gap5'] = MoWxSSe2xTe2x_predictions5
MoxWS2xSe2xTe_p5['gap5'] = MoxWS2xSe2xTe_predictions5
MoxWS2xSeTe2x_p5['gap5'] = MoxWS2xSeTe2x_predictions5
MoxWSSe2xTe2x_p5['gap5'] = MoxWSSe2xTe2x_predictions5

MoWS_pre5 = pd.DataFrame(MoWS_p5)
MoWSe_pre5 = pd.DataFrame(MoWSe_p5)
MoWTe_pre5 = pd.DataFrame(MoWTe_p5)
MoSSe_pre5 = pd.DataFrame(MoSSe_p5)
MoSTe_pre5 = pd.DataFrame(MoSTe_p5)
MoSeTe_pre5 = pd.DataFrame(MoSeTe_p5)
WSSe_pre5 = pd.DataFrame(WSSe_p5)
WSTe_pre5 = pd.DataFrame(WSTe_p5)
WSeTe_pre5 = pd.DataFrame(WSeTe_p5)

MoxWS2xSe_pre5 = pd.DataFrame(MoxWS2xSe_p5)
MoxWS2xTe_pre5 = pd.DataFrame(MoxWS2xTe_p5)
MoxWSe2xTe_pre5 = pd.DataFrame(MoxWSe2xTe_p5)
MoxWSSe2x_pre5 = pd.DataFrame(MoxWSSe2x_p5)
MoxWSTe2x_pre5 = pd.DataFrame(MoxWSTe2x_p5)
MoxWSeTe2x_pre5 = pd.DataFrame(MoxWSeTe2x_p5)
MoS2xSe2xTe_pre5 = pd.DataFrame(MoS2xSe2xTe_p5)
MoS2xSeTe2x_pre5 = pd.DataFrame(MoS2xSeTe2x_p5)
MoSSe2xTe2x_pre5 = pd.DataFrame(MoSSe2xTe2x_p5)
WS2xSe2xTe_pre5 = pd.DataFrame(WS2xSe2xTe_p5)
WS2xSeTe2x_pre5 = pd.DataFrame(WS2xSeTe2x_p5)
WSSe2xTe2x_pre5 = pd.DataFrame(WSSe2xTe2x_p5)

MoWxS2xSe2xTe_pre5 = pd.DataFrame(MoWxS2xSe2xTe_p5)
MoWxS2xSeTe2x_pre5 = pd.DataFrame(MoWxS2xSeTe2x_p5)
MoWxSSe2xTe2x_pre5 = pd.DataFrame(MoWxSSe2xTe2x_p5)
MoxWS2xSe2xTe_pre5 = pd.DataFrame(MoxWS2xSe2xTe_p5)
MoxWS2xSeTe2x_pre5 = pd.DataFrame(MoxWS2xSeTe2x_p5)
MoxWSSe2xTe2x_pre5 = pd.DataFrame(MoxWSSe2xTe2x_p5)

# 6
train_dataset6 = pd.concat([dataset1, dataset2, dataset5])
validat_dataset6 = dataset4
test_dataset6 = dataset3

train_features6 = train_dataset6.copy()
validat_features6 = validat_dataset6.copy()
test_features6 = test_dataset6.copy()

train_labels6 = train_features6.pop('bandgap')
validat_labels6 = validat_features6.pop('bandgap')
test_labels6 = test_features6.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_6 = MyModel()

model_6.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_6 = model_6.fit(train_features6, train_labels6,
                    validation_data=(validat_features6, validat_labels6), 
                    epochs=3000, verbose=0)

test_results_6 = model_6.evaluate(test_features6, test_labels6, verbose=0)

model_6.summary()

MoWS_predictions6 = model_6.predict(MoWS_features).flatten()
MoWSe_predictions6 = model_6.predict(MoWSe_features).flatten()
MoWTe_predictions6 = model_6.predict(MoWTe_features).flatten()
MoSSe_predictions6 = model_6.predict(MoSSe_features).flatten()
MoSTe_predictions6 = model_6.predict(MoSTe_features).flatten()
MoSeTe_predictions6 = model_6.predict(MoSeTe_features).flatten()
WSSe_predictions6 = model_6.predict(WSSe_features).flatten()
WSTe_predictions6 = model_6.predict(WSTe_features).flatten()
WSeTe_predictions6 = model_6.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions6 = model_6.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions6 = model_6.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions6 = model_6.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions6 = model_6.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions6 = model_6.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions6 = model_6.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions6 = model_6.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions6 = model_6.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions6 = model_6.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions6 = model_6.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions6 = model_6.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions6 = model_6.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions6 = model_6.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions6 = model_6.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions6 = model_6.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions6 = model_6.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions6 = model_6.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions6 = model_6.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p6 = {}
MoWSe_p6 = {}
MoWTe_p6 = {}
MoSSe_p6 = {}
MoSTe_p6 = {}
MoSeTe_p6 = {}
WSSe_p6 = {}
WSTe_p6 = {}
WSeTe_p6 = {}

MoxWS2xSe_p6 = {}
MoxWS2xTe_p6 = {}
MoxWSe2xTe_p6 = {}
MoxWSSe2x_p6 = {}
MoxWSTe2x_p6 = {}
MoxWSeTe2x_p6 = {}
MoS2xSe2xTe_p6 = {}
MoS2xSeTe2x_p6 = {}
MoSSe2xTe2x_p6 = {}
WS2xSe2xTe_p6 = {}
WS2xSeTe2x_p6 = {}
WSSe2xTe2x_p6 = {}

MoWxS2xSe2xTe_p6 = {}
MoWxS2xSeTe2x_p6 = {}
MoWxSSe2xTe2x_p6 = {}
MoxWS2xSe2xTe_p6 = {}
MoxWS2xSeTe2x_p6 = {}
MoxWSSe2xTe2x_p6 = {}

MoWS_p6['gap6'] = MoWS_predictions6
MoWSe_p6['gap6'] = MoWSe_predictions6
MoWTe_p6['gap6'] = MoWTe_predictions6
MoSSe_p6['gap6'] = MoSSe_predictions6
MoSTe_p6['gap6'] = MoSTe_predictions6
MoSeTe_p6['gap6'] = MoSeTe_predictions6
WSSe_p6['gap6'] = WSSe_predictions6
WSTe_p6['gap6'] = WSTe_predictions6
WSeTe_p6['gap6'] = WSeTe_predictions6

MoxWS2xSe_p6['gap6'] = MoxWS2xSe_predictions6
MoxWS2xTe_p6['gap6'] = MoxWS2xTe_predictions6
MoxWSe2xTe_p6['gap6'] = MoxWSe2xTe_predictions6
MoxWSSe2x_p6['gap6'] = MoxWSSe2x_predictions6
MoxWSTe2x_p6['gap6'] = MoxWSTe2x_predictions6
MoxWSeTe2x_p6['gap6'] = MoxWSeTe2x_predictions6
MoS2xSe2xTe_p6['gap6'] = MoS2xSe2xTe_predictions6
MoS2xSeTe2x_p6['gap6'] = MoS2xSeTe2x_predictions6
MoSSe2xTe2x_p6['gap6'] = MoSSe2xTe2x_predictions6
WS2xSe2xTe_p6['gap6'] = WS2xSe2xTe_predictions6
WS2xSeTe2x_p6['gap6'] = WS2xSeTe2x_predictions6
WSSe2xTe2x_p6['gap6'] = WSSe2xTe2x_predictions6

MoWxS2xSe2xTe_p6['gap6'] = MoWxS2xSe2xTe_predictions6
MoWxS2xSeTe2x_p6['gap6'] = MoWxS2xSeTe2x_predictions6
MoWxSSe2xTe2x_p6['gap6'] = MoWxSSe2xTe2x_predictions6
MoxWS2xSe2xTe_p6['gap6'] = MoxWS2xSe2xTe_predictions6
MoxWS2xSeTe2x_p6['gap6'] = MoxWS2xSeTe2x_predictions6
MoxWSSe2xTe2x_p6['gap6'] = MoxWSSe2xTe2x_predictions6

MoWS_pre6 = pd.DataFrame(MoWS_p6)
MoWSe_pre6 = pd.DataFrame(MoWSe_p6)
MoWTe_pre6 = pd.DataFrame(MoWTe_p6)
MoSSe_pre6 = pd.DataFrame(MoSSe_p6)
MoSTe_pre6 = pd.DataFrame(MoSTe_p6)
MoSeTe_pre6 = pd.DataFrame(MoSeTe_p6)
WSSe_pre6 = pd.DataFrame(WSSe_p6)
WSTe_pre6 = pd.DataFrame(WSTe_p6)
WSeTe_pre6 = pd.DataFrame(WSeTe_p6)

MoxWS2xSe_pre6 = pd.DataFrame(MoxWS2xSe_p6)
MoxWS2xTe_pre6 = pd.DataFrame(MoxWS2xTe_p6)
MoxWSe2xTe_pre6 = pd.DataFrame(MoxWSe2xTe_p6)
MoxWSSe2x_pre6 = pd.DataFrame(MoxWSSe2x_p6)
MoxWSTe2x_pre6 = pd.DataFrame(MoxWSTe2x_p6)
MoxWSeTe2x_pre6 = pd.DataFrame(MoxWSeTe2x_p6)
MoS2xSe2xTe_pre6 = pd.DataFrame(MoS2xSe2xTe_p6)
MoS2xSeTe2x_pre6 = pd.DataFrame(MoS2xSeTe2x_p6)
MoSSe2xTe2x_pre6 = pd.DataFrame(MoSSe2xTe2x_p6)
WS2xSe2xTe_pre6 = pd.DataFrame(WS2xSe2xTe_p6)
WS2xSeTe2x_pre6 = pd.DataFrame(WS2xSeTe2x_p6)
WSSe2xTe2x_pre6 = pd.DataFrame(WSSe2xTe2x_p6)

MoWxS2xSe2xTe_pre6 = pd.DataFrame(MoWxS2xSe2xTe_p6)
MoWxS2xSeTe2x_pre6 = pd.DataFrame(MoWxS2xSeTe2x_p6)
MoWxSSe2xTe2x_pre6 = pd.DataFrame(MoWxSSe2xTe2x_p6)
MoxWS2xSe2xTe_pre6 = pd.DataFrame(MoxWS2xSe2xTe_p6)
MoxWS2xSeTe2x_pre6 = pd.DataFrame(MoxWS2xSeTe2x_p6)
MoxWSSe2xTe2x_pre6 = pd.DataFrame(MoxWSSe2xTe2x_p6)

# 7
train_dataset7 = pd.concat([dataset1, dataset3, dataset4])
validat_dataset7 = dataset2
test_dataset7 = dataset5

train_features7 = train_dataset7.copy()
validat_features7 = validat_dataset7.copy()
test_features7 = test_dataset7.copy()

train_labels7 = train_features7.pop('bandgap')
validat_labels7 = validat_features7.pop('bandgap')
test_labels7 = test_features7.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_7 = MyModel()

model_7.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_7 = model_7.fit(train_features7, train_labels7,
                    validation_data=(validat_features7, validat_labels7), 
                    epochs=3000, verbose=0)

test_results_7 = model_7.evaluate(test_features7, test_labels7, verbose=0)

model_7.summary()

MoWS_predictions7 = model_7.predict(MoWS_features).flatten()
MoWSe_predictions7 = model_7.predict(MoWSe_features).flatten()
MoWTe_predictions7 = model_7.predict(MoWTe_features).flatten()
MoSSe_predictions7 = model_7.predict(MoSSe_features).flatten()
MoSTe_predictions7 = model_7.predict(MoSTe_features).flatten()
MoSeTe_predictions7 = model_7.predict(MoSeTe_features).flatten()
WSSe_predictions7 = model_7.predict(WSSe_features).flatten()
WSTe_predictions7 = model_7.predict(WSTe_features).flatten()
WSeTe_predictions7 = model_7.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions7 = model_7.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions7 = model_7.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions7 = model_7.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions7 = model_7.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions7 = model_7.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions7 = model_7.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions7 = model_7.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions7 = model_7.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions7 = model_7.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions7 = model_7.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions7 = model_7.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions7 = model_7.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions7 = model_7.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions7 = model_7.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions7 = model_7.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions7 = model_7.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions7 = model_7.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions7 = model_7.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p7 = {}
MoWSe_p7 = {}
MoWTe_p7 = {}
MoSSe_p7 = {}
MoSTe_p7 = {}
MoSeTe_p7 = {}
WSSe_p7 = {}
WSTe_p7 = {}
WSeTe_p7 = {}

MoxWS2xSe_p7 = {}
MoxWS2xTe_p7 = {}
MoxWSe2xTe_p7 = {}
MoxWSSe2x_p7 = {}
MoxWSTe2x_p7 = {}
MoxWSeTe2x_p7 = {}
MoS2xSe2xTe_p7 = {}
MoS2xSeTe2x_p7 = {}
MoSSe2xTe2x_p7 = {}
WS2xSe2xTe_p7 = {}
WS2xSeTe2x_p7 = {}
WSSe2xTe2x_p7 = {}

MoWxS2xSe2xTe_p7 = {}
MoWxS2xSeTe2x_p7 = {}
MoWxSSe2xTe2x_p7 = {}
MoxWS2xSe2xTe_p7 = {}
MoxWS2xSeTe2x_p7 = {}
MoxWSSe2xTe2x_p7 = {}

MoWS_p7['gap7'] = MoWS_predictions7
MoWSe_p7['gap7'] = MoWSe_predictions7
MoWTe_p7['gap7'] = MoWTe_predictions7
MoSSe_p7['gap7'] = MoSSe_predictions7
MoSTe_p7['gap7'] = MoSTe_predictions7
MoSeTe_p7['gap7'] = MoSeTe_predictions7
WSSe_p7['gap7'] = WSSe_predictions7
WSTe_p7['gap7'] = WSTe_predictions7
WSeTe_p7['gap7'] = WSeTe_predictions7

MoxWS2xSe_p7['gap7'] = MoxWS2xSe_predictions7
MoxWS2xTe_p7['gap7'] = MoxWS2xTe_predictions7
MoxWSe2xTe_p7['gap7'] = MoxWSe2xTe_predictions7
MoxWSSe2x_p7['gap7'] = MoxWSSe2x_predictions7
MoxWSTe2x_p7['gap7'] = MoxWSTe2x_predictions7
MoxWSeTe2x_p7['gap7'] = MoxWSeTe2x_predictions7
MoS2xSe2xTe_p7['gap7'] = MoS2xSe2xTe_predictions7
MoS2xSeTe2x_p7['gap7'] = MoS2xSeTe2x_predictions7
MoSSe2xTe2x_p7['gap7'] = MoSSe2xTe2x_predictions7
WS2xSe2xTe_p7['gap7'] = WS2xSe2xTe_predictions7
WS2xSeTe2x_p7['gap7'] = WS2xSeTe2x_predictions7
WSSe2xTe2x_p7['gap7'] = WSSe2xTe2x_predictions7

MoWxS2xSe2xTe_p7['gap7'] = MoWxS2xSe2xTe_predictions7
MoWxS2xSeTe2x_p7['gap7'] = MoWxS2xSeTe2x_predictions7
MoWxSSe2xTe2x_p7['gap7'] = MoWxSSe2xTe2x_predictions7
MoxWS2xSe2xTe_p7['gap7'] = MoxWS2xSe2xTe_predictions7
MoxWS2xSeTe2x_p7['gap7'] = MoxWS2xSeTe2x_predictions7
MoxWSSe2xTe2x_p7['gap7'] = MoxWSSe2xTe2x_predictions7

MoWS_pre7 = pd.DataFrame(MoWS_p7)
MoWSe_pre7 = pd.DataFrame(MoWSe_p7)
MoWTe_pre7 = pd.DataFrame(MoWTe_p7)
MoSSe_pre7 = pd.DataFrame(MoSSe_p7)
MoSTe_pre7 = pd.DataFrame(MoSTe_p7)
MoSeTe_pre7 = pd.DataFrame(MoSeTe_p7)
WSSe_pre7 = pd.DataFrame(WSSe_p7)
WSTe_pre7 = pd.DataFrame(WSTe_p7)
WSeTe_pre7 = pd.DataFrame(WSeTe_p7)

MoxWS2xSe_pre7 = pd.DataFrame(MoxWS2xSe_p7)
MoxWS2xTe_pre7 = pd.DataFrame(MoxWS2xTe_p7)
MoxWSe2xTe_pre7 = pd.DataFrame(MoxWSe2xTe_p7)
MoxWSSe2x_pre7 = pd.DataFrame(MoxWSSe2x_p7)
MoxWSTe2x_pre7 = pd.DataFrame(MoxWSTe2x_p7)
MoxWSeTe2x_pre7 = pd.DataFrame(MoxWSeTe2x_p7)
MoS2xSe2xTe_pre7 = pd.DataFrame(MoS2xSe2xTe_p7)
MoS2xSeTe2x_pre7 = pd.DataFrame(MoS2xSeTe2x_p7)
MoSSe2xTe2x_pre7 = pd.DataFrame(MoSSe2xTe2x_p7)
WS2xSe2xTe_pre7 = pd.DataFrame(WS2xSe2xTe_p7)
WS2xSeTe2x_pre7 = pd.DataFrame(WS2xSeTe2x_p7)
WSSe2xTe2x_pre7 = pd.DataFrame(WSSe2xTe2x_p7)

MoWxS2xSe2xTe_pre7 = pd.DataFrame(MoWxS2xSe2xTe_p7)
MoWxS2xSeTe2x_pre7 = pd.DataFrame(MoWxS2xSeTe2x_p7)
MoWxSSe2xTe2x_pre7 = pd.DataFrame(MoWxSSe2xTe2x_p7)
MoxWS2xSe2xTe_pre7 = pd.DataFrame(MoxWS2xSe2xTe_p7)
MoxWS2xSeTe2x_pre7 = pd.DataFrame(MoxWS2xSeTe2x_p7)
MoxWSSe2xTe2x_pre7 = pd.DataFrame(MoxWSSe2xTe2x_p7)

# 8
train_dataset8 = pd.concat([dataset1, dataset3, dataset4])
validat_dataset8 = dataset5
test_dataset8 = dataset2

train_features8 = train_dataset8.copy()
validat_features8 = validat_dataset8.copy()
test_features8 = test_dataset8.copy()

train_labels8 = train_features8.pop('bandgap')
validat_labels8 = validat_features8.pop('bandgap')
test_labels8 = test_features8.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_8 = MyModel()

model_8.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_8 = model_8.fit(train_features8, train_labels8,
                    validation_data=(validat_features8, validat_labels8), 
                    epochs=3000, verbose=0)

test_results_8 = model_8.evaluate(test_features8, test_labels8, verbose=0)

model_8.summary()

MoWS_predictions8 = model_8.predict(MoWS_features).flatten()
MoWSe_predictions8 = model_8.predict(MoWSe_features).flatten()
MoWTe_predictions8 = model_8.predict(MoWTe_features).flatten()
MoSSe_predictions8 = model_8.predict(MoSSe_features).flatten()
MoSTe_predictions8 = model_8.predict(MoSTe_features).flatten()
MoSeTe_predictions8 = model_8.predict(MoSeTe_features).flatten()
WSSe_predictions8 = model_8.predict(WSSe_features).flatten()
WSTe_predictions8 = model_8.predict(WSTe_features).flatten()
WSeTe_predictions8 = model_8.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions8 = model_8.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions8 = model_8.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions8 = model_8.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions8 = model_8.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions8 = model_8.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions8 = model_8.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions8 = model_8.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions8 = model_8.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions8 = model_8.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions8 = model_8.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions8 = model_8.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions8 = model_8.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions8 = model_8.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions8 = model_8.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions8 = model_8.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions8 = model_8.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions8 = model_8.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions8 = model_8.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p8 = {}
MoWSe_p8 = {}
MoWTe_p8 = {}
MoSSe_p8 = {}
MoSTe_p8 = {}
MoSeTe_p8 = {}
WSSe_p8 = {}
WSTe_p8 = {}
WSeTe_p8 = {}

MoxWS2xSe_p8 = {}
MoxWS2xTe_p8 = {}
MoxWSe2xTe_p8 = {}
MoxWSSe2x_p8 = {}
MoxWSTe2x_p8 = {}
MoxWSeTe2x_p8 = {}
MoS2xSe2xTe_p8 = {}
MoS2xSeTe2x_p8 = {}
MoSSe2xTe2x_p8 = {}
WS2xSe2xTe_p8 = {}
WS2xSeTe2x_p8 = {}
WSSe2xTe2x_p8 = {}

MoWxS2xSe2xTe_p8 = {}
MoWxS2xSeTe2x_p8 = {}
MoWxSSe2xTe2x_p8 = {}
MoxWS2xSe2xTe_p8 = {}
MoxWS2xSeTe2x_p8 = {}
MoxWSSe2xTe2x_p8 = {}

MoWS_p8['gap8'] = MoWS_predictions8
MoWSe_p8['gap8'] = MoWSe_predictions8
MoWTe_p8['gap8'] = MoWTe_predictions8
MoSSe_p8['gap8'] = MoSSe_predictions8
MoSTe_p8['gap8'] = MoSTe_predictions8
MoSeTe_p8['gap8'] = MoSeTe_predictions8
WSSe_p8['gap8'] = WSSe_predictions8
WSTe_p8['gap8'] = WSTe_predictions8
WSeTe_p8['gap8'] = WSeTe_predictions8

MoxWS2xSe_p8['gap8'] = MoxWS2xSe_predictions8
MoxWS2xTe_p8['gap8'] = MoxWS2xTe_predictions8
MoxWSe2xTe_p8['gap8'] = MoxWSe2xTe_predictions8
MoxWSSe2x_p8['gap8'] = MoxWSSe2x_predictions8
MoxWSTe2x_p8['gap8'] = MoxWSTe2x_predictions8
MoxWSeTe2x_p8['gap8'] = MoxWSeTe2x_predictions8
MoS2xSe2xTe_p8['gap8'] = MoS2xSe2xTe_predictions8
MoS2xSeTe2x_p8['gap8'] = MoS2xSeTe2x_predictions8
MoSSe2xTe2x_p8['gap8'] = MoSSe2xTe2x_predictions8
WS2xSe2xTe_p8['gap8'] = WS2xSe2xTe_predictions8
WS2xSeTe2x_p8['gap8'] = WS2xSeTe2x_predictions8
WSSe2xTe2x_p8['gap8'] = WSSe2xTe2x_predictions8

MoWxS2xSe2xTe_p8['gap8'] = MoWxS2xSe2xTe_predictions8
MoWxS2xSeTe2x_p8['gap8'] = MoWxS2xSeTe2x_predictions8
MoWxSSe2xTe2x_p8['gap8'] = MoWxSSe2xTe2x_predictions8
MoxWS2xSe2xTe_p8['gap8'] = MoxWS2xSe2xTe_predictions8
MoxWS2xSeTe2x_p8['gap8'] = MoxWS2xSeTe2x_predictions8
MoxWSSe2xTe2x_p8['gap8'] = MoxWSSe2xTe2x_predictions8

MoWS_pre8 = pd.DataFrame(MoWS_p8)
MoWSe_pre8 = pd.DataFrame(MoWSe_p8)
MoWTe_pre8 = pd.DataFrame(MoWTe_p8)
MoSSe_pre8 = pd.DataFrame(MoSSe_p8)
MoSTe_pre8 = pd.DataFrame(MoSTe_p8)
MoSeTe_pre8 = pd.DataFrame(MoSeTe_p8)
WSSe_pre8 = pd.DataFrame(WSSe_p8)
WSTe_pre8 = pd.DataFrame(WSTe_p8)
WSeTe_pre8 = pd.DataFrame(WSeTe_p8)

MoxWS2xSe_pre8 = pd.DataFrame(MoxWS2xSe_p8)
MoxWS2xTe_pre8 = pd.DataFrame(MoxWS2xTe_p8)
MoxWSe2xTe_pre8 = pd.DataFrame(MoxWSe2xTe_p8)
MoxWSSe2x_pre8 = pd.DataFrame(MoxWSSe2x_p8)
MoxWSTe2x_pre8 = pd.DataFrame(MoxWSTe2x_p8)
MoxWSeTe2x_pre8 = pd.DataFrame(MoxWSeTe2x_p8)
MoS2xSe2xTe_pre8 = pd.DataFrame(MoS2xSe2xTe_p8)
MoS2xSeTe2x_pre8 = pd.DataFrame(MoS2xSeTe2x_p8)
MoSSe2xTe2x_pre8 = pd.DataFrame(MoSSe2xTe2x_p8)
WS2xSe2xTe_pre8 = pd.DataFrame(WS2xSe2xTe_p8)
WS2xSeTe2x_pre8 = pd.DataFrame(WS2xSeTe2x_p8)
WSSe2xTe2x_pre8 = pd.DataFrame(WSSe2xTe2x_p8)

MoWxS2xSe2xTe_pre8 = pd.DataFrame(MoWxS2xSe2xTe_p8)
MoWxS2xSeTe2x_pre8 = pd.DataFrame(MoWxS2xSeTe2x_p8)
MoWxSSe2xTe2x_pre8 = pd.DataFrame(MoWxSSe2xTe2x_p8)
MoxWS2xSe2xTe_pre8 = pd.DataFrame(MoxWS2xSe2xTe_p8)
MoxWS2xSeTe2x_pre8 = pd.DataFrame(MoxWS2xSeTe2x_p8)
MoxWSSe2xTe2x_pre8 = pd.DataFrame(MoxWSSe2xTe2x_p8)

# 9
train_dataset9 = pd.concat([dataset1, dataset3, dataset5])
validat_dataset9 = dataset2
test_dataset9 = dataset4

train_features9 = train_dataset9.copy()
validat_features9 = validat_dataset9.copy()
test_features9 = test_dataset9.copy()

train_labels9 = train_features9.pop('bandgap')
validat_labels9 = validat_features9.pop('bandgap')
test_labels9 = test_features9.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_9 = MyModel()

model_9.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_9 = model_9.fit(train_features9, train_labels9,
                    validation_data=(validat_features9, validat_labels9), 
                    epochs=3000, verbose=0)

test_results_9 = model_9.evaluate(test_features9, test_labels9, verbose=0)

model_9.summary()

MoWS_predictions9 = model_9.predict(MoWS_features).flatten()
MoWSe_predictions9 = model_9.predict(MoWSe_features).flatten()
MoWTe_predictions9 = model_9.predict(MoWTe_features).flatten()
MoSSe_predictions9 = model_9.predict(MoSSe_features).flatten()
MoSTe_predictions9 = model_9.predict(MoSTe_features).flatten()
MoSeTe_predictions9 = model_9.predict(MoSeTe_features).flatten()
WSSe_predictions9 = model_9.predict(WSSe_features).flatten()
WSTe_predictions9 = model_9.predict(WSTe_features).flatten()
WSeTe_predictions9 = model_9.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions9 = model_9.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions9 = model_9.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions9 = model_9.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions9 = model_9.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions9 = model_9.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions9 = model_9.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions9 = model_9.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions9 = model_9.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions9 = model_9.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions9 = model_9.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions9 = model_9.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions9 = model_9.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions9 = model_9.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions9 = model_9.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions9 = model_9.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions9 = model_9.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions9 = model_9.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions9 = model_9.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p9 = {}
MoWSe_p9 = {}
MoWTe_p9 = {}
MoSSe_p9 = {}
MoSTe_p9 = {}
MoSeTe_p9 = {}
WSSe_p9 = {}
WSTe_p9 = {}
WSeTe_p9 = {}

MoxWS2xSe_p9 = {}
MoxWS2xTe_p9 = {}
MoxWSe2xTe_p9 = {}
MoxWSSe2x_p9 = {}
MoxWSTe2x_p9 = {}
MoxWSeTe2x_p9 = {}
MoS2xSe2xTe_p9 = {}
MoS2xSeTe2x_p9 = {}
MoSSe2xTe2x_p9 = {}
WS2xSe2xTe_p9 = {}
WS2xSeTe2x_p9 = {}
WSSe2xTe2x_p9 = {}

MoWxS2xSe2xTe_p9 = {}
MoWxS2xSeTe2x_p9 = {}
MoWxSSe2xTe2x_p9 = {}
MoxWS2xSe2xTe_p9 = {}
MoxWS2xSeTe2x_p9 = {}
MoxWSSe2xTe2x_p9 = {}

MoWS_p9['gap9'] = MoWS_predictions9
MoWSe_p9['gap9'] = MoWSe_predictions9
MoWTe_p9['gap9'] = MoWTe_predictions9
MoSSe_p9['gap9'] = MoSSe_predictions9
MoSTe_p9['gap9'] = MoSTe_predictions9
MoSeTe_p9['gap9'] = MoSeTe_predictions9
WSSe_p9['gap9'] = WSSe_predictions9
WSTe_p9['gap9'] = WSTe_predictions9
WSeTe_p9['gap9'] = WSeTe_predictions9

MoxWS2xSe_p9['gap9'] = MoxWS2xSe_predictions9
MoxWS2xTe_p9['gap9'] = MoxWS2xTe_predictions9
MoxWSe2xTe_p9['gap9'] = MoxWSe2xTe_predictions9
MoxWSSe2x_p9['gap9'] = MoxWSSe2x_predictions9
MoxWSTe2x_p9['gap9'] = MoxWSTe2x_predictions9
MoxWSeTe2x_p9['gap9'] = MoxWSeTe2x_predictions9
MoS2xSe2xTe_p9['gap9'] = MoS2xSe2xTe_predictions9
MoS2xSeTe2x_p9['gap9'] = MoS2xSeTe2x_predictions9
MoSSe2xTe2x_p9['gap9'] = MoSSe2xTe2x_predictions9
WS2xSe2xTe_p9['gap9'] = WS2xSe2xTe_predictions9
WS2xSeTe2x_p9['gap9'] = WS2xSeTe2x_predictions9
WSSe2xTe2x_p9['gap9'] = WSSe2xTe2x_predictions9

MoWxS2xSe2xTe_p9['gap9'] = MoWxS2xSe2xTe_predictions9
MoWxS2xSeTe2x_p9['gap9'] = MoWxS2xSeTe2x_predictions9
MoWxSSe2xTe2x_p9['gap9'] = MoWxSSe2xTe2x_predictions9
MoxWS2xSe2xTe_p9['gap9'] = MoxWS2xSe2xTe_predictions9
MoxWS2xSeTe2x_p9['gap9'] = MoxWS2xSeTe2x_predictions9
MoxWSSe2xTe2x_p9['gap9'] = MoxWSSe2xTe2x_predictions9

MoWS_pre9 = pd.DataFrame(MoWS_p9)
MoWSe_pre9 = pd.DataFrame(MoWSe_p9)
MoWTe_pre9 = pd.DataFrame(MoWTe_p9)
MoSSe_pre9 = pd.DataFrame(MoSSe_p9)
MoSTe_pre9 = pd.DataFrame(MoSTe_p9)
MoSeTe_pre9 = pd.DataFrame(MoSeTe_p9)
WSSe_pre9 = pd.DataFrame(WSSe_p9)
WSTe_pre9 = pd.DataFrame(WSTe_p9)
WSeTe_pre9 = pd.DataFrame(WSeTe_p9)

MoxWS2xSe_pre9 = pd.DataFrame(MoxWS2xSe_p9)
MoxWS2xTe_pre9 = pd.DataFrame(MoxWS2xTe_p9)
MoxWSe2xTe_pre9 = pd.DataFrame(MoxWSe2xTe_p9)
MoxWSSe2x_pre9 = pd.DataFrame(MoxWSSe2x_p9)
MoxWSTe2x_pre9 = pd.DataFrame(MoxWSTe2x_p9)
MoxWSeTe2x_pre9 = pd.DataFrame(MoxWSeTe2x_p9)
MoS2xSe2xTe_pre9 = pd.DataFrame(MoS2xSe2xTe_p9)
MoS2xSeTe2x_pre9 = pd.DataFrame(MoS2xSeTe2x_p9)
MoSSe2xTe2x_pre9 = pd.DataFrame(MoSSe2xTe2x_p9)
WS2xSe2xTe_pre9 = pd.DataFrame(WS2xSe2xTe_p9)
WS2xSeTe2x_pre9 = pd.DataFrame(WS2xSeTe2x_p9)
WSSe2xTe2x_pre9 = pd.DataFrame(WSSe2xTe2x_p9)

MoWxS2xSe2xTe_pre9 = pd.DataFrame(MoWxS2xSe2xTe_p9)
MoWxS2xSeTe2x_pre9 = pd.DataFrame(MoWxS2xSeTe2x_p9)
MoWxSSe2xTe2x_pre9 = pd.DataFrame(MoWxSSe2xTe2x_p9)
MoxWS2xSe2xTe_pre9 = pd.DataFrame(MoxWS2xSe2xTe_p9)
MoxWS2xSeTe2x_pre9 = pd.DataFrame(MoxWS2xSeTe2x_p9)
MoxWSSe2xTe2x_pre9 = pd.DataFrame(MoxWSSe2xTe2x_p9)

# 10
train_dataset10 = pd.concat([dataset1, dataset3, dataset5])
validat_dataset10 = dataset4
test_dataset10 = dataset2

train_features10 = train_dataset10.copy()
validat_features10 = validat_dataset10.copy()
test_features10 = test_dataset10.copy()

train_labels10 = train_features10.pop('bandgap')
validat_labels10 = validat_features10.pop('bandgap')
test_labels10 = test_features10.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_10 = MyModel()

model_10.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_10 = model_10.fit(train_features10, train_labels10,
                    validation_data=(validat_features10, validat_labels10), 
                    epochs=3000, verbose=0)

test_results_10 = model_10.evaluate(test_features10, test_labels10, verbose=0)

model_10.summary()

MoWS_predictions10 = model_10.predict(MoWS_features).flatten()
MoWSe_predictions10 = model_10.predict(MoWSe_features).flatten()
MoWTe_predictions10 = model_10.predict(MoWTe_features).flatten()
MoSSe_predictions10 = model_10.predict(MoSSe_features).flatten()
MoSTe_predictions10 = model_10.predict(MoSTe_features).flatten()
MoSeTe_predictions10 = model_10.predict(MoSeTe_features).flatten()
WSSe_predictions10 = model_10.predict(WSSe_features).flatten()
WSTe_predictions10 = model_10.predict(WSTe_features).flatten()
WSeTe_predictions10 = model_10.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions10 = model_10.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions10 = model_10.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions10 = model_10.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions10 = model_10.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions10 = model_10.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions10 = model_10.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions10 = model_10.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions10 = model_10.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions10 = model_10.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions10 = model_10.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions10 = model_10.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions10 = model_10.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions10 = model_10.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions10 = model_10.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions10 = model_10.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions10 = model_10.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions10 = model_10.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions10 = model_10.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p10 = {}
MoWSe_p10 = {}
MoWTe_p10 = {}
MoSSe_p10 = {}
MoSTe_p10 = {}
MoSeTe_p10 = {}
WSSe_p10 = {}
WSTe_p10 = {}
WSeTe_p10 = {}

MoxWS2xSe_p10 = {}
MoxWS2xTe_p10 = {}
MoxWSe2xTe_p10 = {}
MoxWSSe2x_p10 = {}
MoxWSTe2x_p10 = {}
MoxWSeTe2x_p10 = {}
MoS2xSe2xTe_p10 = {}
MoS2xSeTe2x_p10 = {}
MoSSe2xTe2x_p10 = {}
WS2xSe2xTe_p10 = {}
WS2xSeTe2x_p10 = {}
WSSe2xTe2x_p10 = {}

MoWxS2xSe2xTe_p10 = {}
MoWxS2xSeTe2x_p10 = {}
MoWxSSe2xTe2x_p10 = {}
MoxWS2xSe2xTe_p10 = {}
MoxWS2xSeTe2x_p10 = {}
MoxWSSe2xTe2x_p10 = {}

MoWS_p10['gap10'] = MoWS_predictions10
MoWSe_p10['gap10'] = MoWSe_predictions10
MoWTe_p10['gap10'] = MoWTe_predictions10
MoSSe_p10['gap10'] = MoSSe_predictions10
MoSTe_p10['gap10'] = MoSTe_predictions10
MoSeTe_p10['gap10'] = MoSeTe_predictions10
WSSe_p10['gap10'] = WSSe_predictions10
WSTe_p10['gap10'] = WSTe_predictions10
WSeTe_p10['gap10'] = WSeTe_predictions10

MoxWS2xSe_p10['gap10'] = MoxWS2xSe_predictions10
MoxWS2xTe_p10['gap10'] = MoxWS2xTe_predictions10
MoxWSe2xTe_p10['gap10'] = MoxWSe2xTe_predictions10
MoxWSSe2x_p10['gap10'] = MoxWSSe2x_predictions10
MoxWSTe2x_p10['gap10'] = MoxWSTe2x_predictions10
MoxWSeTe2x_p10['gap10'] = MoxWSeTe2x_predictions10
MoS2xSe2xTe_p10['gap10'] = MoS2xSe2xTe_predictions10
MoS2xSeTe2x_p10['gap10'] = MoS2xSeTe2x_predictions10
MoSSe2xTe2x_p10['gap10'] = MoSSe2xTe2x_predictions10
WS2xSe2xTe_p10['gap10'] = WS2xSe2xTe_predictions10
WS2xSeTe2x_p10['gap10'] = WS2xSeTe2x_predictions10
WSSe2xTe2x_p10['gap10'] = WSSe2xTe2x_predictions10

MoWxS2xSe2xTe_p10['gap10'] = MoWxS2xSe2xTe_predictions10
MoWxS2xSeTe2x_p10['gap10'] = MoWxS2xSeTe2x_predictions10
MoWxSSe2xTe2x_p10['gap10'] = MoWxSSe2xTe2x_predictions10
MoxWS2xSe2xTe_p10['gap10'] = MoxWS2xSe2xTe_predictions10
MoxWS2xSeTe2x_p10['gap10'] = MoxWS2xSeTe2x_predictions10
MoxWSSe2xTe2x_p10['gap10'] = MoxWSSe2xTe2x_predictions10

MoWS_pre10 = pd.DataFrame(MoWS_p10)
MoWSe_pre10 = pd.DataFrame(MoWSe_p10)
MoWTe_pre10 = pd.DataFrame(MoWTe_p10)
MoSSe_pre10 = pd.DataFrame(MoSSe_p10)
MoSTe_pre10 = pd.DataFrame(MoSTe_p10)
MoSeTe_pre10 = pd.DataFrame(MoSeTe_p10)
WSSe_pre10 = pd.DataFrame(WSSe_p10)
WSTe_pre10 = pd.DataFrame(WSTe_p10)
WSeTe_pre10 = pd.DataFrame(WSeTe_p10)

MoxWS2xSe_pre10 = pd.DataFrame(MoxWS2xSe_p10)
MoxWS2xTe_pre10 = pd.DataFrame(MoxWS2xTe_p10)
MoxWSe2xTe_pre10 = pd.DataFrame(MoxWSe2xTe_p10)
MoxWSSe2x_pre10 = pd.DataFrame(MoxWSSe2x_p10)
MoxWSTe2x_pre10 = pd.DataFrame(MoxWSTe2x_p10)
MoxWSeTe2x_pre10 = pd.DataFrame(MoxWSeTe2x_p10)
MoS2xSe2xTe_pre10 = pd.DataFrame(MoS2xSe2xTe_p10)
MoS2xSeTe2x_pre10 = pd.DataFrame(MoS2xSeTe2x_p10)
MoSSe2xTe2x_pre10 = pd.DataFrame(MoSSe2xTe2x_p10)
WS2xSe2xTe_pre10 = pd.DataFrame(WS2xSe2xTe_p10)
WS2xSeTe2x_pre10 = pd.DataFrame(WS2xSeTe2x_p10)
WSSe2xTe2x_pre10 = pd.DataFrame(WSSe2xTe2x_p10)

MoWxS2xSe2xTe_pre10 = pd.DataFrame(MoWxS2xSe2xTe_p10)
MoWxS2xSeTe2x_pre10 = pd.DataFrame(MoWxS2xSeTe2x_p10)
MoWxSSe2xTe2x_pre10 = pd.DataFrame(MoWxSSe2xTe2x_p10)
MoxWS2xSe2xTe_pre10 = pd.DataFrame(MoxWS2xSe2xTe_p10)
MoxWS2xSeTe2x_pre10 = pd.DataFrame(MoxWS2xSeTe2x_p10)
MoxWSSe2xTe2x_pre10 = pd.DataFrame(MoxWSSe2xTe2x_p10)

# 11
train_dataset11 = pd.concat([dataset1, dataset4, dataset5])
validat_dataset11 = dataset2
test_dataset11 = dataset3

train_features11 = train_dataset11.copy()
validat_features11 = validat_dataset11.copy()
test_features11 = test_dataset11.copy()

train_labels11 = train_features11.pop('bandgap')
validat_labels11 = validat_features11.pop('bandgap')
test_labels11 = test_features11.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_11 = MyModel()

model_11.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_11 = model_11.fit(train_features11, train_labels11,
                    validation_data=(validat_features11, validat_labels11), 
                    epochs=3000, verbose=0)

test_results_11 = model_11.evaluate(test_features11, test_labels11, verbose=0)

model_11.summary()

MoWS_predictions11 = model_11.predict(MoWS_features).flatten()
MoWSe_predictions11 = model_11.predict(MoWSe_features).flatten()
MoWTe_predictions11 = model_11.predict(MoWTe_features).flatten()
MoSSe_predictions11 = model_11.predict(MoSSe_features).flatten()
MoSTe_predictions11 = model_11.predict(MoSTe_features).flatten()
MoSeTe_predictions11 = model_11.predict(MoSeTe_features).flatten()
WSSe_predictions11 = model_11.predict(WSSe_features).flatten()
WSTe_predictions11 = model_11.predict(WSTe_features).flatten()
WSeTe_predictions11 = model_11.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions11 = model_11.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions11 = model_11.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions11 = model_11.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions11 = model_11.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions11 = model_11.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions11 = model_11.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions11 = model_11.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions11 = model_11.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions11 = model_11.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions11 = model_11.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions11 = model_11.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions11 = model_11.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions11 = model_11.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions11 = model_11.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions11 = model_11.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions11 = model_11.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions11 = model_11.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions11 = model_11.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p11 = {}
MoWSe_p11 = {}
MoWTe_p11 = {}
MoSSe_p11 = {}
MoSTe_p11 = {}
MoSeTe_p11 = {}
WSSe_p11 = {}
WSTe_p11 = {}
WSeTe_p11 = {}

MoxWS2xSe_p11 = {}
MoxWS2xTe_p11 = {}
MoxWSe2xTe_p11 = {}
MoxWSSe2x_p11 = {}
MoxWSTe2x_p11 = {}
MoxWSeTe2x_p11 = {}
MoS2xSe2xTe_p11 = {}
MoS2xSeTe2x_p11 = {}
MoSSe2xTe2x_p11 = {}
WS2xSe2xTe_p11 = {}
WS2xSeTe2x_p11 = {}
WSSe2xTe2x_p11 = {}

MoWxS2xSe2xTe_p11 = {}
MoWxS2xSeTe2x_p11 = {}
MoWxSSe2xTe2x_p11 = {}
MoxWS2xSe2xTe_p11 = {}
MoxWS2xSeTe2x_p11 = {}
MoxWSSe2xTe2x_p11 = {}

MoWS_p11['gap11'] = MoWS_predictions11
MoWSe_p11['gap11'] = MoWSe_predictions11
MoWTe_p11['gap11'] = MoWTe_predictions11
MoSSe_p11['gap11'] = MoSSe_predictions11
MoSTe_p11['gap11'] = MoSTe_predictions11
MoSeTe_p11['gap11'] = MoSeTe_predictions11
WSSe_p11['gap11'] = WSSe_predictions11
WSTe_p11['gap11'] = WSTe_predictions11
WSeTe_p11['gap11'] = WSeTe_predictions11

MoxWS2xSe_p11['gap11'] = MoxWS2xSe_predictions11
MoxWS2xTe_p11['gap11'] = MoxWS2xTe_predictions11
MoxWSe2xTe_p11['gap11'] = MoxWSe2xTe_predictions11
MoxWSSe2x_p11['gap11'] = MoxWSSe2x_predictions11
MoxWSTe2x_p11['gap11'] = MoxWSTe2x_predictions11
MoxWSeTe2x_p11['gap11'] = MoxWSeTe2x_predictions11
MoS2xSe2xTe_p11['gap11'] = MoS2xSe2xTe_predictions11
MoS2xSeTe2x_p11['gap11'] = MoS2xSeTe2x_predictions11
MoSSe2xTe2x_p11['gap11'] = MoSSe2xTe2x_predictions11
WS2xSe2xTe_p11['gap11'] = WS2xSe2xTe_predictions11
WS2xSeTe2x_p11['gap11'] = WS2xSeTe2x_predictions11
WSSe2xTe2x_p11['gap11'] = WSSe2xTe2x_predictions11

MoWxS2xSe2xTe_p11['gap11'] = MoWxS2xSe2xTe_predictions11
MoWxS2xSeTe2x_p11['gap11'] = MoWxS2xSeTe2x_predictions11
MoWxSSe2xTe2x_p11['gap11'] = MoWxSSe2xTe2x_predictions11
MoxWS2xSe2xTe_p11['gap11'] = MoxWS2xSe2xTe_predictions11
MoxWS2xSeTe2x_p11['gap11'] = MoxWS2xSeTe2x_predictions11
MoxWSSe2xTe2x_p11['gap11'] = MoxWSSe2xTe2x_predictions11

MoWS_pre11 = pd.DataFrame(MoWS_p11)
MoWSe_pre11 = pd.DataFrame(MoWSe_p11)
MoWTe_pre11 = pd.DataFrame(MoWTe_p11)
MoSSe_pre11 = pd.DataFrame(MoSSe_p11)
MoSTe_pre11 = pd.DataFrame(MoSTe_p11)
MoSeTe_pre11 = pd.DataFrame(MoSeTe_p11)
WSSe_pre11 = pd.DataFrame(WSSe_p11)
WSTe_pre11 = pd.DataFrame(WSTe_p11)
WSeTe_pre11 = pd.DataFrame(WSeTe_p11)

MoxWS2xSe_pre11 = pd.DataFrame(MoxWS2xSe_p11)
MoxWS2xTe_pre11 = pd.DataFrame(MoxWS2xTe_p11)
MoxWSe2xTe_pre11 = pd.DataFrame(MoxWSe2xTe_p11)
MoxWSSe2x_pre11 = pd.DataFrame(MoxWSSe2x_p11)
MoxWSTe2x_pre11 = pd.DataFrame(MoxWSTe2x_p11)
MoxWSeTe2x_pre11 = pd.DataFrame(MoxWSeTe2x_p11)
MoS2xSe2xTe_pre11 = pd.DataFrame(MoS2xSe2xTe_p11)
MoS2xSeTe2x_pre11 = pd.DataFrame(MoS2xSeTe2x_p11)
MoSSe2xTe2x_pre11 = pd.DataFrame(MoSSe2xTe2x_p11)
WS2xSe2xTe_pre11 = pd.DataFrame(WS2xSe2xTe_p11)
WS2xSeTe2x_pre11 = pd.DataFrame(WS2xSeTe2x_p11)
WSSe2xTe2x_pre11 = pd.DataFrame(WSSe2xTe2x_p11)

MoWxS2xSe2xTe_pre11 = pd.DataFrame(MoWxS2xSe2xTe_p11)
MoWxS2xSeTe2x_pre11 = pd.DataFrame(MoWxS2xSeTe2x_p11)
MoWxSSe2xTe2x_pre11 = pd.DataFrame(MoWxSSe2xTe2x_p11)
MoxWS2xSe2xTe_pre11 = pd.DataFrame(MoxWS2xSe2xTe_p11)
MoxWS2xSeTe2x_pre11 = pd.DataFrame(MoxWS2xSeTe2x_p11)
MoxWSSe2xTe2x_pre11 = pd.DataFrame(MoxWSSe2xTe2x_p11)

# 12
train_dataset12 = pd.concat([dataset1, dataset4, dataset5])
validat_dataset12 = dataset3
test_dataset12 = dataset2

train_features12 = train_dataset12.copy()
validat_features12 = validat_dataset12.copy()
test_features12 = test_dataset12.copy()

train_labels12 = train_features12.pop('bandgap')
validat_labels12 = validat_features12.pop('bandgap')
test_labels12 = test_features12.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_12 = MyModel()

model_12.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_12 = model_12.fit(train_features12, train_labels12,
                    validation_data=(validat_features12, validat_labels12), 
                    epochs=3000, verbose=0)

test_results_12 = model_12.evaluate(test_features12, test_labels12, verbose=0)

model_12.summary()

MoWS_predictions12 = model_12.predict(MoWS_features).flatten()
MoWSe_predictions12 = model_12.predict(MoWSe_features).flatten()
MoWTe_predictions12 = model_12.predict(MoWTe_features).flatten()
MoSSe_predictions12 = model_12.predict(MoSSe_features).flatten()
MoSTe_predictions12 = model_12.predict(MoSTe_features).flatten()
MoSeTe_predictions12 = model_12.predict(MoSeTe_features).flatten()
WSSe_predictions12 = model_12.predict(WSSe_features).flatten()
WSTe_predictions12 = model_12.predict(WSTe_features).flatten()
WSeTe_predictions12 = model_12.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions12 = model_12.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions12 = model_12.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions12 = model_12.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions12 = model_12.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions12 = model_12.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions12 = model_12.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions12 = model_12.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions12 = model_12.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions12 = model_12.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions12 = model_12.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions12 = model_12.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions12 = model_12.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions12 = model_12.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions12 = model_12.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions12 = model_12.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions12 = model_12.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions12 = model_12.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions12 = model_12.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p12 = {}
MoWSe_p12 = {}
MoWTe_p12 = {}
MoSSe_p12 = {}
MoSTe_p12 = {}
MoSeTe_p12 = {}
WSSe_p12 = {}
WSTe_p12 = {}
WSeTe_p12 = {}

MoxWS2xSe_p12 = {}
MoxWS2xTe_p12 = {}
MoxWSe2xTe_p12 = {}
MoxWSSe2x_p12 = {}
MoxWSTe2x_p12 = {}
MoxWSeTe2x_p12 = {}
MoS2xSe2xTe_p12 = {}
MoS2xSeTe2x_p12 = {}
MoSSe2xTe2x_p12 = {}
WS2xSe2xTe_p12 = {}
WS2xSeTe2x_p12 = {}
WSSe2xTe2x_p12 = {}

MoWxS2xSe2xTe_p12 = {}
MoWxS2xSeTe2x_p12 = {}
MoWxSSe2xTe2x_p12 = {}
MoxWS2xSe2xTe_p12 = {}
MoxWS2xSeTe2x_p12 = {}
MoxWSSe2xTe2x_p12 = {}

MoWS_p12['gap12'] = MoWS_predictions12
MoWSe_p12['gap12'] = MoWSe_predictions12
MoWTe_p12['gap12'] = MoWTe_predictions12
MoSSe_p12['gap12'] = MoSSe_predictions12
MoSTe_p12['gap12'] = MoSTe_predictions12
MoSeTe_p12['gap12'] = MoSeTe_predictions12
WSSe_p12['gap12'] = WSSe_predictions12
WSTe_p12['gap12'] = WSTe_predictions12
WSeTe_p12['gap12'] = WSeTe_predictions12

MoxWS2xSe_p12['gap12'] = MoxWS2xSe_predictions12
MoxWS2xTe_p12['gap12'] = MoxWS2xTe_predictions12
MoxWSe2xTe_p12['gap12'] = MoxWSe2xTe_predictions12
MoxWSSe2x_p12['gap12'] = MoxWSSe2x_predictions12
MoxWSTe2x_p12['gap12'] = MoxWSTe2x_predictions12
MoxWSeTe2x_p12['gap12'] = MoxWSeTe2x_predictions12
MoS2xSe2xTe_p12['gap12'] = MoS2xSe2xTe_predictions12
MoS2xSeTe2x_p12['gap12'] = MoS2xSeTe2x_predictions12
MoSSe2xTe2x_p12['gap12'] = MoSSe2xTe2x_predictions12
WS2xSe2xTe_p12['gap12'] = WS2xSe2xTe_predictions12
WS2xSeTe2x_p12['gap12'] = WS2xSeTe2x_predictions12
WSSe2xTe2x_p12['gap12'] = WSSe2xTe2x_predictions12

MoWxS2xSe2xTe_p12['gap12'] = MoWxS2xSe2xTe_predictions12
MoWxS2xSeTe2x_p12['gap12'] = MoWxS2xSeTe2x_predictions12
MoWxSSe2xTe2x_p12['gap12'] = MoWxSSe2xTe2x_predictions12
MoxWS2xSe2xTe_p12['gap12'] = MoxWS2xSe2xTe_predictions12
MoxWS2xSeTe2x_p12['gap12'] = MoxWS2xSeTe2x_predictions12
MoxWSSe2xTe2x_p12['gap12'] = MoxWSSe2xTe2x_predictions12

MoWS_pre12 = pd.DataFrame(MoWS_p12)
MoWSe_pre12 = pd.DataFrame(MoWSe_p12)
MoWTe_pre12 = pd.DataFrame(MoWTe_p12)
MoSSe_pre12 = pd.DataFrame(MoSSe_p12)
MoSTe_pre12 = pd.DataFrame(MoSTe_p12)
MoSeTe_pre12 = pd.DataFrame(MoSeTe_p12)
WSSe_pre12 = pd.DataFrame(WSSe_p12)
WSTe_pre12 = pd.DataFrame(WSTe_p12)
WSeTe_pre12 = pd.DataFrame(WSeTe_p12)

MoxWS2xSe_pre12 = pd.DataFrame(MoxWS2xSe_p12)
MoxWS2xTe_pre12 = pd.DataFrame(MoxWS2xTe_p12)
MoxWSe2xTe_pre12 = pd.DataFrame(MoxWSe2xTe_p12)
MoxWSSe2x_pre12 = pd.DataFrame(MoxWSSe2x_p12)
MoxWSTe2x_pre12 = pd.DataFrame(MoxWSTe2x_p12)
MoxWSeTe2x_pre12 = pd.DataFrame(MoxWSeTe2x_p12)
MoS2xSe2xTe_pre12 = pd.DataFrame(MoS2xSe2xTe_p12)
MoS2xSeTe2x_pre12 = pd.DataFrame(MoS2xSeTe2x_p12)
MoSSe2xTe2x_pre12 = pd.DataFrame(MoSSe2xTe2x_p12)
WS2xSe2xTe_pre12 = pd.DataFrame(WS2xSe2xTe_p12)
WS2xSeTe2x_pre12 = pd.DataFrame(WS2xSeTe2x_p12)
WSSe2xTe2x_pre12 = pd.DataFrame(WSSe2xTe2x_p12)

MoWxS2xSe2xTe_pre12 = pd.DataFrame(MoWxS2xSe2xTe_p12)
MoWxS2xSeTe2x_pre12 = pd.DataFrame(MoWxS2xSeTe2x_p12)
MoWxSSe2xTe2x_pre12 = pd.DataFrame(MoWxSSe2xTe2x_p12)
MoxWS2xSe2xTe_pre12 = pd.DataFrame(MoxWS2xSe2xTe_p12)
MoxWS2xSeTe2x_pre12 = pd.DataFrame(MoxWS2xSeTe2x_p12)
MoxWSSe2xTe2x_pre12 = pd.DataFrame(MoxWSSe2xTe2x_p12)

# 13
train_dataset13 = pd.concat([dataset2, dataset3, dataset4])
validat_dataset13 = dataset1
test_dataset13 = dataset5

train_features13 = train_dataset13.copy()
validat_features13 = validat_dataset13.copy()
test_features13 = test_dataset13.copy()

train_labels13 = train_features13.pop('bandgap')
validat_labels13 = validat_features13.pop('bandgap')
test_labels13 = test_features13.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_13 = MyModel()

model_13.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_13 = model_13.fit(train_features13, train_labels13,
                    validation_data=(validat_features13, validat_labels13), 
                    epochs=3000, verbose=0)

test_results_13 = model_13.evaluate(test_features13, test_labels13, verbose=0)

model_13.summary()

MoWS_predictions13 = model_13.predict(MoWS_features).flatten()
MoWSe_predictions13 = model_13.predict(MoWSe_features).flatten()
MoWTe_predictions13 = model_13.predict(MoWTe_features).flatten()
MoSSe_predictions13 = model_13.predict(MoSSe_features).flatten()
MoSTe_predictions13 = model_13.predict(MoSTe_features).flatten()
MoSeTe_predictions13 = model_13.predict(MoSeTe_features).flatten()
WSSe_predictions13 = model_13.predict(WSSe_features).flatten()
WSTe_predictions13 = model_13.predict(WSTe_features).flatten()
WSeTe_predictions13 = model_13.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions13 = model_13.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions13 = model_13.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions13 = model_13.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions13 = model_13.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions13 = model_13.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions13 = model_13.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions13 = model_13.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions13 = model_13.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions13 = model_13.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions13 = model_13.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions13 = model_13.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions13 = model_13.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions13 = model_13.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions13 = model_13.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions13 = model_13.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions13 = model_13.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions13 = model_13.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions13 = model_13.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p13 = {}
MoWSe_p13 = {}
MoWTe_p13 = {}
MoSSe_p13 = {}
MoSTe_p13 = {}
MoSeTe_p13 = {}
WSSe_p13 = {}
WSTe_p13 = {}
WSeTe_p13 = {}

MoxWS2xSe_p13 = {}
MoxWS2xTe_p13 = {}
MoxWSe2xTe_p13 = {}
MoxWSSe2x_p13 = {}
MoxWSTe2x_p13 = {}
MoxWSeTe2x_p13 = {}
MoS2xSe2xTe_p13 = {}
MoS2xSeTe2x_p13 = {}
MoSSe2xTe2x_p13 = {}
WS2xSe2xTe_p13 = {}
WS2xSeTe2x_p13 = {}
WSSe2xTe2x_p13 = {}

MoWxS2xSe2xTe_p13 = {}
MoWxS2xSeTe2x_p13 = {}
MoWxSSe2xTe2x_p13 = {}
MoxWS2xSe2xTe_p13 = {}
MoxWS2xSeTe2x_p13 = {}
MoxWSSe2xTe2x_p13 = {}

MoWS_p13['gap13'] = MoWS_predictions13
MoWSe_p13['gap13'] = MoWSe_predictions13
MoWTe_p13['gap13'] = MoWTe_predictions13
MoSSe_p13['gap13'] = MoSSe_predictions13
MoSTe_p13['gap13'] = MoSTe_predictions13
MoSeTe_p13['gap13'] = MoSeTe_predictions13
WSSe_p13['gap13'] = WSSe_predictions13
WSTe_p13['gap13'] = WSTe_predictions13
WSeTe_p13['gap13'] = WSeTe_predictions13

MoxWS2xSe_p13['gap13'] = MoxWS2xSe_predictions13
MoxWS2xTe_p13['gap13'] = MoxWS2xTe_predictions13
MoxWSe2xTe_p13['gap13'] = MoxWSe2xTe_predictions13
MoxWSSe2x_p13['gap13'] = MoxWSSe2x_predictions13
MoxWSTe2x_p13['gap13'] = MoxWSTe2x_predictions13
MoxWSeTe2x_p13['gap13'] = MoxWSeTe2x_predictions13
MoS2xSe2xTe_p13['gap13'] = MoS2xSe2xTe_predictions13
MoS2xSeTe2x_p13['gap13'] = MoS2xSeTe2x_predictions13
MoSSe2xTe2x_p13['gap13'] = MoSSe2xTe2x_predictions13
WS2xSe2xTe_p13['gap13'] = WS2xSe2xTe_predictions13
WS2xSeTe2x_p13['gap13'] = WS2xSeTe2x_predictions13
WSSe2xTe2x_p13['gap13'] = WSSe2xTe2x_predictions13

MoWxS2xSe2xTe_p13['gap13'] = MoWxS2xSe2xTe_predictions13
MoWxS2xSeTe2x_p13['gap13'] = MoWxS2xSeTe2x_predictions13
MoWxSSe2xTe2x_p13['gap13'] = MoWxSSe2xTe2x_predictions13
MoxWS2xSe2xTe_p13['gap13'] = MoxWS2xSe2xTe_predictions13
MoxWS2xSeTe2x_p13['gap13'] = MoxWS2xSeTe2x_predictions13
MoxWSSe2xTe2x_p13['gap13'] = MoxWSSe2xTe2x_predictions13

MoWS_pre13 = pd.DataFrame(MoWS_p13)
MoWSe_pre13 = pd.DataFrame(MoWSe_p13)
MoWTe_pre13 = pd.DataFrame(MoWTe_p13)
MoSSe_pre13 = pd.DataFrame(MoSSe_p13)
MoSTe_pre13 = pd.DataFrame(MoSTe_p13)
MoSeTe_pre13 = pd.DataFrame(MoSeTe_p13)
WSSe_pre13 = pd.DataFrame(WSSe_p13)
WSTe_pre13 = pd.DataFrame(WSTe_p13)
WSeTe_pre13 = pd.DataFrame(WSeTe_p13)

MoxWS2xSe_pre13 = pd.DataFrame(MoxWS2xSe_p13)
MoxWS2xTe_pre13 = pd.DataFrame(MoxWS2xTe_p13)
MoxWSe2xTe_pre13 = pd.DataFrame(MoxWSe2xTe_p13)
MoxWSSe2x_pre13 = pd.DataFrame(MoxWSSe2x_p13)
MoxWSTe2x_pre13 = pd.DataFrame(MoxWSTe2x_p13)
MoxWSeTe2x_pre13 = pd.DataFrame(MoxWSeTe2x_p13)
MoS2xSe2xTe_pre13 = pd.DataFrame(MoS2xSe2xTe_p13)
MoS2xSeTe2x_pre13 = pd.DataFrame(MoS2xSeTe2x_p13)
MoSSe2xTe2x_pre13 = pd.DataFrame(MoSSe2xTe2x_p13)
WS2xSe2xTe_pre13 = pd.DataFrame(WS2xSe2xTe_p13)
WS2xSeTe2x_pre13 = pd.DataFrame(WS2xSeTe2x_p13)
WSSe2xTe2x_pre13 = pd.DataFrame(WSSe2xTe2x_p13)

MoWxS2xSe2xTe_pre13 = pd.DataFrame(MoWxS2xSe2xTe_p13)
MoWxS2xSeTe2x_pre13 = pd.DataFrame(MoWxS2xSeTe2x_p13)
MoWxSSe2xTe2x_pre13 = pd.DataFrame(MoWxSSe2xTe2x_p13)
MoxWS2xSe2xTe_pre13 = pd.DataFrame(MoxWS2xSe2xTe_p13)
MoxWS2xSeTe2x_pre13 = pd.DataFrame(MoxWS2xSeTe2x_p13)
MoxWSSe2xTe2x_pre13 = pd.DataFrame(MoxWSSe2xTe2x_p13)

# 14
train_dataset14 = pd.concat([dataset2, dataset3, dataset4])
validat_dataset14 = dataset5
test_dataset14 = dataset1

train_features14 = train_dataset14.copy()
validat_features14 = validat_dataset14.copy()
test_features14 = test_dataset14.copy()

train_labels14 = train_features14.pop('bandgap')
validat_labels14 = validat_features14.pop('bandgap')
test_labels14 = test_features14.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_14 = MyModel()

model_14.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_14 = model_14.fit(train_features14, train_labels14,
                    validation_data=(validat_features14, validat_labels14), 
                    epochs=3000, verbose=0)

test_results_14 = model_14.evaluate(test_features14, test_labels14, verbose=0)

model_14.summary()

MoWS_predictions14 = model_14.predict(MoWS_features).flatten()
MoWSe_predictions14 = model_14.predict(MoWSe_features).flatten()
MoWTe_predictions14 = model_14.predict(MoWTe_features).flatten()
MoSSe_predictions14 = model_14.predict(MoSSe_features).flatten()
MoSTe_predictions14 = model_14.predict(MoSTe_features).flatten()
MoSeTe_predictions14 = model_14.predict(MoSeTe_features).flatten()
WSSe_predictions14 = model_14.predict(WSSe_features).flatten()
WSTe_predictions14 = model_14.predict(WSTe_features).flatten()
WSeTe_predictions14 = model_14.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions14 = model_14.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions14 = model_14.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions14 = model_14.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions14 = model_14.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions14 = model_14.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions14 = model_14.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions14 = model_14.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions14 = model_14.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions14 = model_14.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions14 = model_14.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions14 = model_14.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions14 = model_14.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions14 = model_14.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions14 = model_14.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions14 = model_14.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions14 = model_14.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions14 = model_14.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions14 = model_14.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p14 = {}
MoWSe_p14 = {}
MoWTe_p14 = {}
MoSSe_p14 = {}
MoSTe_p14 = {}
MoSeTe_p14 = {}
WSSe_p14 = {}
WSTe_p14 = {}
WSeTe_p14 = {}

MoxWS2xSe_p14 = {}
MoxWS2xTe_p14 = {}
MoxWSe2xTe_p14 = {}
MoxWSSe2x_p14 = {}
MoxWSTe2x_p14 = {}
MoxWSeTe2x_p14 = {}
MoS2xSe2xTe_p14 = {}
MoS2xSeTe2x_p14 = {}
MoSSe2xTe2x_p14 = {}
WS2xSe2xTe_p14 = {}
WS2xSeTe2x_p14 = {}
WSSe2xTe2x_p14 = {}

MoWxS2xSe2xTe_p14 = {}
MoWxS2xSeTe2x_p14 = {}
MoWxSSe2xTe2x_p14 = {}
MoxWS2xSe2xTe_p14 = {}
MoxWS2xSeTe2x_p14 = {}
MoxWSSe2xTe2x_p14 = {}

MoWS_p14['gap14'] = MoWS_predictions14
MoWSe_p14['gap14'] = MoWSe_predictions14
MoWTe_p14['gap14'] = MoWTe_predictions14
MoSSe_p14['gap14'] = MoSSe_predictions14
MoSTe_p14['gap14'] = MoSTe_predictions14
MoSeTe_p14['gap14'] = MoSeTe_predictions14
WSSe_p14['gap14'] = WSSe_predictions14
WSTe_p14['gap14'] = WSTe_predictions14
WSeTe_p14['gap14'] = WSeTe_predictions14

MoxWS2xSe_p14['gap14'] = MoxWS2xSe_predictions14
MoxWS2xTe_p14['gap14'] = MoxWS2xTe_predictions14
MoxWSe2xTe_p14['gap14'] = MoxWSe2xTe_predictions14
MoxWSSe2x_p14['gap14'] = MoxWSSe2x_predictions14
MoxWSTe2x_p14['gap14'] = MoxWSTe2x_predictions14
MoxWSeTe2x_p14['gap14'] = MoxWSeTe2x_predictions14
MoS2xSe2xTe_p14['gap14'] = MoS2xSe2xTe_predictions14
MoS2xSeTe2x_p14['gap14'] = MoS2xSeTe2x_predictions14
MoSSe2xTe2x_p14['gap14'] = MoSSe2xTe2x_predictions14
WS2xSe2xTe_p14['gap14'] = WS2xSe2xTe_predictions14
WS2xSeTe2x_p14['gap14'] = WS2xSeTe2x_predictions14
WSSe2xTe2x_p14['gap14'] = WSSe2xTe2x_predictions14

MoWxS2xSe2xTe_p14['gap14'] = MoWxS2xSe2xTe_predictions14
MoWxS2xSeTe2x_p14['gap14'] = MoWxS2xSeTe2x_predictions14
MoWxSSe2xTe2x_p14['gap14'] = MoWxSSe2xTe2x_predictions14
MoxWS2xSe2xTe_p14['gap14'] = MoxWS2xSe2xTe_predictions14
MoxWS2xSeTe2x_p14['gap14'] = MoxWS2xSeTe2x_predictions14
MoxWSSe2xTe2x_p14['gap14'] = MoxWSSe2xTe2x_predictions14

MoWS_pre14 = pd.DataFrame(MoWS_p14)
MoWSe_pre14 = pd.DataFrame(MoWSe_p14)
MoWTe_pre14 = pd.DataFrame(MoWTe_p14)
MoSSe_pre14 = pd.DataFrame(MoSSe_p14)
MoSTe_pre14 = pd.DataFrame(MoSTe_p14)
MoSeTe_pre14 = pd.DataFrame(MoSeTe_p14)
WSSe_pre14 = pd.DataFrame(WSSe_p14)
WSTe_pre14 = pd.DataFrame(WSTe_p14)
WSeTe_pre14 = pd.DataFrame(WSeTe_p14)

MoxWS2xSe_pre14 = pd.DataFrame(MoxWS2xSe_p14)
MoxWS2xTe_pre14 = pd.DataFrame(MoxWS2xTe_p14)
MoxWSe2xTe_pre14 = pd.DataFrame(MoxWSe2xTe_p14)
MoxWSSe2x_pre14 = pd.DataFrame(MoxWSSe2x_p14)
MoxWSTe2x_pre14 = pd.DataFrame(MoxWSTe2x_p14)
MoxWSeTe2x_pre14 = pd.DataFrame(MoxWSeTe2x_p14)
MoS2xSe2xTe_pre14 = pd.DataFrame(MoS2xSe2xTe_p14)
MoS2xSeTe2x_pre14 = pd.DataFrame(MoS2xSeTe2x_p14)
MoSSe2xTe2x_pre14 = pd.DataFrame(MoSSe2xTe2x_p14)
WS2xSe2xTe_pre14 = pd.DataFrame(WS2xSe2xTe_p14)
WS2xSeTe2x_pre14 = pd.DataFrame(WS2xSeTe2x_p14)
WSSe2xTe2x_pre14 = pd.DataFrame(WSSe2xTe2x_p14)

MoWxS2xSe2xTe_pre14 = pd.DataFrame(MoWxS2xSe2xTe_p14)
MoWxS2xSeTe2x_pre14 = pd.DataFrame(MoWxS2xSeTe2x_p14)
MoWxSSe2xTe2x_pre14 = pd.DataFrame(MoWxSSe2xTe2x_p14)
MoxWS2xSe2xTe_pre14 = pd.DataFrame(MoxWS2xSe2xTe_p14)
MoxWS2xSeTe2x_pre14 = pd.DataFrame(MoxWS2xSeTe2x_p14)
MoxWSSe2xTe2x_pre14 = pd.DataFrame(MoxWSSe2xTe2x_p14)

# 15
train_dataset15 = pd.concat([dataset2, dataset3, dataset5])
validat_dataset15 = dataset1
test_dataset15 = dataset4

train_features15 = train_dataset15.copy()
validat_features15 = validat_dataset15.copy()
test_features15 = test_dataset15.copy()

train_labels15 = train_features15.pop('bandgap')
validat_labels15 = validat_features15.pop('bandgap')
test_labels15 = test_features15.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_15 = MyModel()

model_15.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_15 = model_15.fit(train_features15, train_labels15,
                    validation_data=(validat_features15, validat_labels15), 
                    epochs=3000, verbose=0)

test_results_15 = model_15.evaluate(test_features15, test_labels15, verbose=0)

model_15.summary()

MoWS_predictions15 = model_15.predict(MoWS_features).flatten()
MoWSe_predictions15 = model_15.predict(MoWSe_features).flatten()
MoWTe_predictions15 = model_15.predict(MoWTe_features).flatten()
MoSSe_predictions15 = model_15.predict(MoSSe_features).flatten()
MoSTe_predictions15 = model_15.predict(MoSTe_features).flatten()
MoSeTe_predictions15 = model_15.predict(MoSeTe_features).flatten()
WSSe_predictions15 = model_15.predict(WSSe_features).flatten()
WSTe_predictions15 = model_15.predict(WSTe_features).flatten()
WSeTe_predictions15 = model_15.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions15 = model_15.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions15 = model_15.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions15 = model_15.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions15 = model_15.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions15 = model_15.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions15 = model_15.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions15 = model_15.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions15 = model_15.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions15 = model_15.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions15 = model_15.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions15 = model_15.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions15 = model_15.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions15 = model_15.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions15 = model_15.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions15 = model_15.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions15 = model_15.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions15 = model_15.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions15 = model_15.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p15 = {}
MoWSe_p15 = {}
MoWTe_p15 = {}
MoSSe_p15 = {}
MoSTe_p15 = {}
MoSeTe_p15 = {}
WSSe_p15 = {}
WSTe_p15 = {}
WSeTe_p15 = {}

MoxWS2xSe_p15 = {}
MoxWS2xTe_p15 = {}
MoxWSe2xTe_p15 = {}
MoxWSSe2x_p15 = {}
MoxWSTe2x_p15 = {}
MoxWSeTe2x_p15 = {}
MoS2xSe2xTe_p15 = {}
MoS2xSeTe2x_p15 = {}
MoSSe2xTe2x_p15 = {}
WS2xSe2xTe_p15 = {}
WS2xSeTe2x_p15 = {}
WSSe2xTe2x_p15 = {}

MoWxS2xSe2xTe_p15 = {}
MoWxS2xSeTe2x_p15 = {}
MoWxSSe2xTe2x_p15 = {}
MoxWS2xSe2xTe_p15 = {}
MoxWS2xSeTe2x_p15 = {}
MoxWSSe2xTe2x_p15 = {}

MoWS_p15['gap15'] = MoWS_predictions15
MoWSe_p15['gap15'] = MoWSe_predictions15
MoWTe_p15['gap15'] = MoWTe_predictions15
MoSSe_p15['gap15'] = MoSSe_predictions15
MoSTe_p15['gap15'] = MoSTe_predictions15
MoSeTe_p15['gap15'] = MoSeTe_predictions15
WSSe_p15['gap15'] = WSSe_predictions15
WSTe_p15['gap15'] = WSTe_predictions15
WSeTe_p15['gap15'] = WSeTe_predictions15

MoxWS2xSe_p15['gap15'] = MoxWS2xSe_predictions15
MoxWS2xTe_p15['gap15'] = MoxWS2xTe_predictions15
MoxWSe2xTe_p15['gap15'] = MoxWSe2xTe_predictions15
MoxWSSe2x_p15['gap15'] = MoxWSSe2x_predictions15
MoxWSTe2x_p15['gap15'] = MoxWSTe2x_predictions15
MoxWSeTe2x_p15['gap15'] = MoxWSeTe2x_predictions15
MoS2xSe2xTe_p15['gap15'] = MoS2xSe2xTe_predictions15
MoS2xSeTe2x_p15['gap15'] = MoS2xSeTe2x_predictions15
MoSSe2xTe2x_p15['gap15'] = MoSSe2xTe2x_predictions15
WS2xSe2xTe_p15['gap15'] = WS2xSe2xTe_predictions15
WS2xSeTe2x_p15['gap15'] = WS2xSeTe2x_predictions15
WSSe2xTe2x_p15['gap15'] = WSSe2xTe2x_predictions15

MoWxS2xSe2xTe_p15['gap15'] = MoWxS2xSe2xTe_predictions15
MoWxS2xSeTe2x_p15['gap15'] = MoWxS2xSeTe2x_predictions15
MoWxSSe2xTe2x_p15['gap15'] = MoWxSSe2xTe2x_predictions15
MoxWS2xSe2xTe_p15['gap15'] = MoxWS2xSe2xTe_predictions15
MoxWS2xSeTe2x_p15['gap15'] = MoxWS2xSeTe2x_predictions15
MoxWSSe2xTe2x_p15['gap15'] = MoxWSSe2xTe2x_predictions15

MoWS_pre15 = pd.DataFrame(MoWS_p15)
MoWSe_pre15 = pd.DataFrame(MoWSe_p15)
MoWTe_pre15 = pd.DataFrame(MoWTe_p15)
MoSSe_pre15 = pd.DataFrame(MoSSe_p15)
MoSTe_pre15 = pd.DataFrame(MoSTe_p15)
MoSeTe_pre15 = pd.DataFrame(MoSeTe_p15)
WSSe_pre15 = pd.DataFrame(WSSe_p15)
WSTe_pre15 = pd.DataFrame(WSTe_p15)
WSeTe_pre15 = pd.DataFrame(WSeTe_p15)

MoxWS2xSe_pre15 = pd.DataFrame(MoxWS2xSe_p15)
MoxWS2xTe_pre15 = pd.DataFrame(MoxWS2xTe_p15)
MoxWSe2xTe_pre15 = pd.DataFrame(MoxWSe2xTe_p15)
MoxWSSe2x_pre15 = pd.DataFrame(MoxWSSe2x_p15)
MoxWSTe2x_pre15 = pd.DataFrame(MoxWSTe2x_p15)
MoxWSeTe2x_pre15 = pd.DataFrame(MoxWSeTe2x_p15)
MoS2xSe2xTe_pre15 = pd.DataFrame(MoS2xSe2xTe_p15)
MoS2xSeTe2x_pre15 = pd.DataFrame(MoS2xSeTe2x_p15)
MoSSe2xTe2x_pre15 = pd.DataFrame(MoSSe2xTe2x_p15)
WS2xSe2xTe_pre15 = pd.DataFrame(WS2xSe2xTe_p15)
WS2xSeTe2x_pre15 = pd.DataFrame(WS2xSeTe2x_p15)
WSSe2xTe2x_pre15 = pd.DataFrame(WSSe2xTe2x_p15)

MoWxS2xSe2xTe_pre15 = pd.DataFrame(MoWxS2xSe2xTe_p15)
MoWxS2xSeTe2x_pre15 = pd.DataFrame(MoWxS2xSeTe2x_p15)
MoWxSSe2xTe2x_pre15 = pd.DataFrame(MoWxSSe2xTe2x_p15)
MoxWS2xSe2xTe_pre15 = pd.DataFrame(MoxWS2xSe2xTe_p15)
MoxWS2xSeTe2x_pre15 = pd.DataFrame(MoxWS2xSeTe2x_p15)
MoxWSSe2xTe2x_pre15 = pd.DataFrame(MoxWSSe2xTe2x_p15)

# 16
train_dataset16 = pd.concat([dataset2, dataset3, dataset5])
validat_dataset16 = dataset4
test_dataset16 = dataset1

train_features16 = train_dataset16.copy()
validat_features16 = validat_dataset16.copy()
test_features16 = test_dataset16.copy()

train_labels16 = train_features16.pop('bandgap')
validat_labels16 = validat_features16.pop('bandgap')
test_labels16 = test_features16.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_16 = MyModel()

model_16.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_16 = model_16.fit(train_features16, train_labels16,
                    validation_data=(validat_features16, validat_labels16), 
                    epochs=3000, verbose=0)

test_results_16 = model_16.evaluate(test_features16, test_labels16, verbose=0)

model_16.summary()

MoWS_predictions16 = model_16.predict(MoWS_features).flatten()
MoWSe_predictions16 = model_16.predict(MoWSe_features).flatten()
MoWTe_predictions16 = model_16.predict(MoWTe_features).flatten()
MoSSe_predictions16 = model_16.predict(MoSSe_features).flatten()
MoSTe_predictions16 = model_16.predict(MoSTe_features).flatten()
MoSeTe_predictions16 = model_16.predict(MoSeTe_features).flatten()
WSSe_predictions16 = model_16.predict(WSSe_features).flatten()
WSTe_predictions16 = model_16.predict(WSTe_features).flatten()
WSeTe_predictions16 = model_16.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions16 = model_16.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions16 = model_16.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions16 = model_16.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions16 = model_16.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions16 = model_16.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions16 = model_16.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions16 = model_16.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions16 = model_16.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions16 = model_16.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions16 = model_16.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions16 = model_16.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions16 = model_16.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions16 = model_16.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions16 = model_16.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions16 = model_16.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions16 = model_16.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions16 = model_16.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions16 = model_16.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p16 = {}
MoWSe_p16 = {}
MoWTe_p16 = {}
MoSSe_p16 = {}
MoSTe_p16 = {}
MoSeTe_p16 = {}
WSSe_p16 = {}
WSTe_p16 = {}
WSeTe_p16 = {}

MoxWS2xSe_p16 = {}
MoxWS2xTe_p16 = {}
MoxWSe2xTe_p16 = {}
MoxWSSe2x_p16 = {}
MoxWSTe2x_p16 = {}
MoxWSeTe2x_p16 = {}
MoS2xSe2xTe_p16 = {}
MoS2xSeTe2x_p16 = {}
MoSSe2xTe2x_p16 = {}
WS2xSe2xTe_p16 = {}
WS2xSeTe2x_p16 = {}
WSSe2xTe2x_p16 = {}

MoWxS2xSe2xTe_p16 = {}
MoWxS2xSeTe2x_p16 = {}
MoWxSSe2xTe2x_p16 = {}
MoxWS2xSe2xTe_p16 = {}
MoxWS2xSeTe2x_p16 = {}
MoxWSSe2xTe2x_p16 = {}

MoWS_p16['gap16'] = MoWS_predictions16
MoWSe_p16['gap16'] = MoWSe_predictions16
MoWTe_p16['gap16'] = MoWTe_predictions16
MoSSe_p16['gap16'] = MoSSe_predictions16
MoSTe_p16['gap16'] = MoSTe_predictions16
MoSeTe_p16['gap16'] = MoSeTe_predictions16
WSSe_p16['gap16'] = WSSe_predictions16
WSTe_p16['gap16'] = WSTe_predictions16
WSeTe_p16['gap16'] = WSeTe_predictions16

MoxWS2xSe_p16['gap16'] = MoxWS2xSe_predictions16
MoxWS2xTe_p16['gap16'] = MoxWS2xTe_predictions16
MoxWSe2xTe_p16['gap16'] = MoxWSe2xTe_predictions16
MoxWSSe2x_p16['gap16'] = MoxWSSe2x_predictions16
MoxWSTe2x_p16['gap16'] = MoxWSTe2x_predictions16
MoxWSeTe2x_p16['gap16'] = MoxWSeTe2x_predictions16
MoS2xSe2xTe_p16['gap16'] = MoS2xSe2xTe_predictions16
MoS2xSeTe2x_p16['gap16'] = MoS2xSeTe2x_predictions16
MoSSe2xTe2x_p16['gap16'] = MoSSe2xTe2x_predictions16
WS2xSe2xTe_p16['gap16'] = WS2xSe2xTe_predictions16
WS2xSeTe2x_p16['gap16'] = WS2xSeTe2x_predictions16
WSSe2xTe2x_p16['gap16'] = WSSe2xTe2x_predictions16

MoWxS2xSe2xTe_p16['gap16'] = MoWxS2xSe2xTe_predictions16
MoWxS2xSeTe2x_p16['gap16'] = MoWxS2xSeTe2x_predictions16
MoWxSSe2xTe2x_p16['gap16'] = MoWxSSe2xTe2x_predictions16
MoxWS2xSe2xTe_p16['gap16'] = MoxWS2xSe2xTe_predictions16
MoxWS2xSeTe2x_p16['gap16'] = MoxWS2xSeTe2x_predictions16
MoxWSSe2xTe2x_p16['gap16'] = MoxWSSe2xTe2x_predictions16

MoWS_pre16 = pd.DataFrame(MoWS_p16)
MoWSe_pre16 = pd.DataFrame(MoWSe_p16)
MoWTe_pre16 = pd.DataFrame(MoWTe_p16)
MoSSe_pre16 = pd.DataFrame(MoSSe_p16)
MoSTe_pre16 = pd.DataFrame(MoSTe_p16)
MoSeTe_pre16 = pd.DataFrame(MoSeTe_p16)
WSSe_pre16 = pd.DataFrame(WSSe_p16)
WSTe_pre16 = pd.DataFrame(WSTe_p16)
WSeTe_pre16 = pd.DataFrame(WSeTe_p16)

MoxWS2xSe_pre16 = pd.DataFrame(MoxWS2xSe_p16)
MoxWS2xTe_pre16 = pd.DataFrame(MoxWS2xTe_p16)
MoxWSe2xTe_pre16 = pd.DataFrame(MoxWSe2xTe_p16)
MoxWSSe2x_pre16 = pd.DataFrame(MoxWSSe2x_p16)
MoxWSTe2x_pre16 = pd.DataFrame(MoxWSTe2x_p16)
MoxWSeTe2x_pre16 = pd.DataFrame(MoxWSeTe2x_p16)
MoS2xSe2xTe_pre16 = pd.DataFrame(MoS2xSe2xTe_p16)
MoS2xSeTe2x_pre16 = pd.DataFrame(MoS2xSeTe2x_p16)
MoSSe2xTe2x_pre16 = pd.DataFrame(MoSSe2xTe2x_p16)
WS2xSe2xTe_pre16 = pd.DataFrame(WS2xSe2xTe_p16)
WS2xSeTe2x_pre16 = pd.DataFrame(WS2xSeTe2x_p16)
WSSe2xTe2x_pre16 = pd.DataFrame(WSSe2xTe2x_p16)

MoWxS2xSe2xTe_pre16 = pd.DataFrame(MoWxS2xSe2xTe_p16)
MoWxS2xSeTe2x_pre16 = pd.DataFrame(MoWxS2xSeTe2x_p16)
MoWxSSe2xTe2x_pre16 = pd.DataFrame(MoWxSSe2xTe2x_p16)
MoxWS2xSe2xTe_pre16 = pd.DataFrame(MoxWS2xSe2xTe_p16)
MoxWS2xSeTe2x_pre16 = pd.DataFrame(MoxWS2xSeTe2x_p16)
MoxWSSe2xTe2x_pre16 = pd.DataFrame(MoxWSSe2xTe2x_p16)

# 17
train_dataset17 = pd.concat([dataset2, dataset4, dataset5])
validat_dataset17 = dataset1
test_dataset17 = dataset3

train_features17 = train_dataset17.copy()
validat_features17 = validat_dataset17.copy()
test_features17 = test_dataset17.copy()

train_labels17 = train_features17.pop('bandgap')
validat_labels17 = validat_features17.pop('bandgap')
test_labels17 = test_features17.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_17 = MyModel()

model_17.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_17 = model_17.fit(train_features17, train_labels17,
                    validation_data=(validat_features17, validat_labels17), 
                    epochs=3000, verbose=0)

test_results_17 = model_17.evaluate(test_features17, test_labels17, verbose=0)

model_17.summary()

MoWS_predictions17 = model_17.predict(MoWS_features).flatten()
MoWSe_predictions17 = model_17.predict(MoWSe_features).flatten()
MoWTe_predictions17 = model_17.predict(MoWTe_features).flatten()
MoSSe_predictions17 = model_17.predict(MoSSe_features).flatten()
MoSTe_predictions17 = model_17.predict(MoSTe_features).flatten()
MoSeTe_predictions17 = model_17.predict(MoSeTe_features).flatten()
WSSe_predictions17 = model_17.predict(WSSe_features).flatten()
WSTe_predictions17 = model_17.predict(WSTe_features).flatten()
WSeTe_predictions17 = model_17.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions17 = model_17.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions17 = model_17.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions17 = model_17.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions17 = model_17.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions17 = model_17.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions17 = model_17.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions17 = model_17.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions17 = model_17.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions17 = model_17.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions17 = model_17.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions17 = model_17.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions17 = model_17.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions17 = model_17.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions17 = model_17.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions17 = model_17.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions17 = model_17.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions17 = model_17.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions17 = model_17.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p17 = {}
MoWSe_p17 = {}
MoWTe_p17 = {}
MoSSe_p17 = {}
MoSTe_p17 = {}
MoSeTe_p17 = {}
WSSe_p17 = {}
WSTe_p17 = {}
WSeTe_p17 = {}

MoxWS2xSe_p17 = {}
MoxWS2xTe_p17 = {}
MoxWSe2xTe_p17 = {}
MoxWSSe2x_p17 = {}
MoxWSTe2x_p17 = {}
MoxWSeTe2x_p17 = {}
MoS2xSe2xTe_p17 = {}
MoS2xSeTe2x_p17 = {}
MoSSe2xTe2x_p17 = {}
WS2xSe2xTe_p17 = {}
WS2xSeTe2x_p17 = {}
WSSe2xTe2x_p17 = {}

MoWxS2xSe2xTe_p17 = {}
MoWxS2xSeTe2x_p17 = {}
MoWxSSe2xTe2x_p17 = {}
MoxWS2xSe2xTe_p17 = {}
MoxWS2xSeTe2x_p17 = {}
MoxWSSe2xTe2x_p17 = {}

MoWS_p17['gap17'] = MoWS_predictions17
MoWSe_p17['gap17'] = MoWSe_predictions17
MoWTe_p17['gap17'] = MoWTe_predictions17
MoSSe_p17['gap17'] = MoSSe_predictions17
MoSTe_p17['gap17'] = MoSTe_predictions17
MoSeTe_p17['gap17'] = MoSeTe_predictions17
WSSe_p17['gap17'] = WSSe_predictions17
WSTe_p17['gap17'] = WSTe_predictions17
WSeTe_p17['gap17'] = WSeTe_predictions17

MoxWS2xSe_p17['gap17'] = MoxWS2xSe_predictions17
MoxWS2xTe_p17['gap17'] = MoxWS2xTe_predictions17
MoxWSe2xTe_p17['gap17'] = MoxWSe2xTe_predictions17
MoxWSSe2x_p17['gap17'] = MoxWSSe2x_predictions17
MoxWSTe2x_p17['gap17'] = MoxWSTe2x_predictions17
MoxWSeTe2x_p17['gap17'] = MoxWSeTe2x_predictions17
MoS2xSe2xTe_p17['gap17'] = MoS2xSe2xTe_predictions17
MoS2xSeTe2x_p17['gap17'] = MoS2xSeTe2x_predictions17
MoSSe2xTe2x_p17['gap17'] = MoSSe2xTe2x_predictions17
WS2xSe2xTe_p17['gap17'] = WS2xSe2xTe_predictions17
WS2xSeTe2x_p17['gap17'] = WS2xSeTe2x_predictions17
WSSe2xTe2x_p17['gap17'] = WSSe2xTe2x_predictions17

MoWxS2xSe2xTe_p17['gap17'] = MoWxS2xSe2xTe_predictions17
MoWxS2xSeTe2x_p17['gap17'] = MoWxS2xSeTe2x_predictions17
MoWxSSe2xTe2x_p17['gap17'] = MoWxSSe2xTe2x_predictions17
MoxWS2xSe2xTe_p17['gap17'] = MoxWS2xSe2xTe_predictions17
MoxWS2xSeTe2x_p17['gap17'] = MoxWS2xSeTe2x_predictions17
MoxWSSe2xTe2x_p17['gap17'] = MoxWSSe2xTe2x_predictions17

MoWS_pre17 = pd.DataFrame(MoWS_p17)
MoWSe_pre17 = pd.DataFrame(MoWSe_p17)
MoWTe_pre17 = pd.DataFrame(MoWTe_p17)
MoSSe_pre17 = pd.DataFrame(MoSSe_p17)
MoSTe_pre17 = pd.DataFrame(MoSTe_p17)
MoSeTe_pre17 = pd.DataFrame(MoSeTe_p17)
WSSe_pre17 = pd.DataFrame(WSSe_p17)
WSTe_pre17 = pd.DataFrame(WSTe_p17)
WSeTe_pre17 = pd.DataFrame(WSeTe_p17)

MoxWS2xSe_pre17 = pd.DataFrame(MoxWS2xSe_p17)
MoxWS2xTe_pre17 = pd.DataFrame(MoxWS2xTe_p17)
MoxWSe2xTe_pre17 = pd.DataFrame(MoxWSe2xTe_p17)
MoxWSSe2x_pre17 = pd.DataFrame(MoxWSSe2x_p17)
MoxWSTe2x_pre17 = pd.DataFrame(MoxWSTe2x_p17)
MoxWSeTe2x_pre17 = pd.DataFrame(MoxWSeTe2x_p17)
MoS2xSe2xTe_pre17 = pd.DataFrame(MoS2xSe2xTe_p17)
MoS2xSeTe2x_pre17 = pd.DataFrame(MoS2xSeTe2x_p17)
MoSSe2xTe2x_pre17 = pd.DataFrame(MoSSe2xTe2x_p17)
WS2xSe2xTe_pre17 = pd.DataFrame(WS2xSe2xTe_p17)
WS2xSeTe2x_pre17 = pd.DataFrame(WS2xSeTe2x_p17)
WSSe2xTe2x_pre17 = pd.DataFrame(WSSe2xTe2x_p17)

MoWxS2xSe2xTe_pre17 = pd.DataFrame(MoWxS2xSe2xTe_p17)
MoWxS2xSeTe2x_pre17 = pd.DataFrame(MoWxS2xSeTe2x_p17)
MoWxSSe2xTe2x_pre17 = pd.DataFrame(MoWxSSe2xTe2x_p17)
MoxWS2xSe2xTe_pre17 = pd.DataFrame(MoxWS2xSe2xTe_p17)
MoxWS2xSeTe2x_pre17 = pd.DataFrame(MoxWS2xSeTe2x_p17)
MoxWSSe2xTe2x_pre17 = pd.DataFrame(MoxWSSe2xTe2x_p17)

# 18
train_dataset18 = pd.concat([dataset2, dataset4, dataset5])
validat_dataset18 = dataset3
test_dataset18 = dataset1

train_features18 = train_dataset18.copy()
validat_features18 = validat_dataset18.copy()
test_features18 = test_dataset18.copy()

train_labels18 = train_features18.pop('bandgap')
validat_labels18 = validat_features18.pop('bandgap')
test_labels18 = test_features18.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_18 = MyModel()

model_18.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_18 = model_18.fit(train_features18, train_labels18,
                    validation_data=(validat_features18, validat_labels18), 
                    epochs=3000, verbose=0)

test_results_18 = model_18.evaluate(test_features18, test_labels18, verbose=0)

model_18.summary()

MoWS_predictions18 = model_18.predict(MoWS_features).flatten()
MoWSe_predictions18 = model_18.predict(MoWSe_features).flatten()
MoWTe_predictions18 = model_18.predict(MoWTe_features).flatten()
MoSSe_predictions18 = model_18.predict(MoSSe_features).flatten()
MoSTe_predictions18 = model_18.predict(MoSTe_features).flatten()
MoSeTe_predictions18 = model_18.predict(MoSeTe_features).flatten()
WSSe_predictions18 = model_18.predict(WSSe_features).flatten()
WSTe_predictions18 = model_18.predict(WSTe_features).flatten()
WSeTe_predictions18 = model_18.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions18 = model_18.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions18 = model_18.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions18 = model_18.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions18 = model_18.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions18 = model_18.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions18 = model_18.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions18 = model_18.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions18 = model_18.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions18 = model_18.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions18 = model_18.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions18 = model_18.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions18 = model_18.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions18 = model_18.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions18 = model_18.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions18 = model_18.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions18 = model_18.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions18 = model_18.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions18 = model_18.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p18 = {}
MoWSe_p18 = {}
MoWTe_p18 = {}
MoSSe_p18 = {}
MoSTe_p18 = {}
MoSeTe_p18 = {}
WSSe_p18 = {}
WSTe_p18 = {}
WSeTe_p18 = {}

MoxWS2xSe_p18 = {}
MoxWS2xTe_p18 = {}
MoxWSe2xTe_p18 = {}
MoxWSSe2x_p18 = {}
MoxWSTe2x_p18 = {}
MoxWSeTe2x_p18 = {}
MoS2xSe2xTe_p18 = {}
MoS2xSeTe2x_p18 = {}
MoSSe2xTe2x_p18 = {}
WS2xSe2xTe_p18 = {}
WS2xSeTe2x_p18 = {}
WSSe2xTe2x_p18 = {}

MoWxS2xSe2xTe_p18 = {}
MoWxS2xSeTe2x_p18 = {}
MoWxSSe2xTe2x_p18 = {}
MoxWS2xSe2xTe_p18 = {}
MoxWS2xSeTe2x_p18 = {}
MoxWSSe2xTe2x_p18 = {}

MoWS_p18['gap18'] = MoWS_predictions18
MoWSe_p18['gap18'] = MoWSe_predictions18
MoWTe_p18['gap18'] = MoWTe_predictions18
MoSSe_p18['gap18'] = MoSSe_predictions18
MoSTe_p18['gap18'] = MoSTe_predictions18
MoSeTe_p18['gap18'] = MoSeTe_predictions18
WSSe_p18['gap18'] = WSSe_predictions18
WSTe_p18['gap18'] = WSTe_predictions18
WSeTe_p18['gap18'] = WSeTe_predictions18

MoxWS2xSe_p18['gap18'] = MoxWS2xSe_predictions18
MoxWS2xTe_p18['gap18'] = MoxWS2xTe_predictions18
MoxWSe2xTe_p18['gap18'] = MoxWSe2xTe_predictions18
MoxWSSe2x_p18['gap18'] = MoxWSSe2x_predictions18
MoxWSTe2x_p18['gap18'] = MoxWSTe2x_predictions18
MoxWSeTe2x_p18['gap18'] = MoxWSeTe2x_predictions18
MoS2xSe2xTe_p18['gap18'] = MoS2xSe2xTe_predictions18
MoS2xSeTe2x_p18['gap18'] = MoS2xSeTe2x_predictions18
MoSSe2xTe2x_p18['gap18'] = MoSSe2xTe2x_predictions18
WS2xSe2xTe_p18['gap18'] = WS2xSe2xTe_predictions18
WS2xSeTe2x_p18['gap18'] = WS2xSeTe2x_predictions18
WSSe2xTe2x_p18['gap18'] = WSSe2xTe2x_predictions18

MoWxS2xSe2xTe_p18['gap18'] = MoWxS2xSe2xTe_predictions18
MoWxS2xSeTe2x_p18['gap18'] = MoWxS2xSeTe2x_predictions18
MoWxSSe2xTe2x_p18['gap18'] = MoWxSSe2xTe2x_predictions18
MoxWS2xSe2xTe_p18['gap18'] = MoxWS2xSe2xTe_predictions18
MoxWS2xSeTe2x_p18['gap18'] = MoxWS2xSeTe2x_predictions18
MoxWSSe2xTe2x_p18['gap18'] = MoxWSSe2xTe2x_predictions18

MoWS_pre18 = pd.DataFrame(MoWS_p18)
MoWSe_pre18 = pd.DataFrame(MoWSe_p18)
MoWTe_pre18 = pd.DataFrame(MoWTe_p18)
MoSSe_pre18 = pd.DataFrame(MoSSe_p18)
MoSTe_pre18 = pd.DataFrame(MoSTe_p18)
MoSeTe_pre18 = pd.DataFrame(MoSeTe_p18)
WSSe_pre18 = pd.DataFrame(WSSe_p18)
WSTe_pre18 = pd.DataFrame(WSTe_p18)
WSeTe_pre18 = pd.DataFrame(WSeTe_p18)

MoxWS2xSe_pre18 = pd.DataFrame(MoxWS2xSe_p18)
MoxWS2xTe_pre18 = pd.DataFrame(MoxWS2xTe_p18)
MoxWSe2xTe_pre18 = pd.DataFrame(MoxWSe2xTe_p18)
MoxWSSe2x_pre18 = pd.DataFrame(MoxWSSe2x_p18)
MoxWSTe2x_pre18 = pd.DataFrame(MoxWSTe2x_p18)
MoxWSeTe2x_pre18 = pd.DataFrame(MoxWSeTe2x_p18)
MoS2xSe2xTe_pre18 = pd.DataFrame(MoS2xSe2xTe_p18)
MoS2xSeTe2x_pre18 = pd.DataFrame(MoS2xSeTe2x_p18)
MoSSe2xTe2x_pre18 = pd.DataFrame(MoSSe2xTe2x_p18)
WS2xSe2xTe_pre18 = pd.DataFrame(WS2xSe2xTe_p18)
WS2xSeTe2x_pre18 = pd.DataFrame(WS2xSeTe2x_p18)
WSSe2xTe2x_pre18 = pd.DataFrame(WSSe2xTe2x_p18)

MoWxS2xSe2xTe_pre18 = pd.DataFrame(MoWxS2xSe2xTe_p18)
MoWxS2xSeTe2x_pre18 = pd.DataFrame(MoWxS2xSeTe2x_p18)
MoWxSSe2xTe2x_pre18 = pd.DataFrame(MoWxSSe2xTe2x_p18)
MoxWS2xSe2xTe_pre18 = pd.DataFrame(MoxWS2xSe2xTe_p18)
MoxWS2xSeTe2x_pre18 = pd.DataFrame(MoxWS2xSeTe2x_p18)
MoxWSSe2xTe2x_pre18 = pd.DataFrame(MoxWSSe2xTe2x_p18)

# 19
train_dataset19 = pd.concat([dataset3, dataset4, dataset5])
validat_dataset19 = dataset1
test_dataset19 = dataset2

train_features19 = train_dataset19.copy()
validat_features19 = validat_dataset19.copy()
test_features19 = test_dataset19.copy()

train_labels19 = train_features19.pop('bandgap')
validat_labels19 = validat_features19.pop('bandgap')
test_labels19 = test_features19.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_19 = MyModel()

model_19.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_19 = model_19.fit(train_features19, train_labels19,
                    validation_data=(validat_features19, validat_labels19), 
                    epochs=3000, verbose=0)

test_results_19 = model_19.evaluate(test_features19, test_labels19, verbose=0)

model_19.summary()

MoWS_predictions19 = model_19.predict(MoWS_features).flatten()
MoWSe_predictions19 = model_19.predict(MoWSe_features).flatten()
MoWTe_predictions19 = model_19.predict(MoWTe_features).flatten()
MoSSe_predictions19 = model_19.predict(MoSSe_features).flatten()
MoSTe_predictions19 = model_19.predict(MoSTe_features).flatten()
MoSeTe_predictions19 = model_19.predict(MoSeTe_features).flatten()
WSSe_predictions19 = model_19.predict(WSSe_features).flatten()
WSTe_predictions19 = model_19.predict(WSTe_features).flatten()
WSeTe_predictions19 = model_19.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions19 = model_19.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions19 = model_19.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions19 = model_19.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions19 = model_19.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions19 = model_19.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions19 = model_19.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions19 = model_19.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions19 = model_19.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions19 = model_19.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions19 = model_19.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions19 = model_19.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions19 = model_19.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions19 = model_19.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions19 = model_19.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions19 = model_19.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions19 = model_19.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions19 = model_19.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions19 = model_19.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p19 = {}
MoWSe_p19 = {}
MoWTe_p19 = {}
MoSSe_p19 = {}
MoSTe_p19 = {}
MoSeTe_p19 = {}
WSSe_p19 = {}
WSTe_p19 = {}
WSeTe_p19 = {}

MoxWS2xSe_p19 = {}
MoxWS2xTe_p19 = {}
MoxWSe2xTe_p19 = {}
MoxWSSe2x_p19 = {}
MoxWSTe2x_p19 = {}
MoxWSeTe2x_p19 = {}
MoS2xSe2xTe_p19 = {}
MoS2xSeTe2x_p19 = {}
MoSSe2xTe2x_p19 = {}
WS2xSe2xTe_p19 = {}
WS2xSeTe2x_p19 = {}
WSSe2xTe2x_p19 = {}

MoWxS2xSe2xTe_p19 = {}
MoWxS2xSeTe2x_p19 = {}
MoWxSSe2xTe2x_p19 = {}
MoxWS2xSe2xTe_p19 = {}
MoxWS2xSeTe2x_p19 = {}
MoxWSSe2xTe2x_p19 = {}

MoWS_p19['gap19'] = MoWS_predictions19
MoWSe_p19['gap19'] = MoWSe_predictions19
MoWTe_p19['gap19'] = MoWTe_predictions19
MoSSe_p19['gap19'] = MoSSe_predictions19
MoSTe_p19['gap19'] = MoSTe_predictions19
MoSeTe_p19['gap19'] = MoSeTe_predictions19
WSSe_p19['gap19'] = WSSe_predictions19
WSTe_p19['gap19'] = WSTe_predictions19
WSeTe_p19['gap19'] = WSeTe_predictions19

MoxWS2xSe_p19['gap19'] = MoxWS2xSe_predictions19
MoxWS2xTe_p19['gap19'] = MoxWS2xTe_predictions19
MoxWSe2xTe_p19['gap19'] = MoxWSe2xTe_predictions19
MoxWSSe2x_p19['gap19'] = MoxWSSe2x_predictions19
MoxWSTe2x_p19['gap19'] = MoxWSTe2x_predictions19
MoxWSeTe2x_p19['gap19'] = MoxWSeTe2x_predictions19
MoS2xSe2xTe_p19['gap19'] = MoS2xSe2xTe_predictions19
MoS2xSeTe2x_p19['gap19'] = MoS2xSeTe2x_predictions19
MoSSe2xTe2x_p19['gap19'] = MoSSe2xTe2x_predictions19
WS2xSe2xTe_p19['gap19'] = WS2xSe2xTe_predictions19
WS2xSeTe2x_p19['gap19'] = WS2xSeTe2x_predictions19
WSSe2xTe2x_p19['gap19'] = WSSe2xTe2x_predictions19

MoWxS2xSe2xTe_p19['gap19'] = MoWxS2xSe2xTe_predictions19
MoWxS2xSeTe2x_p19['gap19'] = MoWxS2xSeTe2x_predictions19
MoWxSSe2xTe2x_p19['gap19'] = MoWxSSe2xTe2x_predictions19
MoxWS2xSe2xTe_p19['gap19'] = MoxWS2xSe2xTe_predictions19
MoxWS2xSeTe2x_p19['gap19'] = MoxWS2xSeTe2x_predictions19
MoxWSSe2xTe2x_p19['gap19'] = MoxWSSe2xTe2x_predictions19

MoWS_pre19 = pd.DataFrame(MoWS_p19)
MoWSe_pre19 = pd.DataFrame(MoWSe_p19)
MoWTe_pre19 = pd.DataFrame(MoWTe_p19)
MoSSe_pre19 = pd.DataFrame(MoSSe_p19)
MoSTe_pre19 = pd.DataFrame(MoSTe_p19)
MoSeTe_pre19 = pd.DataFrame(MoSeTe_p19)
WSSe_pre19 = pd.DataFrame(WSSe_p19)
WSTe_pre19 = pd.DataFrame(WSTe_p19)
WSeTe_pre19 = pd.DataFrame(WSeTe_p19)

MoxWS2xSe_pre19 = pd.DataFrame(MoxWS2xSe_p19)
MoxWS2xTe_pre19 = pd.DataFrame(MoxWS2xTe_p19)
MoxWSe2xTe_pre19 = pd.DataFrame(MoxWSe2xTe_p19)
MoxWSSe2x_pre19 = pd.DataFrame(MoxWSSe2x_p19)
MoxWSTe2x_pre19 = pd.DataFrame(MoxWSTe2x_p19)
MoxWSeTe2x_pre19 = pd.DataFrame(MoxWSeTe2x_p19)
MoS2xSe2xTe_pre19 = pd.DataFrame(MoS2xSe2xTe_p19)
MoS2xSeTe2x_pre19 = pd.DataFrame(MoS2xSeTe2x_p19)
MoSSe2xTe2x_pre19 = pd.DataFrame(MoSSe2xTe2x_p19)
WS2xSe2xTe_pre19 = pd.DataFrame(WS2xSe2xTe_p19)
WS2xSeTe2x_pre19 = pd.DataFrame(WS2xSeTe2x_p19)
WSSe2xTe2x_pre19 = pd.DataFrame(WSSe2xTe2x_p19)

MoWxS2xSe2xTe_pre19 = pd.DataFrame(MoWxS2xSe2xTe_p19)
MoWxS2xSeTe2x_pre19 = pd.DataFrame(MoWxS2xSeTe2x_p19)
MoWxSSe2xTe2x_pre19 = pd.DataFrame(MoWxSSe2xTe2x_p19)
MoxWS2xSe2xTe_pre19 = pd.DataFrame(MoxWS2xSe2xTe_p19)
MoxWS2xSeTe2x_pre19 = pd.DataFrame(MoxWS2xSeTe2x_p19)
MoxWSSe2xTe2x_pre19 = pd.DataFrame(MoxWSSe2xTe2x_p19)

# 20
train_dataset20 = pd.concat([dataset3, dataset4, dataset5])
validat_dataset20 = dataset2
test_dataset20 = dataset1

train_features20 = train_dataset20.copy()
validat_features20 = validat_dataset20.copy()
test_features20 = test_dataset20.copy()

train_labels20 = train_features20.pop('bandgap')
validat_labels20 = validat_features20.pop('bandgap')
test_labels20 = test_features20.pop('bandgap')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(8, activation='sigmoid')
    self.d2 = Dense(8, activation='sigmoid')
    # elu, gelu, relu, relu6, selu, sigmoid, silu, softplus, softsign, swish, tanh
    self.d3 = Dense(1)
    
  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

model_20 = MyModel()

model_20.compile(optimizer=optimizers.Adam(), loss='mse', metrics=['mae', 'mse'])
# Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

history_20 = model_20.fit(train_features20, train_labels20,
                    validation_data=(validat_features20, validat_labels20), 
                    epochs=3000, verbose=0)

test_results_20 = model_20.evaluate(test_features20, test_labels20, verbose=0)

model_20.summary()

MoWS_predictions20 = model_20.predict(MoWS_features).flatten()
MoWSe_predictions20 = model_20.predict(MoWSe_features).flatten()
MoWTe_predictions20 = model_20.predict(MoWTe_features).flatten()
MoSSe_predictions20 = model_20.predict(MoSSe_features).flatten()
MoSTe_predictions20 = model_20.predict(MoSTe_features).flatten()
MoSeTe_predictions20 = model_20.predict(MoSeTe_features).flatten()
WSSe_predictions20 = model_20.predict(WSSe_features).flatten()
WSTe_predictions20 = model_20.predict(WSTe_features).flatten()
WSeTe_predictions20 = model_20.predict(WSeTe_features).flatten()

MoxWS2xSe_predictions20 = model_20.predict(MoxWS2xSe_features).flatten()
MoxWS2xTe_predictions20 = model_20.predict(MoxWS2xTe_features).flatten()
MoxWSe2xTe_predictions20 = model_20.predict(MoxWSe2xTe_features).flatten()
MoxWSSe2x_predictions20 = model_20.predict(MoxWSSe2x_features).flatten()
MoxWSTe2x_predictions20 = model_20.predict(MoxWSTe2x_features).flatten()
MoxWSeTe2x_predictions20 = model_20.predict(MoxWSeTe2x_features).flatten()
MoS2xSe2xTe_predictions20 = model_20.predict(MoS2xSe2xTe_features).flatten()
MoS2xSeTe2x_predictions20 = model_20.predict(MoS2xSeTe2x_features).flatten()
MoSSe2xTe2x_predictions20 = model_20.predict(MoSSe2xTe2x_features).flatten()
WS2xSe2xTe_predictions20 = model_20.predict(WS2xSe2xTe_features).flatten()
WS2xSeTe2x_predictions20 = model_20.predict(WS2xSeTe2x_features).flatten()
WSSe2xTe2x_predictions20 = model_20.predict(WSSe2xTe2x_features).flatten()

MoWxS2xSe2xTe_predictions20 = model_20.predict(MoWxS2xSe2xTe_features).flatten()
MoWxS2xSeTe2x_predictions20 = model_20.predict(MoWxS2xSeTe2x_features).flatten()
MoWxSSe2xTe2x_predictions20 = model_20.predict(MoWxSSe2xTe2x_features).flatten()
MoxWS2xSe2xTe_predictions20 = model_20.predict(MoxWS2xSe2xTe_features).flatten()
MoxWS2xSeTe2x_predictions20 = model_20.predict(MoxWS2xSeTe2x_features).flatten()
MoxWSSe2xTe2x_predictions20 = model_20.predict(MoxWSSe2xTe2x_features).flatten()

MoWS_p20 = {}
MoWSe_p20 = {}
MoWTe_p20 = {}
MoSSe_p20 = {}
MoSTe_p20 = {}
MoSeTe_p20 = {}
WSSe_p20 = {}
WSTe_p20 = {}
WSeTe_p20 = {}

MoxWS2xSe_p20 = {}
MoxWS2xTe_p20 = {}
MoxWSe2xTe_p20 = {}
MoxWSSe2x_p20 = {}
MoxWSTe2x_p20 = {}
MoxWSeTe2x_p20 = {}
MoS2xSe2xTe_p20 = {}
MoS2xSeTe2x_p20 = {}
MoSSe2xTe2x_p20 = {}
WS2xSe2xTe_p20 = {}
WS2xSeTe2x_p20 = {}
WSSe2xTe2x_p20 = {}

MoWxS2xSe2xTe_p20 = {}
MoWxS2xSeTe2x_p20 = {}
MoWxSSe2xTe2x_p20 = {}
MoxWS2xSe2xTe_p20 = {}
MoxWS2xSeTe2x_p20 = {}
MoxWSSe2xTe2x_p20 = {}

MoWS_p20['gap20'] = MoWS_predictions20
MoWSe_p20['gap20'] = MoWSe_predictions20
MoWTe_p20['gap20'] = MoWTe_predictions20
MoSSe_p20['gap20'] = MoSSe_predictions20
MoSTe_p20['gap20'] = MoSTe_predictions20
MoSeTe_p20['gap20'] = MoSeTe_predictions20
WSSe_p20['gap20'] = WSSe_predictions20
WSTe_p20['gap20'] = WSTe_predictions20
WSeTe_p20['gap20'] = WSeTe_predictions20

MoxWS2xSe_p20['gap20'] = MoxWS2xSe_predictions20
MoxWS2xTe_p20['gap20'] = MoxWS2xTe_predictions20
MoxWSe2xTe_p20['gap20'] = MoxWSe2xTe_predictions20
MoxWSSe2x_p20['gap20'] = MoxWSSe2x_predictions20
MoxWSTe2x_p20['gap20'] = MoxWSTe2x_predictions20
MoxWSeTe2x_p20['gap20'] = MoxWSeTe2x_predictions20
MoS2xSe2xTe_p20['gap20'] = MoS2xSe2xTe_predictions20
MoS2xSeTe2x_p20['gap20'] = MoS2xSeTe2x_predictions20
MoSSe2xTe2x_p20['gap20'] = MoSSe2xTe2x_predictions20
WS2xSe2xTe_p20['gap20'] = WS2xSe2xTe_predictions20
WS2xSeTe2x_p20['gap20'] = WS2xSeTe2x_predictions20
WSSe2xTe2x_p20['gap20'] = WSSe2xTe2x_predictions20

MoWxS2xSe2xTe_p20['gap20'] = MoWxS2xSe2xTe_predictions20
MoWxS2xSeTe2x_p20['gap20'] = MoWxS2xSeTe2x_predictions20
MoWxSSe2xTe2x_p20['gap20'] = MoWxSSe2xTe2x_predictions20
MoxWS2xSe2xTe_p20['gap20'] = MoxWS2xSe2xTe_predictions20
MoxWS2xSeTe2x_p20['gap20'] = MoxWS2xSeTe2x_predictions20
MoxWSSe2xTe2x_p20['gap20'] = MoxWSSe2xTe2x_predictions20

MoWS_pre20 = pd.DataFrame(MoWS_p20)
MoWSe_pre20 = pd.DataFrame(MoWSe_p20)
MoWTe_pre20 = pd.DataFrame(MoWTe_p20)
MoSSe_pre20 = pd.DataFrame(MoSSe_p20)
MoSTe_pre20 = pd.DataFrame(MoSTe_p20)
MoSeTe_pre20 = pd.DataFrame(MoSeTe_p20)
WSSe_pre20 = pd.DataFrame(WSSe_p20)
WSTe_pre20 = pd.DataFrame(WSTe_p20)
WSeTe_pre20 = pd.DataFrame(WSeTe_p20)

MoxWS2xSe_pre20 = pd.DataFrame(MoxWS2xSe_p20)
MoxWS2xTe_pre20 = pd.DataFrame(MoxWS2xTe_p20)
MoxWSe2xTe_pre20 = pd.DataFrame(MoxWSe2xTe_p20)
MoxWSSe2x_pre20 = pd.DataFrame(MoxWSSe2x_p20)
MoxWSTe2x_pre20 = pd.DataFrame(MoxWSTe2x_p20)
MoxWSeTe2x_pre20 = pd.DataFrame(MoxWSeTe2x_p20)
MoS2xSe2xTe_pre20 = pd.DataFrame(MoS2xSe2xTe_p20)
MoS2xSeTe2x_pre20 = pd.DataFrame(MoS2xSeTe2x_p20)
MoSSe2xTe2x_pre20 = pd.DataFrame(MoSSe2xTe2x_p20)
WS2xSe2xTe_pre20 = pd.DataFrame(WS2xSe2xTe_p20)
WS2xSeTe2x_pre20 = pd.DataFrame(WS2xSeTe2x_p20)
WSSe2xTe2x_pre20 = pd.DataFrame(WSSe2xTe2x_p20)

MoWxS2xSe2xTe_pre20 = pd.DataFrame(MoWxS2xSe2xTe_p20)
MoWxS2xSeTe2x_pre20 = pd.DataFrame(MoWxS2xSeTe2x_p20)
MoWxSSe2xTe2x_pre20 = pd.DataFrame(MoWxSSe2xTe2x_p20)
MoxWS2xSe2xTe_pre20 = pd.DataFrame(MoxWS2xSe2xTe_p20)
MoxWS2xSeTe2x_pre20 = pd.DataFrame(MoxWS2xSeTe2x_p20)
MoxWSSe2xTe2x_pre20 = pd.DataFrame(MoxWSSe2xTe2x_p20)

# summary

MoWS_pre = pd.concat([MoWS_pre1, MoWS_pre2, MoWS_pre3, MoWS_pre4, MoWS_pre5, 
                      MoWS_pre6, MoWS_pre7, MoWS_pre8, MoWS_pre9, MoWS_pre10, 
                      MoWS_pre11, MoWS_pre12, MoWS_pre13, MoWS_pre14, MoWS_pre15, 
                      MoWS_pre16, MoWS_pre17, MoWS_pre18, MoWS_pre19, MoWS_pre20], axis=1)
MoWSe_pre = pd.concat([MoWSe_pre1, MoWSe_pre2, MoWSe_pre3, MoWSe_pre4, MoWSe_pre5, 
                      MoWSe_pre6, MoWSe_pre7, MoWSe_pre8, MoWSe_pre9, MoWSe_pre10, 
                      MoWSe_pre11, MoWSe_pre12, MoWSe_pre13, MoWSe_pre14, MoWSe_pre15, 
                      MoWSe_pre16, MoWSe_pre17, MoWSe_pre18, MoWSe_pre19, MoWSe_pre20], axis=1) 
MoWTe_pre = pd.concat([MoWTe_pre1, MoWTe_pre2, MoWTe_pre3, MoWTe_pre4, MoWTe_pre5, 
                      MoWTe_pre6, MoWTe_pre7, MoWTe_pre8, MoWTe_pre9, MoWTe_pre10, 
                      MoWTe_pre11, MoWTe_pre12, MoWTe_pre13, MoWTe_pre14, MoWTe_pre15, 
                      MoWTe_pre16, MoWTe_pre17, MoWTe_pre18, MoWTe_pre19, MoWTe_pre20], axis=1)
MoSSe_pre = pd.concat([MoSSe_pre1, MoSSe_pre2, MoSSe_pre3, MoSSe_pre4, MoSSe_pre5, 
                      MoSSe_pre6, MoSSe_pre7, MoSSe_pre8, MoSSe_pre9, MoSSe_pre10, 
                      MoSSe_pre11, MoSSe_pre12, MoSSe_pre13, MoSSe_pre14, MoSSe_pre15, 
                      MoSSe_pre16, MoSSe_pre17, MoSSe_pre18, MoSSe_pre19, MoSSe_pre20], axis=1)
MoSTe_pre = pd.concat([MoSTe_pre1, MoSTe_pre2, MoSTe_pre3, MoSTe_pre4, MoSTe_pre5, 
                      MoSTe_pre6, MoSTe_pre7, MoSTe_pre8, MoSTe_pre9, MoSTe_pre10, 
                      MoSTe_pre11, MoSTe_pre12, MoSTe_pre13, MoSTe_pre14, MoSTe_pre15, 
                      MoSTe_pre16, MoSTe_pre17, MoSTe_pre18, MoSTe_pre19, MoSTe_pre20], axis=1)
MoSeTe_pre = pd.concat([MoSeTe_pre1, MoSeTe_pre2, MoSeTe_pre3, MoSeTe_pre4, MoSeTe_pre5, 
                      MoSeTe_pre6, MoSeTe_pre7, MoSeTe_pre8, MoSeTe_pre9, MoSeTe_pre10, 
                      MoSeTe_pre11, MoSeTe_pre12, MoSeTe_pre13, MoSeTe_pre14, MoSeTe_pre15, 
                      MoSeTe_pre16, MoSeTe_pre17, MoSeTe_pre18, MoSeTe_pre19, MoSeTe_pre20], axis=1)
WSSe_pre = pd.concat([WSSe_pre1, WSSe_pre2, WSSe_pre3, WSSe_pre4, WSSe_pre5, 
                      WSSe_pre6, WSSe_pre7, WSSe_pre8, WSSe_pre9, WSSe_pre10, 
                      WSSe_pre11, WSSe_pre12, WSSe_pre13, WSSe_pre14, WSSe_pre15, 
                      WSSe_pre16, WSSe_pre17, WSSe_pre18, WSSe_pre19, WSSe_pre20], axis=1)
WSTe_pre = pd.concat([WSTe_pre1, WSTe_pre2, WSTe_pre3, WSTe_pre4, WSTe_pre5, 
                      WSTe_pre6, WSTe_pre7, WSTe_pre8, WSTe_pre9, WSTe_pre10, 
                      WSTe_pre11, WSTe_pre12, WSTe_pre13, WSTe_pre14, WSTe_pre15, 
                      WSTe_pre16, WSTe_pre17, WSTe_pre18, WSTe_pre19, WSTe_pre20], axis=1)
WSeTe_pre = pd.concat([WSeTe_pre1, WSeTe_pre2, WSeTe_pre3, WSeTe_pre4, WSeTe_pre5, 
                      WSeTe_pre6, WSeTe_pre7, WSeTe_pre8, WSeTe_pre9, WSeTe_pre10, 
                      WSeTe_pre11, WSeTe_pre12, WSeTe_pre13, WSeTe_pre14, WSeTe_pre15, 
                      WSeTe_pre16, WSeTe_pre17, WSeTe_pre18, WSeTe_pre19, WSeTe_pre20], axis=1)
MoxWS2xSe_pre = pd.concat([MoxWS2xSe_pre1, MoxWS2xSe_pre2, MoxWS2xSe_pre3, MoxWS2xSe_pre4, MoxWS2xSe_pre5, 
                      MoxWS2xSe_pre6, MoxWS2xSe_pre7, MoxWS2xSe_pre8, MoxWS2xSe_pre9, MoxWS2xSe_pre10, 
                      MoxWS2xSe_pre11, MoxWS2xSe_pre12, MoxWS2xSe_pre13, MoxWS2xSe_pre14, MoxWS2xSe_pre15, 
                      MoxWS2xSe_pre16, MoxWS2xSe_pre17, MoxWS2xSe_pre18, MoxWS2xSe_pre19, MoxWS2xSe_pre20], axis=1)
MoxWS2xTe_pre = pd.concat([MoxWS2xTe_pre1, MoxWS2xTe_pre2, MoxWS2xTe_pre3, MoxWS2xTe_pre4, MoxWS2xTe_pre5, 
                      MoxWS2xTe_pre6, MoxWS2xTe_pre7, MoxWS2xTe_pre8, MoxWS2xTe_pre9, MoxWS2xTe_pre10, 
                      MoxWS2xTe_pre11, MoxWS2xTe_pre12, MoxWS2xTe_pre13, MoxWS2xTe_pre14, MoxWS2xTe_pre15, 
                      MoxWS2xTe_pre16, MoxWS2xTe_pre17, MoxWS2xTe_pre18, MoxWS2xTe_pre19, MoxWS2xTe_pre20], axis=1)
MoxWSe2xTe_pre = pd.concat([MoxWSe2xTe_pre1, MoxWSe2xTe_pre2, MoxWSe2xTe_pre3, MoxWSe2xTe_pre4, MoxWSe2xTe_pre5, 
                      MoxWSe2xTe_pre6, MoxWSe2xTe_pre7, MoxWSe2xTe_pre8, MoxWSe2xTe_pre9, MoxWSe2xTe_pre10, 
                      MoxWSe2xTe_pre11, MoxWSe2xTe_pre12, MoxWSe2xTe_pre13, MoxWSe2xTe_pre14, MoxWSe2xTe_pre15, 
                      MoxWSe2xTe_pre16, MoxWSe2xTe_pre17, MoxWSe2xTe_pre18, MoxWSe2xTe_pre19, MoxWSe2xTe_pre20], axis=1)
MoxWSSe2x_pre = pd.concat([MoxWSSe2x_pre1, MoxWSSe2x_pre2, MoxWSSe2x_pre3, MoxWSSe2x_pre4, MoxWSSe2x_pre5, 
                      MoxWSSe2x_pre6, MoxWSSe2x_pre7, MoxWSSe2x_pre8, MoxWSSe2x_pre9, MoxWSSe2x_pre10, 
                      MoxWSSe2x_pre11, MoxWSSe2x_pre12, MoxWSSe2x_pre13, MoxWSSe2x_pre14, MoxWSSe2x_pre15, 
                      MoxWSSe2x_pre16, MoxWSSe2x_pre17, MoxWSSe2x_pre18, MoxWSSe2x_pre19, MoxWSSe2x_pre20], axis=1)
MoxWSTe2x_pre = pd.concat([MoxWSTe2x_pre1, MoxWSTe2x_pre2, MoxWSTe2x_pre3, MoxWSTe2x_pre4, MoxWSTe2x_pre5, 
                      MoxWSTe2x_pre6, MoxWSTe2x_pre7, MoxWSTe2x_pre8, MoxWSTe2x_pre9, MoxWSTe2x_pre10, 
                      MoxWSTe2x_pre11, MoxWSTe2x_pre12, MoxWSTe2x_pre13, MoxWSTe2x_pre14, MoxWSTe2x_pre15, 
                      MoxWSTe2x_pre16, MoxWSTe2x_pre17, MoxWSTe2x_pre18, MoxWSTe2x_pre19, MoxWSTe2x_pre20], axis=1)
MoxWSeTe2x_pre = pd.concat([MoxWSeTe2x_pre1, MoxWSeTe2x_pre2, MoxWSeTe2x_pre3, MoxWSeTe2x_pre4, MoxWSeTe2x_pre5, 
                      MoxWSeTe2x_pre6, MoxWSeTe2x_pre7, MoxWSeTe2x_pre8, MoxWSeTe2x_pre9, MoxWSeTe2x_pre10, 
                      MoxWSeTe2x_pre11, MoxWSeTe2x_pre12, MoxWSeTe2x_pre13, MoxWSeTe2x_pre14, MoxWSeTe2x_pre15, 
                      MoxWSeTe2x_pre16, MoxWSeTe2x_pre17, MoxWSeTe2x_pre18, MoxWSeTe2x_pre19, MoxWSeTe2x_pre20], axis=1)
MoS2xSe2xTe_pre = pd.concat([MoS2xSe2xTe_pre1, MoS2xSe2xTe_pre2, MoS2xSe2xTe_pre3, MoS2xSe2xTe_pre4, MoS2xSe2xTe_pre5, 
                      MoS2xSe2xTe_pre6, MoS2xSe2xTe_pre7, MoS2xSe2xTe_pre8, MoS2xSe2xTe_pre9, MoS2xSe2xTe_pre10, 
                      MoS2xSe2xTe_pre11, MoS2xSe2xTe_pre12, MoS2xSe2xTe_pre13, MoS2xSe2xTe_pre14, MoS2xSe2xTe_pre15, 
                      MoS2xSe2xTe_pre16, MoS2xSe2xTe_pre17, MoS2xSe2xTe_pre18, MoS2xSe2xTe_pre19, MoS2xSe2xTe_pre20], axis=1)
MoS2xSeTe2x_pre = pd.concat([MoS2xSeTe2x_pre1, MoS2xSeTe2x_pre2, MoS2xSeTe2x_pre3, MoS2xSeTe2x_pre4, MoS2xSeTe2x_pre5, 
                      MoS2xSeTe2x_pre6, MoS2xSeTe2x_pre7, MoS2xSeTe2x_pre8, MoS2xSeTe2x_pre9, MoS2xSeTe2x_pre10, 
                      MoS2xSeTe2x_pre11, MoS2xSeTe2x_pre12, MoS2xSeTe2x_pre13, MoS2xSeTe2x_pre14, MoS2xSeTe2x_pre15, 
                      MoS2xSeTe2x_pre16, MoS2xSeTe2x_pre17, MoS2xSeTe2x_pre18, MoS2xSeTe2x_pre19, MoS2xSeTe2x_pre20], axis=1)
MoSSe2xTe2x_pre = pd.concat([MoSSe2xTe2x_pre1, MoSSe2xTe2x_pre2, MoSSe2xTe2x_pre3, MoSSe2xTe2x_pre4, MoSSe2xTe2x_pre5, 
                      MoSSe2xTe2x_pre6, MoSSe2xTe2x_pre7, MoSSe2xTe2x_pre8, MoSSe2xTe2x_pre9, MoSSe2xTe2x_pre10, 
                      MoSSe2xTe2x_pre11, MoSSe2xTe2x_pre12, MoSSe2xTe2x_pre13, MoSSe2xTe2x_pre14, MoSSe2xTe2x_pre15, 
                      MoSSe2xTe2x_pre16, MoSSe2xTe2x_pre17, MoSSe2xTe2x_pre18, MoSSe2xTe2x_pre19, MoSSe2xTe2x_pre20], axis=1)
WS2xSe2xTe_pre = pd.concat([WS2xSe2xTe_pre1, WS2xSe2xTe_pre2, WS2xSe2xTe_pre3, WS2xSe2xTe_pre4, WS2xSe2xTe_pre5, 
                      WS2xSe2xTe_pre6, WS2xSe2xTe_pre7, WS2xSe2xTe_pre8, WS2xSe2xTe_pre9, WS2xSe2xTe_pre10, 
                      WS2xSe2xTe_pre11, WS2xSe2xTe_pre12, WS2xSe2xTe_pre13, WS2xSe2xTe_pre14, WS2xSe2xTe_pre15, 
                      WS2xSe2xTe_pre16, WS2xSe2xTe_pre17, WS2xSe2xTe_pre18, WS2xSe2xTe_pre19, WS2xSe2xTe_pre20], axis=1)
WS2xSeTe2x_pre = pd.concat([WS2xSeTe2x_pre1, WS2xSeTe2x_pre2, WS2xSeTe2x_pre3, WS2xSeTe2x_pre4, WS2xSeTe2x_pre5, 
                      WS2xSeTe2x_pre6, WS2xSeTe2x_pre7, WS2xSeTe2x_pre8, WS2xSeTe2x_pre9, WS2xSeTe2x_pre10, 
                      WS2xSeTe2x_pre11, WS2xSeTe2x_pre12, WS2xSeTe2x_pre13, WS2xSeTe2x_pre14, WS2xSeTe2x_pre15, 
                      WS2xSeTe2x_pre16, WS2xSeTe2x_pre17, WS2xSeTe2x_pre18, WS2xSeTe2x_pre19, WS2xSeTe2x_pre20], axis=1)
WSSe2xTe2x_pre = pd.concat([WSSe2xTe2x_pre1, WSSe2xTe2x_pre2, WSSe2xTe2x_pre3, WSSe2xTe2x_pre4, WSSe2xTe2x_pre5, 
                      WSSe2xTe2x_pre6, WSSe2xTe2x_pre7, WSSe2xTe2x_pre8, WSSe2xTe2x_pre9, WSSe2xTe2x_pre10, 
                      WSSe2xTe2x_pre11, WSSe2xTe2x_pre12, WSSe2xTe2x_pre13, WSSe2xTe2x_pre14, WSSe2xTe2x_pre15, 
                      WSSe2xTe2x_pre16, WSSe2xTe2x_pre17, WSSe2xTe2x_pre18, WSSe2xTe2x_pre19, WSSe2xTe2x_pre20], axis=1)
MoWxS2xSe2xTe_pre = pd.concat([MoWxS2xSe2xTe_pre1, MoWxS2xSe2xTe_pre2, MoWxS2xSe2xTe_pre3, MoWxS2xSe2xTe_pre4, MoWxS2xSe2xTe_pre5, 
                      MoWxS2xSe2xTe_pre6, MoWxS2xSe2xTe_pre7, MoWxS2xSe2xTe_pre8, MoWxS2xSe2xTe_pre9, MoWxS2xSe2xTe_pre10, 
                      MoWxS2xSe2xTe_pre11, MoWxS2xSe2xTe_pre12, MoWxS2xSe2xTe_pre13, MoWxS2xSe2xTe_pre14, MoWxS2xSe2xTe_pre15, 
                      MoWxS2xSe2xTe_pre16, MoWxS2xSe2xTe_pre17, MoWxS2xSe2xTe_pre18, MoWxS2xSe2xTe_pre19, MoWxS2xSe2xTe_pre20], axis=1)
MoWxS2xSeTe2x_pre = pd.concat([MoWxS2xSeTe2x_pre1, MoWxS2xSeTe2x_pre2, MoWxS2xSeTe2x_pre3, MoWxS2xSeTe2x_pre4, MoWxS2xSeTe2x_pre5, 
                      MoWxS2xSeTe2x_pre6, MoWxS2xSeTe2x_pre7, MoWxS2xSeTe2x_pre8, MoWxS2xSeTe2x_pre9, MoWxS2xSeTe2x_pre10, 
                      MoWxS2xSeTe2x_pre11, MoWxS2xSeTe2x_pre12, MoWxS2xSeTe2x_pre13, MoWxS2xSeTe2x_pre14, MoWxS2xSeTe2x_pre15, 
                      MoWxS2xSeTe2x_pre16, MoWxS2xSeTe2x_pre17, MoWxS2xSeTe2x_pre18, MoWxS2xSeTe2x_pre19, MoWxS2xSeTe2x_pre20], axis=1)
MoWxSSe2xTe2x_pre = pd.concat([MoWxSSe2xTe2x_pre1, MoWxSSe2xTe2x_pre2, MoWxSSe2xTe2x_pre3, MoWxSSe2xTe2x_pre4, MoWxSSe2xTe2x_pre5, 
                      MoWxSSe2xTe2x_pre6, MoWxSSe2xTe2x_pre7, MoWxSSe2xTe2x_pre8, MoWxSSe2xTe2x_pre9, MoWxSSe2xTe2x_pre10, 
                      MoWxSSe2xTe2x_pre11, MoWxSSe2xTe2x_pre12, MoWxSSe2xTe2x_pre13, MoWxSSe2xTe2x_pre14, MoWxSSe2xTe2x_pre15, 
                      MoWxSSe2xTe2x_pre16, MoWxSSe2xTe2x_pre17, MoWxSSe2xTe2x_pre18, MoWxSSe2xTe2x_pre19, MoWxSSe2xTe2x_pre20], axis=1)
MoxWS2xSe2xTe_pre = pd.concat([MoxWS2xSe2xTe_pre1, MoxWS2xSe2xTe_pre2, MoxWS2xSe2xTe_pre3, MoxWS2xSe2xTe_pre4, MoxWS2xSe2xTe_pre5, 
                      MoxWS2xSe2xTe_pre6, MoxWS2xSe2xTe_pre7, MoxWS2xSe2xTe_pre8, MoxWS2xSe2xTe_pre9, MoxWS2xSe2xTe_pre10, 
                      MoxWS2xSe2xTe_pre11, MoxWS2xSe2xTe_pre12, MoxWS2xSe2xTe_pre13, MoxWS2xSe2xTe_pre14, MoxWS2xSe2xTe_pre15, 
                      MoxWS2xSe2xTe_pre16, MoxWS2xSe2xTe_pre17, MoxWS2xSe2xTe_pre18, MoxWS2xSe2xTe_pre19, MoxWS2xSe2xTe_pre20], axis=1)
MoxWS2xSeTe2x_pre = pd.concat([MoxWS2xSeTe2x_pre1, MoxWS2xSeTe2x_pre2, MoxWS2xSeTe2x_pre3, MoxWS2xSeTe2x_pre4, MoxWS2xSeTe2x_pre5, 
                      MoxWS2xSeTe2x_pre6, MoxWS2xSeTe2x_pre7, MoxWS2xSeTe2x_pre8, MoxWS2xSeTe2x_pre9, MoxWS2xSeTe2x_pre10, 
                      MoxWS2xSeTe2x_pre11, MoxWS2xSeTe2x_pre12, MoxWS2xSeTe2x_pre13, MoxWS2xSeTe2x_pre14, MoxWS2xSeTe2x_pre15, 
                      MoxWS2xSeTe2x_pre16, MoxWS2xSeTe2x_pre17, MoxWS2xSeTe2x_pre18, MoxWS2xSeTe2x_pre19, MoxWS2xSeTe2x_pre20], axis=1)
MoxWSSe2xTe2x_pre = pd.concat([MoxWSSe2xTe2x_pre1, MoxWSSe2xTe2x_pre2, MoxWSSe2xTe2x_pre3, MoxWSSe2xTe2x_pre4, MoxWSSe2xTe2x_pre5, 
                      MoxWSSe2xTe2x_pre6, MoxWSSe2xTe2x_pre7, MoxWSSe2xTe2x_pre8, MoxWSSe2xTe2x_pre9, MoxWSSe2xTe2x_pre10, 
                      MoxWSSe2xTe2x_pre11, MoxWSSe2xTe2x_pre12, MoxWSSe2xTe2x_pre13, MoxWSSe2xTe2x_pre14, MoxWSSe2xTe2x_pre15, 
                      MoxWSSe2xTe2x_pre16, MoxWSSe2xTe2x_pre17, MoxWSSe2xTe2x_pre18, MoxWSSe2xTe2x_pre19, MoxWSSe2xTe2x_pre20], axis=1)
           
MoWS_pre.to_csv('MoWS_gap.csv')
MoWS_gapm = pd.concat([MoWS_pre.mean(axis=1), MoWS_pre.std(axis=1),], axis=1)
MoWS_gapm.to_csv('MoWS_gapm.csv')

MoWSe_pre.to_csv('MoWSe_gap.csv')
MoWSe_gapm = pd.concat([MoWSe_pre.mean(axis=1), MoWSe_pre.std(axis=1),], axis=1)
MoWSe_gapm.to_csv('MoWSe_gapm.csv')

MoWTe_pre.to_csv('MoWTe_gap.csv')
MoWTe_gapm = pd.concat([MoWTe_pre.mean(axis=1), MoWTe_pre.std(axis=1),], axis=1)
MoWTe_gapm.to_csv('MoWTe_gapm.csv')

MoSSe_pre.to_csv('MoSSe_gap.csv')
MoSSe_gapm = pd.concat([MoSSe_pre.mean(axis=1), MoSSe_pre.std(axis=1),], axis=1)
MoSSe_gapm.to_csv('MoSSe_gapm.csv')

MoSTe_pre.to_csv('MoSTe_gap.csv')
MoSTe_gapm = pd.concat([MoSTe_pre.mean(axis=1), MoSTe_pre.std(axis=1),], axis=1)
MoSTe_gapm.to_csv('MoSTe_gapm.csv')

MoSeTe_pre.to_csv('MoSeTe_gap.csv')
MoSeTe_gapm = pd.concat([MoSeTe_pre.mean(axis=1), MoSeTe_pre.std(axis=1),], axis=1)
MoSeTe_gapm.to_csv('MoSeTe_gapm.csv')

WSSe_pre.to_csv('WSSe_gap.csv')
WSSe_gapm = pd.concat([WSSe_pre.mean(axis=1), WSSe_pre.std(axis=1),], axis=1)
WSSe_gapm.to_csv('WSSe_gapm.csv')

WSTe_pre.to_csv('WSTe_gap.csv')
WSTe_gapm = pd.concat([WSTe_pre.mean(axis=1), WSTe_pre.std(axis=1),], axis=1)
WSTe_gapm.to_csv('WSTe_gapm.csv')

WSeTe_pre.to_csv('WSeTe_gap.csv')
WSeTe_gapm = pd.concat([WSeTe_pre.mean(axis=1), WSeTe_pre.std(axis=1),], axis=1)
WSeTe_gapm.to_csv('WSeTe_gapm.csv')

MoxWS2xSe_pre.to_csv('MoxWS2xSe_gap.csv')
MoxWS2xSe_gapm = pd.concat([MoxWS2xSe_pre.mean(axis=1), MoxWS2xSe_pre.std(axis=1),], axis=1)
MoxWS2xSe_gapm.to_csv('MoxWS2xSe_gapm.csv')

MoxWS2xTe_pre.to_csv('MoxWS2xTe_gap.csv')
MoxWS2xTe_gapm = pd.concat([MoxWS2xTe_pre.mean(axis=1), MoxWS2xTe_pre.std(axis=1),], axis=1)
MoxWS2xTe_gapm.to_csv('MoxWS2xTe_gapm.csv')

MoxWSe2xTe_pre.to_csv('MoxWSe2xTe_gap.csv')
MoxWSe2xTe_gapm = pd.concat([MoxWSe2xTe_pre.mean(axis=1), MoxWSe2xTe_pre.std(axis=1),], axis=1)
MoxWSe2xTe_gapm.to_csv('MoxWSe2xTe_gapm.csv')

MoxWSSe2x_pre.to_csv('MoxWSSe2x_gap.csv')
MoxWSSe2x_gapm = pd.concat([MoxWSSe2x_pre.mean(axis=1), MoxWSSe2x_pre.std(axis=1),], axis=1)
MoxWSSe2x_gapm.to_csv('MoxWSSe2x_gapm.csv')

MoxWSTe2x_pre.to_csv('MoxWSTe2x_gap.csv')
MoxWSTe2x_gapm = pd.concat([MoxWSTe2x_pre.mean(axis=1), MoxWSTe2x_pre.std(axis=1),], axis=1)
MoxWSTe2x_gapm.to_csv('MoxWSTe2x_gapm.csv')

MoxWSeTe2x_pre.to_csv('MoxWSeTe2x_gap.csv')
MoxWSeTe2x_gapm = pd.concat([MoxWSeTe2x_pre.mean(axis=1), MoxWSeTe2x_pre.std(axis=1),], axis=1)
MoxWSeTe2x_gapm.to_csv('MoxWSeTe2x_gapm.csv')

MoS2xSe2xTe_pre.to_csv('MoS2xSe2xTe_gap.csv')
MoS2xSe2xTe_gapm = pd.concat([MoS2xSe2xTe_pre.mean(axis=1), MoS2xSe2xTe_pre.std(axis=1),], axis=1)
MoS2xSe2xTe_gapm.to_csv('MoS2xSe2xTe_gapm.csv')

MoS2xSeTe2x_pre.to_csv('MoS2xSeTe2x_gap.csv')
MoS2xSeTe2x_gapm = pd.concat([MoS2xSeTe2x_pre.mean(axis=1), MoS2xSeTe2x_pre.std(axis=1),], axis=1)
MoS2xSeTe2x_gapm.to_csv('MoS2xSeTe2x_gapm.csv')

MoSSe2xTe2x_pre.to_csv('MoSSe2xTe2x_gap.csv')
MoSSe2xTe2x_gapm = pd.concat([MoSSe2xTe2x_pre.mean(axis=1), MoSSe2xTe2x_pre.std(axis=1),], axis=1)
MoSSe2xTe2x_gapm.to_csv('MoSSe2xTe2x_gapm.csv')

WS2xSe2xTe_pre.to_csv('WS2xSe2xTe_gap.csv')
WS2xSe2xTe_gapm = pd.concat([WS2xSe2xTe_pre.mean(axis=1), WS2xSe2xTe_pre.std(axis=1),], axis=1)
WS2xSe2xTe_gapm.to_csv('WS2xSe2xTe_gapm.csv')

WS2xSeTe2x_pre.to_csv('WS2xSeTe2x_gap.csv')
WS2xSeTe2x_gapm = pd.concat([WS2xSeTe2x_pre.mean(axis=1), WS2xSeTe2x_pre.std(axis=1),], axis=1)
WS2xSeTe2x_gapm.to_csv('WS2xSeTe2x_gapm.csv')

WSSe2xTe2x_pre.to_csv('WSSe2xTe2x_gap.csv')
WSSe2xTe2x_gapm = pd.concat([WSSe2xTe2x_pre.mean(axis=1), WSSe2xTe2x_pre.std(axis=1),], axis=1)
WSSe2xTe2x_gapm.to_csv('WSSe2xTe2x_gapm.csv')

MoWxS2xSe2xTe_pre.to_csv('MoWxS2xSe2xTe_gap.csv')
MoWxS2xSe2xTe_gapm = pd.concat([MoWxS2xSe2xTe_pre.mean(axis=1), MoWxS2xSe2xTe_pre.std(axis=1),], axis=1)
MoWxS2xSe2xTe_gapm.to_csv('MoWxS2xSe2xTe_gapm.csv')

MoWxS2xSeTe2x_pre.to_csv('MoWxS2xSeTe2x_gap.csv')
MoWxS2xSeTe2x_gapm = pd.concat([MoWxS2xSeTe2x_pre.mean(axis=1), MoWxS2xSeTe2x_pre.std(axis=1),], axis=1)
MoWxS2xSeTe2x_gapm.to_csv('MoWxS2xSeTe2x_gapm.csv')

MoWxSSe2xTe2x_pre.to_csv('MoWxSSe2xTe2x_gap.csv')
MoWxSSe2xTe2x_gapm = pd.concat([MoWxSSe2xTe2x_pre.mean(axis=1), MoWxSSe2xTe2x_pre.std(axis=1),], axis=1)
MoWxSSe2xTe2x_gapm.to_csv('MoWxSSe2xTe2x_gapm.csv')

MoxWS2xSe2xTe_pre.to_csv('MoxWS2xSe2xTe_gap.csv')
MoxWS2xSe2xTe_gapm = pd.concat([MoxWS2xSe2xTe_pre.mean(axis=1), MoxWS2xSe2xTe_pre.std(axis=1),], axis=1)
MoxWS2xSe2xTe_gapm.to_csv('MoxWS2xSe2xTe_gapm.csv')

MoxWS2xSeTe2x_pre.to_csv('MoxWS2xSeTe2x_gap.csv')
MoxWS2xSeTe2x_gapm = pd.concat([MoxWS2xSeTe2x_pre.mean(axis=1), MoxWS2xSeTe2x_pre.std(axis=1),], axis=1)
MoxWS2xSeTe2x_gapm.to_csv('MoxWS2xSeTe2x_gapm.csv')

MoxWSSe2xTe2x_pre.to_csv('MoxWSSe2xTe2x_gap.csv')
MoxWSSe2xTe2x_gapm = pd.concat([MoxWSSe2xTe2x_pre.mean(axis=1), MoxWSSe2xTe2x_pre.std(axis=1),], axis=1)
MoxWSSe2xTe2x_gapm.to_csv('MoxWSSe2xTe2x_gapm.csv')


