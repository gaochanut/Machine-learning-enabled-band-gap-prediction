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

history1 = pd.DataFrame(history_1.history)
history1.to_csv('history1.csv')

hist_1 = pd.DataFrame(history_1.history).tail(1)
hist_1.to_csv('fits_1.csv')
test_1 = pd.DataFrame(test_results_1).T
test_1.to_csv('eval_1.csv')

train_predictions1 = model_1.predict(train_features1).flatten()
validat_predictions1 = model_1.predict(validat_features1).flatten()
test_predictions1 = model_1.predict(test_features1).flatten()

plt.figure()
plt.plot(history_1.history['loss'], label='tra_loss')
plt.plot(history_1.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss1.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels1, train_predictions1)
plt.scatter(validat_labels1, validat_predictions1)
plt.scatter(test_labels1, test_predictions1)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity1.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error1 = train_predictions1 - train_labels1
validat_error1 = validat_predictions1 - validat_labels1
test_error1 = test_predictions1 - test_labels1
plt.figure()
plt.hist(train_error1, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError1.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error1, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError1.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error1, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError1.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_1 = {}
validation_1 = {}
test_1 = {}
train_1['train_labels'] = train_labels1
train_1['train_predictions'] = train_predictions1
validation_1['validat_labels'] = validat_labels1
validation_1['validat_predictions'] = validat_predictions1
test_1['test_labels'] = test_labels1
test_1['test_predictions'] = test_predictions1
train_pre1 = pd.DataFrame(train_1)
valid_pre1 = pd.DataFrame(validation_1)
testt_pre1 = pd.DataFrame(test_1)
train_pre1.to_csv('trainpre1.csv')
valid_pre1.to_csv('validpre1.csv')
testt_pre1.to_csv('testtpre1.csv')

train_sum1 = {}
validat_sum1 = {}
test_sum1 = {}
train_sum1['tot'] = (train_pre1['train_labels']-train_pre1['train_labels'].mean())**2
train_sum1['res'] = (train_pre1['train_predictions']-train_pre1['train_labels'])**2
train_s1 = pd.DataFrame(train_sum1)
train_r21 = 1.0-train_s1['res'].sum() / train_s1['tot'].sum()

validat_sum1['tot'] = (valid_pre1['validat_labels']-valid_pre1['validat_labels'].mean())**2
validat_sum1['res'] = (valid_pre1['validat_predictions']-valid_pre1['validat_labels'])**2
validat_s1 = pd.DataFrame(validat_sum1)
validat_r21 = 1.0-validat_s1['res'].sum() / validat_s1['tot'].sum()

test_sum1['tot'] = (testt_pre1['test_labels']-testt_pre1['test_labels'].mean())**2
test_sum1['res'] = (testt_pre1['test_predictions']-testt_pre1['test_labels'])**2
test_s1 = pd.DataFrame(test_sum1)
test_r21 = 1.0-test_s1['res'].sum() / test_s1['tot'].sum()

r21 = {}
r21['train'] = train_r21
r21['validation'] = validat_r21
r21['test'] = test_r21
rr21 = pd.DataFrame(r21, index=['0'])
rr21.to_csv('R21.csv')

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

history2 = pd.DataFrame(history_2.history)
history2.to_csv('history2.csv')

hist_2 = pd.DataFrame(history_2.history).tail(1)
hist_2.to_csv('fits_2.csv')
test_2 = pd.DataFrame(test_results_2).T
test_2.to_csv('eval_2.csv')

train_predictions2 = model_2.predict(train_features2).flatten()
validat_predictions2 = model_2.predict(validat_features2).flatten()
test_predictions2 = model_2.predict(test_features2).flatten()

plt.figure()
plt.plot(history_2.history['loss'], label='tra_loss')
plt.plot(history_2.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss2.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels2, train_predictions2)
plt.scatter(validat_labels2, validat_predictions2)
plt.scatter(test_labels2, test_predictions2)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity2.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error2 = train_predictions2 - train_labels2
validat_error2 = validat_predictions2 - validat_labels2
test_error2 = test_predictions2 - test_labels2
plt.figure()
plt.hist(train_error2, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError2.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error2, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError2.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error2, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError2.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_2 = {}
validation_2 = {}
test_2 = {}
train_2['train_labels'] = train_labels2
train_2['train_predictions'] = train_predictions2
validation_2['validat_labels'] = validat_labels2
validation_2['validat_predictions'] = validat_predictions2
test_2['test_labels'] = test_labels2
test_2['test_predictions'] = test_predictions2
train_pre2 = pd.DataFrame(train_2)
valid_pre2 = pd.DataFrame(validation_2)
testt_pre2 = pd.DataFrame(test_2)
train_pre2.to_csv('trainpre2.csv')
valid_pre2.to_csv('validpre2.csv')
testt_pre2.to_csv('testtpre2.csv')

train_sum2 = {}
validat_sum2 = {}
test_sum2 = {}
train_sum2['tot'] = (train_pre2['train_labels']-train_pre2['train_labels'].mean())**2
train_sum2['res'] = (train_pre2['train_predictions']-train_pre2['train_labels'])**2
train_s2 = pd.DataFrame(train_sum2)
train_r22 = 1.0-train_s2['res'].sum() / train_s2['tot'].sum()

validat_sum2['tot'] = (valid_pre2['validat_labels']-valid_pre2['validat_labels'].mean())**2
validat_sum2['res'] = (valid_pre2['validat_predictions']-valid_pre2['validat_labels'])**2
validat_s2 = pd.DataFrame(validat_sum2)
validat_r22 = 1.0-validat_s2['res'].sum() / validat_s2['tot'].sum()

test_sum2['tot'] = (testt_pre2['test_labels']-testt_pre2['test_labels'].mean())**2
test_sum2['res'] = (testt_pre2['test_predictions']-testt_pre2['test_labels'])**2
test_s2 = pd.DataFrame(test_sum2)
test_r22 = 1.0-test_s2['res'].sum() / test_s2['tot'].sum()

r22 = {}
r22['train'] = train_r22
r22['validation'] = validat_r22
r22['test'] = test_r22
rr22 = pd.DataFrame(r22, index=['0'])
rr22.to_csv('R22.csv')

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

history3 = pd.DataFrame(history_3.history)
history3.to_csv('history3.csv')

hist_3 = pd.DataFrame(history_3.history).tail(1)
hist_3.to_csv('fits_3.csv')
test_3 = pd.DataFrame(test_results_3).T
test_3.to_csv('eval_3.csv')

train_predictions3 = model_3.predict(train_features3).flatten()
validat_predictions3 = model_3.predict(validat_features3).flatten()
test_predictions3 = model_3.predict(test_features3).flatten()

plt.figure()
plt.plot(history_3.history['loss'], label='tra_loss')
plt.plot(history_3.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss3.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels3, train_predictions3)
plt.scatter(validat_labels3, validat_predictions3)
plt.scatter(test_labels3, test_predictions3)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity3.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error3 = train_predictions3 - train_labels3
validat_error3 = validat_predictions3 - validat_labels3
test_error3 = test_predictions3 - test_labels3
plt.figure()
plt.hist(train_error3, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError3.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error3, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError3.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error3, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError3.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_3 = {}
validation_3 = {}
test_3 = {}
train_3['train_labels'] = train_labels3
train_3['train_predictions'] = train_predictions3
validation_3['validat_labels'] = validat_labels3
validation_3['validat_predictions'] = validat_predictions3
test_3['test_labels'] = test_labels3
test_3['test_predictions'] = test_predictions3
train_pre3 = pd.DataFrame(train_3)
valid_pre3 = pd.DataFrame(validation_3)
testt_pre3 = pd.DataFrame(test_3)
train_pre3.to_csv('trainpre3.csv')
valid_pre3.to_csv('validpre3.csv')
testt_pre3.to_csv('testtpre3.csv')

train_sum3 = {}
validat_sum3 = {}
test_sum3 = {}
train_sum3['tot'] = (train_pre3['train_labels']-train_pre3['train_labels'].mean())**2
train_sum3['res'] = (train_pre3['train_predictions']-train_pre3['train_labels'])**2
train_s3 = pd.DataFrame(train_sum3)
train_r23 = 1.0-train_s3['res'].sum() / train_s3['tot'].sum()

validat_sum3['tot'] = (valid_pre3['validat_labels']-valid_pre3['validat_labels'].mean())**2
validat_sum3['res'] = (valid_pre3['validat_predictions']-valid_pre3['validat_labels'])**2
validat_s3 = pd.DataFrame(validat_sum3)
validat_r23 = 1.0-validat_s3['res'].sum() / validat_s3['tot'].sum()

test_sum3['tot'] = (testt_pre3['test_labels']-testt_pre3['test_labels'].mean())**2
test_sum3['res'] = (testt_pre3['test_predictions']-testt_pre3['test_labels'])**2
test_s3 = pd.DataFrame(test_sum3)
test_r23 = 1.0-test_s3['res'].sum() / test_s3['tot'].sum()

r23 = {}
r23['train'] = train_r23
r23['validation'] = validat_r23
r23['test'] = test_r23
rr23 = pd.DataFrame(r23, index=['0'])
rr23.to_csv('R23.csv')

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

history4 = pd.DataFrame(history_4.history)
history4.to_csv('history4.csv')

hist_4 = pd.DataFrame(history_4.history).tail(1)
hist_4.to_csv('fits_4.csv')
test_4 = pd.DataFrame(test_results_4).T
test_4.to_csv('eval_4.csv')

train_predictions4 = model_4.predict(train_features4).flatten()
validat_predictions4 = model_4.predict(validat_features4).flatten()
test_predictions4 = model_4.predict(test_features4).flatten()

plt.figure()
plt.plot(history_4.history['loss'], label='tra_loss')
plt.plot(history_4.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss4.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels4, train_predictions4)
plt.scatter(validat_labels4, validat_predictions4)
plt.scatter(test_labels4, test_predictions4)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity4.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error4 = train_predictions4 - train_labels4
validat_error4 = validat_predictions4 - validat_labels4
test_error4 = test_predictions4 - test_labels4
plt.figure()
plt.hist(train_error4, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError4.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error4, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError4.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error4, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError4.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_4 = {}
validation_4 = {}
test_4 = {}
train_4['train_labels'] = train_labels4
train_4['train_predictions'] = train_predictions4
validation_4['validat_labels'] = validat_labels4
validation_4['validat_predictions'] = validat_predictions4
test_4['test_labels'] = test_labels4
test_4['test_predictions'] = test_predictions4
train_pre4 = pd.DataFrame(train_4)
valid_pre4 = pd.DataFrame(validation_4)
testt_pre4 = pd.DataFrame(test_4)
train_pre4.to_csv('trainpre4.csv')
valid_pre4.to_csv('validpre4.csv')
testt_pre4.to_csv('testtpre4.csv')

train_sum4 = {}
validat_sum4 = {}
test_sum4 = {}
train_sum4['tot'] = (train_pre4['train_labels']-train_pre4['train_labels'].mean())**2
train_sum4['res'] = (train_pre4['train_predictions']-train_pre4['train_labels'])**2
train_s4 = pd.DataFrame(train_sum4)
train_r24 = 1.0-train_s4['res'].sum() / train_s4['tot'].sum()

validat_sum4['tot'] = (valid_pre4['validat_labels']-valid_pre4['validat_labels'].mean())**2
validat_sum4['res'] = (valid_pre4['validat_predictions']-valid_pre4['validat_labels'])**2
validat_s4 = pd.DataFrame(validat_sum4)
validat_r24 = 1.0-validat_s4['res'].sum() / validat_s4['tot'].sum()

test_sum4['tot'] = (testt_pre4['test_labels']-testt_pre4['test_labels'].mean())**2
test_sum4['res'] = (testt_pre4['test_predictions']-testt_pre4['test_labels'])**2
test_s4 = pd.DataFrame(test_sum4)
test_r24 = 1.0-test_s4['res'].sum() / test_s4['tot'].sum()

r24 = {}
r24['train'] = train_r24
r24['validation'] = validat_r24
r24['test'] = test_r24
rr24 = pd.DataFrame(r24, index=['0'])
rr24.to_csv('R24.csv')

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

history5 = pd.DataFrame(history_5.history)
history5.to_csv('history5.csv')

hist_5 = pd.DataFrame(history_5.history).tail(1)
hist_5.to_csv('fits_5.csv')
test_5 = pd.DataFrame(test_results_5).T
test_5.to_csv('eval_5.csv')

train_predictions5 = model_5.predict(train_features5).flatten()
validat_predictions5 = model_5.predict(validat_features5).flatten()
test_predictions5 = model_5.predict(test_features5).flatten()

plt.figure()
plt.plot(history_5.history['loss'], label='tra_loss')
plt.plot(history_5.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss5.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels5, train_predictions5)
plt.scatter(validat_labels5, validat_predictions5)
plt.scatter(test_labels5, test_predictions5)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity5.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error5 = train_predictions5 - train_labels5
validat_error5 = validat_predictions5 - validat_labels5
test_error5 = test_predictions5 - test_labels5
plt.figure()
plt.hist(train_error5, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError5.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error5, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError5.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error5, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError5.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_5 = {}
validation_5 = {}
test_5 = {}
train_5['train_labels'] = train_labels5
train_5['train_predictions'] = train_predictions5
validation_5['validat_labels'] = validat_labels5
validation_5['validat_predictions'] = validat_predictions5
test_5['test_labels'] = test_labels5
test_5['test_predictions'] = test_predictions5
train_pre5 = pd.DataFrame(train_5)
valid_pre5 = pd.DataFrame(validation_5)
testt_pre5 = pd.DataFrame(test_5)
train_pre5.to_csv('trainpre5.csv')
valid_pre5.to_csv('validpre5.csv')
testt_pre5.to_csv('testtpre5.csv')

train_sum5 = {}
validat_sum5 = {}
test_sum5 = {}
train_sum5['tot'] = (train_pre5['train_labels']-train_pre5['train_labels'].mean())**2
train_sum5['res'] = (train_pre5['train_predictions']-train_pre5['train_labels'])**2
train_s5 = pd.DataFrame(train_sum5)
train_r25 = 1.0-train_s5['res'].sum() / train_s5['tot'].sum()

validat_sum5['tot'] = (valid_pre5['validat_labels']-valid_pre5['validat_labels'].mean())**2
validat_sum5['res'] = (valid_pre5['validat_predictions']-valid_pre5['validat_labels'])**2
validat_s5 = pd.DataFrame(validat_sum5)
validat_r25 = 1.0-validat_s5['res'].sum() / validat_s5['tot'].sum()

test_sum5['tot'] = (testt_pre5['test_labels']-testt_pre5['test_labels'].mean())**2
test_sum5['res'] = (testt_pre5['test_predictions']-testt_pre5['test_labels'])**2
test_s5 = pd.DataFrame(test_sum5)
test_r25 = 1.0-test_s5['res'].sum() / test_s5['tot'].sum()

r25 = {}
r25['train'] = train_r25
r25['validation'] = validat_r25
r25['test'] = test_r25
rr25 = pd.DataFrame(r25, index=['0'])
rr25.to_csv('R25.csv')

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

history6 = pd.DataFrame(history_6.history)
history6.to_csv('history6.csv')

hist_6 = pd.DataFrame(history_6.history).tail(1)
hist_6.to_csv('fits_6.csv')
test_6 = pd.DataFrame(test_results_6).T
test_6.to_csv('eval_6.csv')

train_predictions6 = model_6.predict(train_features6).flatten()
validat_predictions6 = model_6.predict(validat_features6).flatten()
test_predictions6 = model_6.predict(test_features6).flatten()

plt.figure()
plt.plot(history_6.history['loss'], label='tra_loss')
plt.plot(history_6.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss6.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels6, train_predictions6)
plt.scatter(validat_labels6, validat_predictions6)
plt.scatter(test_labels6, test_predictions6)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity6.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error6 = train_predictions6 - train_labels6
validat_error6 = validat_predictions6 - validat_labels6
test_error6 = test_predictions6 - test_labels6
plt.figure()
plt.hist(train_error6, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError6.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error6, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError6.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error6, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError6.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_6 = {}
validation_6 = {}
test_6 = {}
train_6['train_labels'] = train_labels6
train_6['train_predictions'] = train_predictions6
validation_6['validat_labels'] = validat_labels6
validation_6['validat_predictions'] = validat_predictions6
test_6['test_labels'] = test_labels6
test_6['test_predictions'] = test_predictions6
train_pre6 = pd.DataFrame(train_6)
valid_pre6 = pd.DataFrame(validation_6)
testt_pre6 = pd.DataFrame(test_6)
train_pre6.to_csv('trainpre6.csv')
valid_pre6.to_csv('validpre6.csv')
testt_pre6.to_csv('testtpre6.csv')

train_sum6 = {}
validat_sum6 = {}
test_sum6 = {}
train_sum6['tot'] = (train_pre6['train_labels']-train_pre6['train_labels'].mean())**2
train_sum6['res'] = (train_pre6['train_predictions']-train_pre6['train_labels'])**2
train_s6 = pd.DataFrame(train_sum6)
train_r26 = 1.0-train_s6['res'].sum() / train_s6['tot'].sum()

validat_sum6['tot'] = (valid_pre6['validat_labels']-valid_pre6['validat_labels'].mean())**2
validat_sum6['res'] = (valid_pre6['validat_predictions']-valid_pre6['validat_labels'])**2
validat_s6 = pd.DataFrame(validat_sum6)
validat_r26 = 1.0-validat_s6['res'].sum() / validat_s6['tot'].sum()

test_sum6['tot'] = (testt_pre6['test_labels']-testt_pre6['test_labels'].mean())**2
test_sum6['res'] = (testt_pre6['test_predictions']-testt_pre6['test_labels'])**2
test_s6 = pd.DataFrame(test_sum6)
test_r26 = 1.0-test_s6['res'].sum() / test_s6['tot'].sum()

r26 = {}
r26['train'] = train_r26
r26['validation'] = validat_r26
r26['test'] = test_r26
rr26 = pd.DataFrame(r26, index=['0'])
rr26.to_csv('R26.csv')

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

history7 = pd.DataFrame(history_7.history)
history7.to_csv('history7.csv')

hist_7 = pd.DataFrame(history_7.history).tail(1)
hist_7.to_csv('fits_7.csv')
test_7 = pd.DataFrame(test_results_7).T
test_7.to_csv('eval_7.csv')

train_predictions7 = model_7.predict(train_features7).flatten()
validat_predictions7 = model_7.predict(validat_features7).flatten()
test_predictions7 = model_7.predict(test_features7).flatten()

plt.figure()
plt.plot(history_7.history['loss'], label='tra_loss')
plt.plot(history_7.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss7.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels7, train_predictions7)
plt.scatter(validat_labels7, validat_predictions7)
plt.scatter(test_labels7, test_predictions7)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity7.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error7 = train_predictions7 - train_labels7
validat_error7 = validat_predictions7 - validat_labels7
test_error7 = test_predictions7 - test_labels7
plt.figure()
plt.hist(train_error7, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError7.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error7, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError7.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error7, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError7.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_7 = {}
validation_7 = {}
test_7 = {}
train_7['train_labels'] = train_labels7
train_7['train_predictions'] = train_predictions7
validation_7['validat_labels'] = validat_labels7
validation_7['validat_predictions'] = validat_predictions7
test_7['test_labels'] = test_labels7
test_7['test_predictions'] = test_predictions7
train_pre7 = pd.DataFrame(train_7)
valid_pre7 = pd.DataFrame(validation_7)
testt_pre7 = pd.DataFrame(test_7)
train_pre7.to_csv('trainpre7.csv')
valid_pre7.to_csv('validpre7.csv')
testt_pre7.to_csv('testtpre7.csv')

train_sum7 = {}
validat_sum7 = {}
test_sum7 = {}
train_sum7['tot'] = (train_pre7['train_labels']-train_pre7['train_labels'].mean())**2
train_sum7['res'] = (train_pre7['train_predictions']-train_pre7['train_labels'])**2
train_s7 = pd.DataFrame(train_sum7)
train_r27 = 1.0-train_s7['res'].sum() / train_s7['tot'].sum()

validat_sum7['tot'] = (valid_pre7['validat_labels']-valid_pre7['validat_labels'].mean())**2
validat_sum7['res'] = (valid_pre7['validat_predictions']-valid_pre7['validat_labels'])**2
validat_s7 = pd.DataFrame(validat_sum7)
validat_r27 = 1.0-validat_s7['res'].sum() / validat_s7['tot'].sum()

test_sum7['tot'] = (testt_pre7['test_labels']-testt_pre7['test_labels'].mean())**2
test_sum7['res'] = (testt_pre7['test_predictions']-testt_pre7['test_labels'])**2
test_s7 = pd.DataFrame(test_sum7)
test_r27 = 1.0-test_s7['res'].sum() / test_s7['tot'].sum()

r27 = {}
r27['train'] = train_r27
r27['validation'] = validat_r27
r27['test'] = test_r27
rr27 = pd.DataFrame(r27, index=['0'])
rr27.to_csv('R27.csv')

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

history8 = pd.DataFrame(history_8.history)
history8.to_csv('history8.csv')

hist_8 = pd.DataFrame(history_8.history).tail(1)
hist_8.to_csv('fits_8.csv')
test_8 = pd.DataFrame(test_results_8).T
test_8.to_csv('eval_8.csv')

train_predictions8 = model_8.predict(train_features8).flatten()
validat_predictions8 = model_8.predict(validat_features8).flatten()
test_predictions8 = model_8.predict(test_features8).flatten()

plt.figure()
plt.plot(history_8.history['loss'], label='tra_loss')
plt.plot(history_8.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss8.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels8, train_predictions8)
plt.scatter(validat_labels8, validat_predictions8)
plt.scatter(test_labels8, test_predictions8)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity8.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error8 = train_predictions8 - train_labels8
validat_error8 = validat_predictions8 - validat_labels8
test_error8 = test_predictions8 - test_labels8
plt.figure()
plt.hist(train_error8, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError8.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error8, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError8.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error8, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError8.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_8 = {}
validation_8 = {}
test_8 = {}
train_8['train_labels'] = train_labels8
train_8['train_predictions'] = train_predictions8
validation_8['validat_labels'] = validat_labels8
validation_8['validat_predictions'] = validat_predictions8
test_8['test_labels'] = test_labels8
test_8['test_predictions'] = test_predictions8
train_pre8 = pd.DataFrame(train_8)
valid_pre8 = pd.DataFrame(validation_8)
testt_pre8 = pd.DataFrame(test_8)
train_pre8.to_csv('trainpre8.csv')
valid_pre8.to_csv('validpre8.csv')
testt_pre8.to_csv('testtpre8.csv')

train_sum8 = {}
validat_sum8 = {}
test_sum8 = {}
train_sum8['tot'] = (train_pre8['train_labels']-train_pre8['train_labels'].mean())**2
train_sum8['res'] = (train_pre8['train_predictions']-train_pre8['train_labels'])**2
train_s8 = pd.DataFrame(train_sum8)
train_r28 = 1.0-train_s8['res'].sum() / train_s8['tot'].sum()

validat_sum8['tot'] = (valid_pre8['validat_labels']-valid_pre8['validat_labels'].mean())**2
validat_sum8['res'] = (valid_pre8['validat_predictions']-valid_pre8['validat_labels'])**2
validat_s8 = pd.DataFrame(validat_sum8)
validat_r28 = 1.0-validat_s8['res'].sum() / validat_s8['tot'].sum()

test_sum8['tot'] = (testt_pre8['test_labels']-testt_pre8['test_labels'].mean())**2
test_sum8['res'] = (testt_pre8['test_predictions']-testt_pre8['test_labels'])**2
test_s8 = pd.DataFrame(test_sum8)
test_r28 = 1.0-test_s8['res'].sum() / test_s8['tot'].sum()

r28 = {}
r28['train'] = train_r28
r28['validation'] = validat_r28
r28['test'] = test_r28
rr28 = pd.DataFrame(r28, index=['0'])
rr28.to_csv('R28.csv')

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

history9 = pd.DataFrame(history_9.history)
history9.to_csv('history9.csv')

hist_9 = pd.DataFrame(history_9.history).tail(1)
hist_9.to_csv('fits_9.csv')
test_9 = pd.DataFrame(test_results_9).T
test_9.to_csv('eval_9.csv')

train_predictions9 = model_9.predict(train_features9).flatten()
validat_predictions9 = model_9.predict(validat_features9).flatten()
test_predictions9 = model_9.predict(test_features9).flatten()

plt.figure()
plt.plot(history_9.history['loss'], label='tra_loss')
plt.plot(history_9.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss9.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels9, train_predictions9)
plt.scatter(validat_labels9, validat_predictions9)
plt.scatter(test_labels9, test_predictions9)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity9.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error9 = train_predictions9 - train_labels9
validat_error9 = validat_predictions9 - validat_labels9
test_error9 = test_predictions9 - test_labels9
plt.figure()
plt.hist(train_error9, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError9.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error9, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError9.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error9, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError9.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_9 = {}
validation_9 = {}
test_9 = {}
train_9['train_labels'] = train_labels9
train_9['train_predictions'] = train_predictions9
validation_9['validat_labels'] = validat_labels9
validation_9['validat_predictions'] = validat_predictions9
test_9['test_labels'] = test_labels9
test_9['test_predictions'] = test_predictions9
train_pre9 = pd.DataFrame(train_9)
valid_pre9 = pd.DataFrame(validation_9)
testt_pre9 = pd.DataFrame(test_9)
train_pre9.to_csv('trainpre9.csv')
valid_pre9.to_csv('validpre9.csv')
testt_pre9.to_csv('testtpre9.csv')

train_sum9 = {}
validat_sum9 = {}
test_sum9 = {}
train_sum9['tot'] = (train_pre9['train_labels']-train_pre9['train_labels'].mean())**2
train_sum9['res'] = (train_pre9['train_predictions']-train_pre9['train_labels'])**2
train_s9 = pd.DataFrame(train_sum9)
train_r29 = 1.0-train_s9['res'].sum() / train_s9['tot'].sum()

validat_sum9['tot'] = (valid_pre9['validat_labels']-valid_pre9['validat_labels'].mean())**2
validat_sum9['res'] = (valid_pre9['validat_predictions']-valid_pre9['validat_labels'])**2
validat_s9 = pd.DataFrame(validat_sum9)
validat_r29 = 1.0-validat_s9['res'].sum() / validat_s9['tot'].sum()

test_sum9['tot'] = (testt_pre9['test_labels']-testt_pre9['test_labels'].mean())**2
test_sum9['res'] = (testt_pre9['test_predictions']-testt_pre9['test_labels'])**2
test_s9 = pd.DataFrame(test_sum9)
test_r29 = 1.0-test_s9['res'].sum() / test_s9['tot'].sum()

r29 = {}
r29['train'] = train_r29
r29['validation'] = validat_r29
r29['test'] = test_r29
rr29 = pd.DataFrame(r29, index=['0'])
rr29.to_csv('R29.csv')

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

history10 = pd.DataFrame(history_10.history)
history10.to_csv('history10.csv')

hist_10 = pd.DataFrame(history_10.history).tail(1)
hist_10.to_csv('fits_10.csv')
test_10 = pd.DataFrame(test_results_10).T
test_10.to_csv('eval_10.csv')

train_predictions10 = model_10.predict(train_features10).flatten()
validat_predictions10 = model_10.predict(validat_features10).flatten()
test_predictions10 = model_10.predict(test_features10).flatten()

plt.figure()
plt.plot(history_10.history['loss'], label='tra_loss')
plt.plot(history_10.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss10.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels10, train_predictions10)
plt.scatter(validat_labels10, validat_predictions10)
plt.scatter(test_labels10, test_predictions10)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity10.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error10 = train_predictions10 - train_labels10
validat_error10 = validat_predictions10 - validat_labels10
test_error10 = test_predictions10 - test_labels10
plt.figure()
plt.hist(train_error10, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError10.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error10, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError10.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error10, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError10.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_10 = {}
validation_10 = {}
test_10 = {}
train_10['train_labels'] = train_labels10
train_10['train_predictions'] = train_predictions10
validation_10['validat_labels'] = validat_labels10
validation_10['validat_predictions'] = validat_predictions10
test_10['test_labels'] = test_labels10
test_10['test_predictions'] = test_predictions10
train_pre10 = pd.DataFrame(train_10)
valid_pre10 = pd.DataFrame(validation_10)
testt_pre10 = pd.DataFrame(test_10)
train_pre10.to_csv('trainpre10.csv')
valid_pre10.to_csv('validpre10.csv')
testt_pre10.to_csv('testtpre10.csv')

train_sum10 = {}
validat_sum10 = {}
test_sum10 = {}
train_sum10['tot'] = (train_pre10['train_labels']-train_pre10['train_labels'].mean())**2
train_sum10['res'] = (train_pre10['train_predictions']-train_pre10['train_labels'])**2
train_s10 = pd.DataFrame(train_sum10)
train_r210 = 1.0-train_s10['res'].sum() / train_s10['tot'].sum()

validat_sum10['tot'] = (valid_pre10['validat_labels']-valid_pre10['validat_labels'].mean())**2
validat_sum10['res'] = (valid_pre10['validat_predictions']-valid_pre10['validat_labels'])**2
validat_s10 = pd.DataFrame(validat_sum10)
validat_r210 = 1.0-validat_s10['res'].sum() / validat_s10['tot'].sum()

test_sum10['tot'] = (testt_pre10['test_labels']-testt_pre10['test_labels'].mean())**2
test_sum10['res'] = (testt_pre10['test_predictions']-testt_pre10['test_labels'])**2
test_s10 = pd.DataFrame(test_sum10)
test_r210 = 1.0-test_s10['res'].sum() / test_s10['tot'].sum()

r210 = {}
r210['train'] = train_r210
r210['validation'] = validat_r210
r210['test'] = test_r210
rr210 = pd.DataFrame(r210, index=['0'])
rr210.to_csv('R210.csv')

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

history11 = pd.DataFrame(history_11.history)
history11.to_csv('history11.csv')

hist_11 = pd.DataFrame(history_11.history).tail(1)
hist_11.to_csv('fits_11.csv')
test_11 = pd.DataFrame(test_results_11).T
test_11.to_csv('eval_11.csv')

train_predictions11 = model_11.predict(train_features11).flatten()
validat_predictions11 = model_11.predict(validat_features11).flatten()
test_predictions11 = model_11.predict(test_features11).flatten()

plt.figure()
plt.plot(history_11.history['loss'], label='tra_loss')
plt.plot(history_11.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss11.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels11, train_predictions11)
plt.scatter(validat_labels11, validat_predictions11)
plt.scatter(test_labels11, test_predictions11)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity11.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error11 = train_predictions11 - train_labels11
validat_error11 = validat_predictions11 - validat_labels11
test_error11 = test_predictions11 - test_labels11
plt.figure()
plt.hist(train_error11, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError11.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error11, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError11.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error11, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError11.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_11 = {}
validation_11 = {}
test_11 = {}
train_11['train_labels'] = train_labels11
train_11['train_predictions'] = train_predictions11
validation_11['validat_labels'] = validat_labels11
validation_11['validat_predictions'] = validat_predictions11
test_11['test_labels'] = test_labels11
test_11['test_predictions'] = test_predictions11
train_pre11 = pd.DataFrame(train_11)
valid_pre11 = pd.DataFrame(validation_11)
testt_pre11 = pd.DataFrame(test_11)
train_pre11.to_csv('trainpre11.csv')
valid_pre11.to_csv('validpre11.csv')
testt_pre11.to_csv('testtpre11.csv')

train_sum11 = {}
validat_sum11 = {}
test_sum11 = {}
train_sum11['tot'] = (train_pre11['train_labels']-train_pre11['train_labels'].mean())**2
train_sum11['res'] = (train_pre11['train_predictions']-train_pre11['train_labels'])**2
train_s11 = pd.DataFrame(train_sum11)
train_r211 = 1.0-train_s11['res'].sum() / train_s11['tot'].sum()

validat_sum11['tot'] = (valid_pre11['validat_labels']-valid_pre11['validat_labels'].mean())**2
validat_sum11['res'] = (valid_pre11['validat_predictions']-valid_pre11['validat_labels'])**2
validat_s11 = pd.DataFrame(validat_sum11)
validat_r211 = 1.0-validat_s11['res'].sum() / validat_s11['tot'].sum()

test_sum11['tot'] = (testt_pre11['test_labels']-testt_pre11['test_labels'].mean())**2
test_sum11['res'] = (testt_pre11['test_predictions']-testt_pre11['test_labels'])**2
test_s11 = pd.DataFrame(test_sum11)
test_r211 = 1.0-test_s11['res'].sum() / test_s11['tot'].sum()

r211 = {}
r211['train'] = train_r211
r211['validation'] = validat_r211
r211['test'] = test_r211
rr211 = pd.DataFrame(r211, index=['0'])
rr211.to_csv('R211.csv')

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

history12 = pd.DataFrame(history_12.history)
history12.to_csv('history12.csv')

hist_12 = pd.DataFrame(history_12.history).tail(1)
hist_12.to_csv('fits_12.csv')
test_12 = pd.DataFrame(test_results_12).T
test_12.to_csv('eval_12.csv')

train_predictions12 = model_12.predict(train_features12).flatten()
validat_predictions12 = model_12.predict(validat_features12).flatten()
test_predictions12 = model_12.predict(test_features12).flatten()

plt.figure()
plt.plot(history_12.history['loss'], label='tra_loss')
plt.plot(history_12.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss12.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels12, train_predictions12)
plt.scatter(validat_labels12, validat_predictions12)
plt.scatter(test_labels12, test_predictions12)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity12.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error12 = train_predictions12 - train_labels12
validat_error12 = validat_predictions12 - validat_labels12
test_error12 = test_predictions12 - test_labels12
plt.figure()
plt.hist(train_error12, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError12.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error12, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError12.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error12, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError12.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_12 = {}
validation_12 = {}
test_12 = {}
train_12['train_labels'] = train_labels12
train_12['train_predictions'] = train_predictions12
validation_12['validat_labels'] = validat_labels12
validation_12['validat_predictions'] = validat_predictions12
test_12['test_labels'] = test_labels12
test_12['test_predictions'] = test_predictions12
train_pre12 = pd.DataFrame(train_12)
valid_pre12 = pd.DataFrame(validation_12)
testt_pre12 = pd.DataFrame(test_12)
train_pre12.to_csv('trainpre12.csv')
valid_pre12.to_csv('validpre12.csv')
testt_pre12.to_csv('testtpre12.csv')

train_sum12 = {}
validat_sum12 = {}
test_sum12 = {}
train_sum12['tot'] = (train_pre12['train_labels']-train_pre12['train_labels'].mean())**2
train_sum12['res'] = (train_pre12['train_predictions']-train_pre12['train_labels'])**2
train_s12 = pd.DataFrame(train_sum12)
train_r212 = 1.0-train_s12['res'].sum() / train_s12['tot'].sum()

validat_sum12['tot'] = (valid_pre12['validat_labels']-valid_pre12['validat_labels'].mean())**2
validat_sum12['res'] = (valid_pre12['validat_predictions']-valid_pre12['validat_labels'])**2
validat_s12 = pd.DataFrame(validat_sum12)
validat_r212 = 1.0-validat_s12['res'].sum() / validat_s12['tot'].sum()

test_sum12['tot'] = (testt_pre12['test_labels']-testt_pre12['test_labels'].mean())**2
test_sum12['res'] = (testt_pre12['test_predictions']-testt_pre12['test_labels'])**2
test_s12 = pd.DataFrame(test_sum12)
test_r212 = 1.0-test_s12['res'].sum() / test_s12['tot'].sum()

r212 = {}
r212['train'] = train_r212
r212['validation'] = validat_r212
r212['test'] = test_r212
rr212 = pd.DataFrame(r212, index=['0'])
rr212.to_csv('R212.csv')

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

history13 = pd.DataFrame(history_13.history)
history13.to_csv('history13.csv')

hist_13 = pd.DataFrame(history_13.history).tail(1)
hist_13.to_csv('fits_13.csv')
test_13 = pd.DataFrame(test_results_13).T
test_13.to_csv('eval_13.csv')

train_predictions13 = model_13.predict(train_features13).flatten()
validat_predictions13 = model_13.predict(validat_features13).flatten()
test_predictions13 = model_13.predict(test_features13).flatten()

plt.figure()
plt.plot(history_13.history['loss'], label='tra_loss')
plt.plot(history_13.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss13.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels13, train_predictions13)
plt.scatter(validat_labels13, validat_predictions13)
plt.scatter(test_labels13, test_predictions13)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity13.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error13 = train_predictions13 - train_labels13
validat_error13 = validat_predictions13 - validat_labels13
test_error13 = test_predictions13 - test_labels13
plt.figure()
plt.hist(train_error13, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError13.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error13, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError13.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error13, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError13.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_13 = {}
validation_13 = {}
test_13 = {}
train_13['train_labels'] = train_labels13
train_13['train_predictions'] = train_predictions13
validation_13['validat_labels'] = validat_labels13
validation_13['validat_predictions'] = validat_predictions13
test_13['test_labels'] = test_labels13
test_13['test_predictions'] = test_predictions13
train_pre13 = pd.DataFrame(train_13)
valid_pre13 = pd.DataFrame(validation_13)
testt_pre13 = pd.DataFrame(test_13)
train_pre13.to_csv('trainpre13.csv')
valid_pre13.to_csv('validpre13.csv')
testt_pre13.to_csv('testtpre13.csv')

train_sum13 = {}
validat_sum13 = {}
test_sum13 = {}
train_sum13['tot'] = (train_pre13['train_labels']-train_pre13['train_labels'].mean())**2
train_sum13['res'] = (train_pre13['train_predictions']-train_pre13['train_labels'])**2
train_s13 = pd.DataFrame(train_sum13)
train_r213 = 1.0-train_s13['res'].sum() / train_s13['tot'].sum()

validat_sum13['tot'] = (valid_pre13['validat_labels']-valid_pre13['validat_labels'].mean())**2
validat_sum13['res'] = (valid_pre13['validat_predictions']-valid_pre13['validat_labels'])**2
validat_s13 = pd.DataFrame(validat_sum13)
validat_r213 = 1.0-validat_s13['res'].sum() / validat_s13['tot'].sum()

test_sum13['tot'] = (testt_pre13['test_labels']-testt_pre13['test_labels'].mean())**2
test_sum13['res'] = (testt_pre13['test_predictions']-testt_pre13['test_labels'])**2
test_s13 = pd.DataFrame(test_sum13)
test_r213 = 1.0-test_s13['res'].sum() / test_s13['tot'].sum()

r213 = {}
r213['train'] = train_r213
r213['validation'] = validat_r213
r213['test'] = test_r213
rr213 = pd.DataFrame(r213, index=['0'])
rr213.to_csv('R213.csv')

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

history14 = pd.DataFrame(history_14.history)
history14.to_csv('history14.csv')

hist_14 = pd.DataFrame(history_14.history).tail(1)
hist_14.to_csv('fits_14.csv')
test_14 = pd.DataFrame(test_results_14).T
test_14.to_csv('eval_14.csv')

train_predictions14 = model_14.predict(train_features14).flatten()
validat_predictions14 = model_14.predict(validat_features14).flatten()
test_predictions14 = model_14.predict(test_features14).flatten()

plt.figure()
plt.plot(history_14.history['loss'], label='tra_loss')
plt.plot(history_14.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss14.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels14, train_predictions14)
plt.scatter(validat_labels14, validat_predictions14)
plt.scatter(test_labels14, test_predictions14)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity14.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error14 = train_predictions14 - train_labels14
validat_error14 = validat_predictions14 - validat_labels14
test_error14 = test_predictions14 - test_labels14
plt.figure()
plt.hist(train_error14, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError14.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error14, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError14.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error14, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError14.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_14 = {}
validation_14 = {}
test_14 = {}
train_14['train_labels'] = train_labels14
train_14['train_predictions'] = train_predictions14
validation_14['validat_labels'] = validat_labels14
validation_14['validat_predictions'] = validat_predictions14
test_14['test_labels'] = test_labels14
test_14['test_predictions'] = test_predictions14
train_pre14 = pd.DataFrame(train_14)
valid_pre14 = pd.DataFrame(validation_14)
testt_pre14 = pd.DataFrame(test_14)
train_pre14.to_csv('trainpre14.csv')
valid_pre14.to_csv('validpre14.csv')
testt_pre14.to_csv('testtpre14.csv')

train_sum14 = {}
validat_sum14 = {}
test_sum14 = {}
train_sum14['tot'] = (train_pre14['train_labels']-train_pre14['train_labels'].mean())**2
train_sum14['res'] = (train_pre14['train_predictions']-train_pre14['train_labels'])**2
train_s14 = pd.DataFrame(train_sum14)
train_r214 = 1.0-train_s14['res'].sum() / train_s14['tot'].sum()

validat_sum14['tot'] = (valid_pre14['validat_labels']-valid_pre14['validat_labels'].mean())**2
validat_sum14['res'] = (valid_pre14['validat_predictions']-valid_pre14['validat_labels'])**2
validat_s14 = pd.DataFrame(validat_sum14)
validat_r214 = 1.0-validat_s14['res'].sum() / validat_s14['tot'].sum()

test_sum14['tot'] = (testt_pre14['test_labels']-testt_pre14['test_labels'].mean())**2
test_sum14['res'] = (testt_pre14['test_predictions']-testt_pre14['test_labels'])**2
test_s14 = pd.DataFrame(test_sum14)
test_r214 = 1.0-test_s14['res'].sum() / test_s14['tot'].sum()

r214 = {}
r214['train'] = train_r214
r214['validation'] = validat_r214
r214['test'] = test_r214
rr214 = pd.DataFrame(r214, index=['0'])
rr214.to_csv('R214.csv')

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

history15 = pd.DataFrame(history_15.history)
history15.to_csv('history15.csv')

hist_15 = pd.DataFrame(history_15.history).tail(1)
hist_15.to_csv('fits_15.csv')
test_15 = pd.DataFrame(test_results_15).T
test_15.to_csv('eval_15.csv')

train_predictions15 = model_15.predict(train_features15).flatten()
validat_predictions15 = model_15.predict(validat_features15).flatten()
test_predictions15 = model_15.predict(test_features15).flatten()

plt.figure()
plt.plot(history_15.history['loss'], label='tra_loss')
plt.plot(history_15.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss15.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels15, train_predictions15)
plt.scatter(validat_labels15, validat_predictions15)
plt.scatter(test_labels15, test_predictions15)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity15.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error15 = train_predictions15 - train_labels15
validat_error15 = validat_predictions15 - validat_labels15
test_error15 = test_predictions15 - test_labels15
plt.figure()
plt.hist(train_error15, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError15.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error15, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError15.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error15, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError15.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_15 = {}
validation_15 = {}
test_15 = {}
train_15['train_labels'] = train_labels15
train_15['train_predictions'] = train_predictions15
validation_15['validat_labels'] = validat_labels15
validation_15['validat_predictions'] = validat_predictions15
test_15['test_labels'] = test_labels15
test_15['test_predictions'] = test_predictions15
train_pre15 = pd.DataFrame(train_15)
valid_pre15 = pd.DataFrame(validation_15)
testt_pre15 = pd.DataFrame(test_15)
train_pre15.to_csv('trainpre15.csv')
valid_pre15.to_csv('validpre15.csv')
testt_pre15.to_csv('testtpre15.csv')

train_sum15 = {}
validat_sum15 = {}
test_sum15 = {}
train_sum15['tot'] = (train_pre15['train_labels']-train_pre15['train_labels'].mean())**2
train_sum15['res'] = (train_pre15['train_predictions']-train_pre15['train_labels'])**2
train_s15 = pd.DataFrame(train_sum15)
train_r215 = 1.0-train_s15['res'].sum() / train_s15['tot'].sum()

validat_sum15['tot'] = (valid_pre15['validat_labels']-valid_pre15['validat_labels'].mean())**2
validat_sum15['res'] = (valid_pre15['validat_predictions']-valid_pre15['validat_labels'])**2
validat_s15 = pd.DataFrame(validat_sum15)
validat_r215 = 1.0-validat_s15['res'].sum() / validat_s15['tot'].sum()

test_sum15['tot'] = (testt_pre15['test_labels']-testt_pre15['test_labels'].mean())**2
test_sum15['res'] = (testt_pre15['test_predictions']-testt_pre15['test_labels'])**2
test_s15 = pd.DataFrame(test_sum15)
test_r215 = 1.0-test_s15['res'].sum() / test_s15['tot'].sum()

r215 = {}
r215['train'] = train_r215
r215['validation'] = validat_r215
r215['test'] = test_r215
rr215 = pd.DataFrame(r215, index=['0'])
rr215.to_csv('R215.csv')

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

history16 = pd.DataFrame(history_16.history)
history16.to_csv('history16.csv')

hist_16 = pd.DataFrame(history_16.history).tail(1)
hist_16.to_csv('fits_16.csv')
test_16 = pd.DataFrame(test_results_16).T
test_16.to_csv('eval_16.csv')

train_predictions16 = model_16.predict(train_features16).flatten()
validat_predictions16 = model_16.predict(validat_features16).flatten()
test_predictions16 = model_16.predict(test_features16).flatten()

plt.figure()
plt.plot(history_16.history['loss'], label='tra_loss')
plt.plot(history_16.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss16.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels16, train_predictions16)
plt.scatter(validat_labels16, validat_predictions16)
plt.scatter(test_labels16, test_predictions16)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity16.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error16 = train_predictions16 - train_labels16
validat_error16 = validat_predictions16 - validat_labels16
test_error16 = test_predictions16 - test_labels16
plt.figure()
plt.hist(train_error16, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError16.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error16, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError16.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error16, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError16.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_16 = {}
validation_16 = {}
test_16 = {}
train_16['train_labels'] = train_labels16
train_16['train_predictions'] = train_predictions16
validation_16['validat_labels'] = validat_labels16
validation_16['validat_predictions'] = validat_predictions16
test_16['test_labels'] = test_labels16
test_16['test_predictions'] = test_predictions16
train_pre16 = pd.DataFrame(train_16)
valid_pre16 = pd.DataFrame(validation_16)
testt_pre16 = pd.DataFrame(test_16)
train_pre16.to_csv('trainpre16.csv')
valid_pre16.to_csv('validpre16.csv')
testt_pre16.to_csv('testtpre16.csv')

train_sum16 = {}
validat_sum16 = {}
test_sum16 = {}
train_sum16['tot'] = (train_pre16['train_labels']-train_pre16['train_labels'].mean())**2
train_sum16['res'] = (train_pre16['train_predictions']-train_pre16['train_labels'])**2
train_s16 = pd.DataFrame(train_sum16)
train_r216 = 1.0-train_s16['res'].sum() / train_s16['tot'].sum()

validat_sum16['tot'] = (valid_pre16['validat_labels']-valid_pre16['validat_labels'].mean())**2
validat_sum16['res'] = (valid_pre16['validat_predictions']-valid_pre16['validat_labels'])**2
validat_s16 = pd.DataFrame(validat_sum16)
validat_r216 = 1.0-validat_s16['res'].sum() / validat_s16['tot'].sum()

test_sum16['tot'] = (testt_pre16['test_labels']-testt_pre16['test_labels'].mean())**2
test_sum16['res'] = (testt_pre16['test_predictions']-testt_pre16['test_labels'])**2
test_s16 = pd.DataFrame(test_sum16)
test_r216 = 1.0-test_s16['res'].sum() / test_s16['tot'].sum()

r216 = {}
r216['train'] = train_r216
r216['validation'] = validat_r216
r216['test'] = test_r216
rr216 = pd.DataFrame(r216, index=['0'])
rr216.to_csv('R216.csv')

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

history17 = pd.DataFrame(history_17.history)
history17.to_csv('history17.csv')

hist_17 = pd.DataFrame(history_17.history).tail(1)
hist_17.to_csv('fits_17.csv')
test_17 = pd.DataFrame(test_results_17).T
test_17.to_csv('eval_17.csv')

train_predictions17 = model_17.predict(train_features17).flatten()
validat_predictions17 = model_17.predict(validat_features17).flatten()
test_predictions17 = model_17.predict(test_features17).flatten()

plt.figure()
plt.plot(history_17.history['loss'], label='tra_loss')
plt.plot(history_17.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss17.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels17, train_predictions17)
plt.scatter(validat_labels17, validat_predictions17)
plt.scatter(test_labels17, test_predictions17)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity17.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error17 = train_predictions17 - train_labels17
validat_error17 = validat_predictions17 - validat_labels17
test_error17 = test_predictions17 - test_labels17
plt.figure()
plt.hist(train_error17, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError17.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error17, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError17.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error17, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError17.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_17 = {}
validation_17 = {}
test_17 = {}
train_17['train_labels'] = train_labels17
train_17['train_predictions'] = train_predictions17
validation_17['validat_labels'] = validat_labels17
validation_17['validat_predictions'] = validat_predictions17
test_17['test_labels'] = test_labels17
test_17['test_predictions'] = test_predictions17
train_pre17 = pd.DataFrame(train_17)
valid_pre17 = pd.DataFrame(validation_17)
testt_pre17 = pd.DataFrame(test_17)
train_pre17.to_csv('trainpre17.csv')
valid_pre17.to_csv('validpre17.csv')
testt_pre17.to_csv('testtpre17.csv')

train_sum17 = {}
validat_sum17 = {}
test_sum17 = {}
train_sum17['tot'] = (train_pre17['train_labels']-train_pre17['train_labels'].mean())**2
train_sum17['res'] = (train_pre17['train_predictions']-train_pre17['train_labels'])**2
train_s17 = pd.DataFrame(train_sum17)
train_r217 = 1.0-train_s17['res'].sum() / train_s17['tot'].sum()

validat_sum17['tot'] = (valid_pre17['validat_labels']-valid_pre17['validat_labels'].mean())**2
validat_sum17['res'] = (valid_pre17['validat_predictions']-valid_pre17['validat_labels'])**2
validat_s17 = pd.DataFrame(validat_sum17)
validat_r217 = 1.0-validat_s17['res'].sum() / validat_s17['tot'].sum()

test_sum17['tot'] = (testt_pre17['test_labels']-testt_pre17['test_labels'].mean())**2
test_sum17['res'] = (testt_pre17['test_predictions']-testt_pre17['test_labels'])**2
test_s17 = pd.DataFrame(test_sum17)
test_r217 = 1.0-test_s17['res'].sum() / test_s17['tot'].sum()

r217 = {}
r217['train'] = train_r217
r217['validation'] = validat_r217
r217['test'] = test_r217
rr217 = pd.DataFrame(r217, index=['0'])
rr217.to_csv('R217.csv')

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

history18 = pd.DataFrame(history_18.history)
history18.to_csv('history18.csv')

hist_18 = pd.DataFrame(history_18.history).tail(1)
hist_18.to_csv('fits_18.csv')
test_18 = pd.DataFrame(test_results_18).T
test_18.to_csv('eval_18.csv')

train_predictions18 = model_18.predict(train_features18).flatten()
validat_predictions18 = model_18.predict(validat_features18).flatten()
test_predictions18 = model_18.predict(test_features18).flatten()

plt.figure()
plt.plot(history_18.history['loss'], label='tra_loss')
plt.plot(history_18.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss18.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels18, train_predictions18)
plt.scatter(validat_labels18, validat_predictions18)
plt.scatter(test_labels18, test_predictions18)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity18.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error18 = train_predictions18 - train_labels18
validat_error18 = validat_predictions18 - validat_labels18
test_error18 = test_predictions18 - test_labels18
plt.figure()
plt.hist(train_error18, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError18.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error18, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError18.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error18, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError18.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_18 = {}
validation_18 = {}
test_18 = {}
train_18['train_labels'] = train_labels18
train_18['train_predictions'] = train_predictions18
validation_18['validat_labels'] = validat_labels18
validation_18['validat_predictions'] = validat_predictions18
test_18['test_labels'] = test_labels18
test_18['test_predictions'] = test_predictions18
train_pre18 = pd.DataFrame(train_18)
valid_pre18 = pd.DataFrame(validation_18)
testt_pre18 = pd.DataFrame(test_18)
train_pre18.to_csv('trainpre18.csv')
valid_pre18.to_csv('validpre18.csv')
testt_pre18.to_csv('testtpre18.csv')

train_sum18 = {}
validat_sum18 = {}
test_sum18 = {}
train_sum18['tot'] = (train_pre18['train_labels']-train_pre18['train_labels'].mean())**2
train_sum18['res'] = (train_pre18['train_predictions']-train_pre18['train_labels'])**2
train_s18 = pd.DataFrame(train_sum18)
train_r218 = 1.0-train_s18['res'].sum() / train_s18['tot'].sum()

validat_sum18['tot'] = (valid_pre18['validat_labels']-valid_pre18['validat_labels'].mean())**2
validat_sum18['res'] = (valid_pre18['validat_predictions']-valid_pre18['validat_labels'])**2
validat_s18 = pd.DataFrame(validat_sum18)
validat_r218 = 1.0-validat_s18['res'].sum() / validat_s18['tot'].sum()

test_sum18['tot'] = (testt_pre18['test_labels']-testt_pre18['test_labels'].mean())**2
test_sum18['res'] = (testt_pre18['test_predictions']-testt_pre18['test_labels'])**2
test_s18 = pd.DataFrame(test_sum18)
test_r218 = 1.0-test_s18['res'].sum() / test_s18['tot'].sum()

r218 = {}
r218['train'] = train_r218
r218['validation'] = validat_r218
r218['test'] = test_r218
rr218 = pd.DataFrame(r218, index=['0'])
rr218.to_csv('R218.csv')

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

history19 = pd.DataFrame(history_19.history)
history19.to_csv('history19.csv')

hist_19 = pd.DataFrame(history_19.history).tail(1)
hist_19.to_csv('fits_19.csv')
test_19 = pd.DataFrame(test_results_19).T
test_19.to_csv('eval_19.csv')

train_predictions19 = model_19.predict(train_features19).flatten()
validat_predictions19 = model_19.predict(validat_features19).flatten()
test_predictions19 = model_19.predict(test_features19).flatten()

plt.figure()
plt.plot(history_19.history['loss'], label='tra_loss')
plt.plot(history_19.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss19.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels19, train_predictions19)
plt.scatter(validat_labels19, validat_predictions19)
plt.scatter(test_labels19, test_predictions19)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity19.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error19 = train_predictions19 - train_labels19
validat_error19 = validat_predictions19 - validat_labels19
test_error19 = test_predictions19 - test_labels19
plt.figure()
plt.hist(train_error19, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError19.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error19, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError19.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error19, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError19.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_19 = {}
validation_19 = {}
test_19 = {}
train_19['train_labels'] = train_labels19
train_19['train_predictions'] = train_predictions19
validation_19['validat_labels'] = validat_labels19
validation_19['validat_predictions'] = validat_predictions19
test_19['test_labels'] = test_labels19
test_19['test_predictions'] = test_predictions19
train_pre19 = pd.DataFrame(train_19)
valid_pre19 = pd.DataFrame(validation_19)
testt_pre19 = pd.DataFrame(test_19)
train_pre19.to_csv('trainpre19.csv')
valid_pre19.to_csv('validpre19.csv')
testt_pre19.to_csv('testtpre19.csv')

train_sum19 = {}
validat_sum19 = {}
test_sum19 = {}
train_sum19['tot'] = (train_pre19['train_labels']-train_pre19['train_labels'].mean())**2
train_sum19['res'] = (train_pre19['train_predictions']-train_pre19['train_labels'])**2
train_s19 = pd.DataFrame(train_sum19)
train_r219 = 1.0-train_s19['res'].sum() / train_s19['tot'].sum()

validat_sum19['tot'] = (valid_pre19['validat_labels']-valid_pre19['validat_labels'].mean())**2
validat_sum19['res'] = (valid_pre19['validat_predictions']-valid_pre19['validat_labels'])**2
validat_s19 = pd.DataFrame(validat_sum19)
validat_r219 = 1.0-validat_s19['res'].sum() / validat_s19['tot'].sum()

test_sum19['tot'] = (testt_pre19['test_labels']-testt_pre19['test_labels'].mean())**2
test_sum19['res'] = (testt_pre19['test_predictions']-testt_pre19['test_labels'])**2
test_s19 = pd.DataFrame(test_sum19)
test_r219 = 1.0-test_s19['res'].sum() / test_s19['tot'].sum()

r219 = {}
r219['train'] = train_r219
r219['validation'] = validat_r219
r219['test'] = test_r219
rr219 = pd.DataFrame(r219, index=['0'])
rr219.to_csv('R219.csv')

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

history20 = pd.DataFrame(history_20.history)
history20.to_csv('history20.csv')

hist_20 = pd.DataFrame(history_20.history).tail(1)
hist_20.to_csv('fits_20.csv')
test_20 = pd.DataFrame(test_results_20).T
test_20.to_csv('eval_20.csv')

train_predictions20 = model_20.predict(train_features20).flatten()
validat_predictions20 = model_20.predict(validat_features20).flatten()
test_predictions20 = model_20.predict(test_features20).flatten()

plt.figure()
plt.plot(history_20.history['loss'], label='tra_loss')
plt.plot(history_20.history['val_loss'], label='val_loss')
plt.xlim([0,3000])
plt.ylim([0,0.01])
plt.xlabel('Epoch')
plt.ylabel('Mean square error ($eV^2$)')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('Loss20.pdf', bbox_inches='tight', dpi=600)
plt.close()

plt.figure()
plt.axes(aspect='equal')
plt.scatter(train_labels20, train_predictions20)
plt.scatter(validat_labels20, validat_predictions20)
plt.scatter(test_labels20, test_predictions20)
plt.xlabel('True Values (eV)')
plt.ylabel('Predictions (eV)')
lims = [1.0,1.9]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
# plt.show()
plt.savefig('Parity20.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_error20 = train_predictions20 - train_labels20
validat_error20 = validat_predictions20 - validat_labels20
test_error20 = test_predictions20 - test_labels20
plt.figure()
plt.hist(train_error20, bins=50)
plt.xlabel('Train Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('trainError20.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(validat_error20, bins=50)
plt.xlabel('Validation Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('validError20.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

plt.figure()
plt.hist(test_error20, bins=50)
plt.xlabel('Test Error (eV)')
plt.ylabel('Count')
# plt.show()
plt.savefig('testError20.pdf', bbox_inches = 'tight', dpi=600)
plt.close()

train_20 = {}
validation_20 = {}
test_20 = {}
train_20['train_labels'] = train_labels20
train_20['train_predictions'] = train_predictions20
validation_20['validat_labels'] = validat_labels20
validation_20['validat_predictions'] = validat_predictions20
test_20['test_labels'] = test_labels20
test_20['test_predictions'] = test_predictions20
train_pre20 = pd.DataFrame(train_20)
valid_pre20 = pd.DataFrame(validation_20)
testt_pre20 = pd.DataFrame(test_20)
train_pre20.to_csv('trainpre20.csv')
valid_pre20.to_csv('validpre20.csv')
testt_pre20.to_csv('testtpre20.csv')

train_sum20 = {}
validat_sum20 = {}
test_sum20 = {}
train_sum20['tot'] = (train_pre20['train_labels']-train_pre20['train_labels'].mean())**2
train_sum20['res'] = (train_pre20['train_predictions']-train_pre20['train_labels'])**2
train_s20 = pd.DataFrame(train_sum20)
train_r220 = 1.0-train_s20['res'].sum() / train_s20['tot'].sum()

validat_sum20['tot'] = (valid_pre20['validat_labels']-valid_pre20['validat_labels'].mean())**2
validat_sum20['res'] = (valid_pre20['validat_predictions']-valid_pre20['validat_labels'])**2
validat_s20 = pd.DataFrame(validat_sum20)
validat_r220 = 1.0-validat_s20['res'].sum() / validat_s20['tot'].sum()

test_sum20['tot'] = (testt_pre20['test_labels']-testt_pre20['test_labels'].mean())**2
test_sum20['res'] = (testt_pre20['test_predictions']-testt_pre20['test_labels'])**2
test_s20 = pd.DataFrame(test_sum20)
test_r220 = 1.0-test_s20['res'].sum() / test_s20['tot'].sum()

r220 = {}
r220['train'] = train_r220
r220['validation'] = validat_r220
r220['test'] = test_r220
rr220 = pd.DataFrame(r220, index=['0'])
rr220.to_csv('R220.csv')

# summary
hist = pd.concat([hist_1, hist_2, hist_3, hist_4, hist_5, 
                  hist_6, hist_7, hist_8, hist_9, hist_10, 
                  hist_11, hist_12, hist_13, hist_14, hist_15, 
                  hist_16, hist_17, hist_18, hist_19, hist_20], 
                  ignore_index=True)

test_1 = pd.read_csv('eval_1.csv')
test_2 = pd.read_csv('eval_2.csv')
test_3 = pd.read_csv('eval_3.csv')
test_4 = pd.read_csv('eval_4.csv')
test_5 = pd.read_csv('eval_5.csv')
test_6 = pd.read_csv('eval_6.csv')
test_7 = pd.read_csv('eval_7.csv')
test_8 = pd.read_csv('eval_8.csv')
test_9 = pd.read_csv('eval_9.csv')
test_10 = pd.read_csv('eval_10.csv')
test_11 = pd.read_csv('eval_11.csv')
test_12 = pd.read_csv('eval_12.csv')
test_13 = pd.read_csv('eval_13.csv')
test_14 = pd.read_csv('eval_14.csv')
test_15 = pd.read_csv('eval_15.csv')
test_16 = pd.read_csv('eval_16.csv')
test_17 = pd.read_csv('eval_17.csv')
test_18 = pd.read_csv('eval_18.csv')
test_19 = pd.read_csv('eval_19.csv')
test_20 = pd.read_csv('eval_20.csv')

test = pd.concat([test_1, test_2, test_3, test_4, test_5, 
                  test_6, test_7, test_8, test_9, test_10, 
                  test_11, test_12, test_13, test_14, test_15, 
                  test_16, test_17, test_18, test_19, test_20], 
                  ignore_index=True)

rr2 = pd.concat([rr21, rr22, rr22, rr24, rr25, 
                 rr26, rr27, rr28, rr29, rr210, 
                 rr211, rr212, rr213, rr214, rr215, 
                 rr216, rr217, rr218, rr219, rr220],
                 ignore_index=True)

histestrr2 = pd.concat([hist, test, rr2], axis=1)
histestrr2.to_csv('histestrr2.csv')

histestrr2.describe().to_csv('result.csv')
