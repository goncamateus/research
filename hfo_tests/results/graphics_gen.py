import os
import re

import matplotlib.pyplot as plt
import numpy as np

file10 = [None, None, None, None]
file10[0] = open('10k_training_Trials.txt', 'r')
file10[1] = open('10k_training_rewards.txt', 'r')
file10[2] = open('10k_test_Trials.txt', 'r')
file10[3] = open('10k_test_rewards.txt', 'r')

file50 = [None, None, None, None]
file50[0] = open('50k_training_Trials.txt', 'r')
file50[1] = open('50k_training_rewards.txt', 'r')
file50[2] = open('50k_test_Trials.txt', 'r')
file50[3] = open('50k_test_rewards.txt', 'r')

data10 = []
for f in file10:
    data10.append(f.readlines())
    f.close()

data50 = []
for f in file50:
    data50.append(f.readlines())
    f.close()

trials = {'10k': {'training': int(data10[0][-4].split(': ')[1])/int(data10[0][-5].split(': ')[1]), 'test': int(data10[2][-4].split(': ')[1])/int(data10[2][-5].split(': ')[1])}, '50k': {
    'training': int(data50[0][-4].split(': ')[1])/int(data50[0][-5].split(': ')[1]), 'test': int(data50[2][-4].split(': ')[1])/int(data50[2][-5].split(': ')[1])}}
match = re.compile('Total reward: [0-9]*.[0-9]*')
reward10_train = []
for r in data10[1]:
    reward10_train.append(float(re.findall(match, r)[0].split(' ')[2])/1000)
reward10_train = np.array(reward10_train)

reward10_test = []
for r in data10[3]:
    reward10_test.append(float(re.findall(match, r)[0].split(' ')[2])/1000)
reward10_test = np.array(reward10_test)

reward50_train = []
for r in data50[1]:
    reward50_train.append(float(re.findall(match, r)[0].split(' ')[2])/1000)
reward50_train = np.array(reward50_train)

reward50_test = []
for r in data50[3]:
    reward50_test.append(float(re.findall(match, r)[0].split(' ')[2])/1000)
reward50_test = np.array(reward50_test)

training = [100 - trials['10k']['training'] *
            100, 100 - trials['50k']['training']*100]
test = [100 - trials['10k']['test']*100, 100 - trials['50k']['test']*100]
names = ['10K Model', '50K Model']

plt.plot(reward10_train)
plt.xticks([x for x in range(0,10000) if x%500==0], rotation='vertical')
plt.show()
