import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

plt.rcParams.update({'font.size': 18})
# file10 = [None, None, None, None]
# file10[0] = open('10k_training_Trials.txt', 'r')
# file10[1] = open('10k_training_rewards.txt', 'r')
# file10[2] = open('10k_test_Trials.txt', 'r')
# file10[3] = open('10k_test_rewards.txt', 'r')

# file50 = [None, None, None, None]
# file50[0] = open('50k_training_Trials.txt', 'r')
# file50[1] = open('50k_training_rewards.txt', 'r')
# file50[2] = open('50k_test_Trials.txt', 'r')
# file50[3] = open('50k_test_rewards.txt', 'r')


train_trials_files = [dict(), dict()]
test_trials_files = [dict(), dict()]
data10k = [dict(), dict()]
data50k = [dict(), dict()]
random = None
for x in os.listdir('.'):
    split = x.split('_')
    f = open(x, 'r')
    name = None
    if x.find('10k_training_Trials.txt') >= 0:
        if split[1].find('k') == -1:
            name = split[0]+'_'+split[1]
        else:
            name = split[0]
        data = f.readlines()
        train_trials_files[0][name] = data
        data10k[0][name] = data
    elif x.find('50k_training_Trials.txt') >= 0:
        if split[1].find('k') == -1:
            name = split[0]+'_'+split[1]
        else:
            name = split[0]
        data = f.readlines()
        train_trials_files[1][name] = data
        data50k[0][name] = data
    if x.find('10k_test_Trials.txt') >= 0:
        if split[1].find('k') == -1:
            name = split[0]+'_'+split[1]
        else:
            name = split[0]
        data = f.readlines()
        test_trials_files[0][name] = data
        data10k[1][name] = data
    elif x.find('50k_test_Trials.txt') >= 0:
        if split[1].find('k') == -1:
            name = split[0]+'_'+split[1]
        else:
            name = split[0]
        data = f.readlines()
        test_trials_files[1][name] = data
        data50k[1][name] = data
    elif x.find('random') >= 0:
        random = f.readlines()
    f.close()

for t in train_trials_files:
    for k, data in t.items():
        t[k] = 1 - int(data[-4].split(': ')[1])/int(data[-5].split(': ')[1])

for t in test_trials_files:
    for k, data in t.items():
        t[k] = 1 - int(data[-4].split(': ')[1])/int(data[-5].split(': ')[1])

random = 1 - int(random[-4].split(': ')[1])/int(random[-5].split(': ')[1])

list10k = [x for x in train_trials_files[0].items()]
list10k.insert(0, ('Random Agent',random))
list10k = sorted(list10k, key=lambda x: x[1])
plt.figure(1, figsize=[9, 3])
plt.bar([x[0] for x in list10k], [x[1] for x in list10k])
plt.title('10k Episodes training')
plt.ylabel('Successful Defenses (%)')
plt.savefig('Train_Success_10k')
#--------------------------
list50k = [x for x in train_trials_files[1].items()]
list50k = sorted(list50k, key=lambda x: x[1])
plt.figure(2, figsize=[9, 3])
plt.bar([x[0] for x in list50k], [x[1] for x in list50k])
plt.title('50k Episodes training')
plt.ylabel('Successful Defenses (%)')
plt.savefig('Train_Success_50k')
#--------------------------
list10k = [x for x in test_trials_files[0].items()]
list10k.insert(0, ('Random Agent',random))
list10k = sorted(list10k, key=lambda x: x[1])
plt.figure(3, figsize=[9, 3])
plt.bar([x[0] for x in list10k], [x[1] for x in list10k])
plt.title('10k Episodes test')
plt.ylabel('Successful Defenses (%)')
plt.savefig('Test_Success_10k')
#--------------------------
list50k = [x for x in test_trials_files[1].items()]
list50k = sorted(list50k, key=lambda x: x[1])
plt.figure(4, figsize=[9, 3])
plt.bar([x[0] for x in list50k], [x[1] for x in list50k])
plt.title('50k Episodes test')
plt.ylabel('Successful Defenses (%)')
plt.savefig('Test_Success_50k')


defenses_10k_training = [[] for x in data10k[0].keys()]
names = [x for x in data10k[0].keys()]
xs = []
for i, (k, data) in enumerate(data10k[0].items()):
    g = 0
    for j, line in enumerate(data[:-5]):
        if line.find('GOAL') == -1:
            g += 1
        if j != 0 and j % 50 == 0:
            if i == 0:
                xs.append(j)
            defenses_10k_training[i].append(g)
            g = 0

xs = np.array(xs)
defenses_10k_training = np.array(defenses_10k_training)

plt.figure(5, figsize=(9, 3))
plt.ylabel('Number of Defenses for 50 Trials')
plt.title('10k Episodes Training Defenses Flow')
for i, d in enumerate(defenses_10k_training):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(xs.min(), xs.max(), 2000)
    spl = spline(xs, d, xnew)  # BSpline object

    plt.plot(xnew, spl, label=names[i])
plt.savefig('10k_training_flow', bbox_inches='tight', pad_inches=0)

#------------------------------------------------------------------

defenses_10k_test = [[] for x in data10k[1].keys()]
names = [x for x in data10k[1].keys()]
xs = []
for i, (k, data) in enumerate(data10k[1].items()):
    g = 0
    for j, line in enumerate(data[:-5]):
        if line.find('GOAL') == -1:
            g += 1
        if j != 0 and j % 50 == 0:
            if i == 0:
                xs.append(j)
            defenses_10k_test[i].append(g)
            g = 0

xs = np.array(xs)
defenses_10k_test = np.array(defenses_10k_test)

plt.figure(6, figsize=(9, 3))
plt.ylabel('Number of Defenses for 50 Trials')
plt.title('10k Episodes Test Defenses Flow')
for i, d in enumerate(defenses_10k_test):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(xs.min(), xs.max(), 200)
    spl = spline(xs, d, xnew)  # BSpline object

    plt.plot(xnew, spl, label=names[i])
plt.savefig('10k_test_flow', bbox_inches='tight', pad_inches=0)

#------------------------------------------------------------------

defenses_50k_training = [[] for x in data50k[0].keys()]
names = [x for x in data50k[0].keys()]
xs = []
for i, (k, data) in enumerate(data50k[0].items()):
    g = 0
    for j, line in enumerate(data[:-5]):
        if line.find('GOAL') == -1:
            g += 1
        if j != 0 and j % 50 == 0:
            if i == 0:
                xs.append(j)
            defenses_50k_training[i].append(g)
            g = 0

xs = np.array(xs)
defenses_50k_training = np.array(defenses_50k_training)

plt.figure(7, figsize=(9, 3))
plt.ylabel('Number of Defenses for 50 Trials')
plt.title('50k Episodes Training Defenses Flow')
for i, d in enumerate(defenses_50k_training):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(xs.min(), xs.max(), 10000)
    spl = spline(xs, d, xnew)  # BSpline object

    plt.plot(xnew, spl, label=names[i])
plt.savefig('50k_training_flow', bbox_inches='tight', pad_inches=0)

#------------------------------------------------------------------

defenses_50k_test = [[] for x in data50k[1].keys()]
names = [x for x in data50k[1].keys()]
xs = []
for i, (k, data) in enumerate(data50k[1].items()):
    g = 0
    for j, line in enumerate(data[:-5]):
        if line.find('GOAL') == -1:
            g += 1
        if j != 0 and j % 50 == 0:
            if i == 0:
                xs.append(j)
            defenses_50k_test[i].append(g)
            g = 0

xs = np.array(xs)
defenses_50k_test = np.array(defenses_50k_test)

plt.figure(8, figsize=(9, 3))
plt.ylabel('Number of Defenses for 50 Trials')
plt.title('50k Episodes Test Defenses Flow')
for i, d in enumerate(defenses_50k_test):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(xs.min(), xs.max(), 200)
    spl = spline(xs, d, xnew)  # BSpline object

    plt.plot(xnew, spl, label=names[i])
plt.savefig('50k_test_flow', bbox_inches='tight', pad_inches=0)
#------------------------------------------------------------------