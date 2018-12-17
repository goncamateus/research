import os
import tqdm
import re

print('\n')
print('Base training')
for i in tqdm.tqdm(range(1,4)):
    os.system('./HFO_RC.sh {} {} >> training_base.txt'.format('base',i))

print('Helios training')
for i in tqdm.tqdm(range(1,4)):
    os.system('./HFO_RC.sh {} {} >> training_helios.txt'.format('helios',i))

with open('training_base.txt', 'r') as tb:
    data = tb.read()
    finish = re.findall('EndOfTrial: [0-9]* / 1000',data)
    for i, f in enumerate(finish): 
        print('OPs {}:'.format(i+1))
        print(f.split(' ')[1])

with open('training_helios.txt', 'r') as tb:
    data = tb.read()
    finish = re.findall('EndOfTrial: [0-9]* / 1000',data)
    for i, f in enumerate(finish): 
        print('OPs {}:'.format(i+1))
        print(f.split(' ')[1])