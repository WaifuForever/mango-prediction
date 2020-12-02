import os
TRAINING_DIR = 'C:/Users/Waifu/Desktop/data/Prediction/'

print(os.listdir(TRAINING_DIR))

for f in os.listdir(TRAINING_DIR):
    DIR = os.path.join(TRAINING_DIR, f)
    if os.path.isfile(DIR):
        x = DIR.split('.') 
        os.rename(DIR, x[0] + '(2).' + x[1])
        print(DIR)
print("done")