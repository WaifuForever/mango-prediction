import os
from shutil import copyfile

TRAINING_DIR = "./Data/Training"
DIR = "./Data/temp/Frente"

#lote1F_Rotten
'''lote = ["003", "005", "012", "013", "015", "016", "017", "018", "019", "022", "023", "025", "031", "032",
 "036", "038", "040", "042", "044", "046", "049", "051", "052", "054", "056", "058", "060", "065", "067",
 "068", "069", "071", "072", "073", "075", "083", "084", "085", "086", "088", "094", "095", "096"]'''

'''
#lote2F_Rotten
lote = ["001", "005", "006", "009", "014", "016", "026", "033", "034", "035", "036", "039", "041", "042",
 "043", "045", "048", "051", "055", "056", "061", "062", "064", "068", "070", "072", "074", "078", "079",
 "081", "090", "093", "094", "097", "099", "103", "104", "105", "108", "111", "113", "115", "116", 117]
'''
'''
#lote2V_Rotten
lote = ["001", "002", "003", "004", "007", "008", "009", "014", "015", "016", "017", "022", "025", "027",
 "028", "041", "045", "047", "048", "050", "052", "055", "057", "058", "059", "066", "069", "070", "075",
 "081", "082", "084", "087", "088", "089", "091", "092", "094", "095", "096", "097", "098", "099",
 "104", "105", "107", 109, 110, 111, 112, 114, 115, 117, 118, 119]
'''
'''
#lote3F_Rotten
lote = ["008", "012", "016", "018", "021", "038", "045", "047", "049"]
'''
'''
#lote3V_Rotten
lote = ["001", "003", "012", "014", "018", "027", "034", "041"]
'''
onlyfiles = []
for path in os.listdir(DIR):   
  onlyfiles.append(path)

x = 0

Rotten_num= 1

print(len(onlyfiles))
print(len(lote))

print(onlyfiles[2])

for j in range(x, len(lote)):
  #print("k = {onlyfiles} ---- j = {name}".format(onlyfiles = onlyfiles[k], name = j))
  name = "mL03_{num}V.jpg".format(num = lote[j])
  if name in onlyfiles:
    if os.path.exists(DIR + '/' + name):
      x+=1
      print("Rotten {rotten} - {name}".format(rotten = Rotten_num, name = lote[j]))
      Rotten_num += 1
      current_path = DIR + '/' + name                  
      copyfile(current_path, TRAINING_DIR + '/Rotten/' + name)
      onlyfiles.remove(name)
      
    else:   
      x+=1    
      print("{name} does not exists in directory".format(name= name))

  else:         
    x+=1     
    print("{name} does not exists in onlyfiles".format(name= name))

Good_num= 1
for k in range(0, len(onlyfiles)):
  #print("k = {onlyfiles} ---- j = {name}".format(onlyfiles = onlyfiles[k], name = j))
 
 
  if os.path.exists(DIR + '/' + onlyfiles[k]):
    x+=1
    print("Good {rotten} - {num}".format(rotten = Good_num, num = onlyfiles[k]))
    Good_num += 1
    current_path = DIR + '/' + onlyfiles[k]                  
    copyfile(current_path, TRAINING_DIR + '/Good/' + onlyfiles[k])
    
    
  else:   
    x+=1    
    print("{name} does not exists in directory".format(name= onlyfiles[k]))

 
