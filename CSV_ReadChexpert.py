import numpy as np
import pandas as pd
import os
csv_file = "../Chest/chexpertchestxrays-u20210408/train_cheXbert.csv"  # #modify to your ChexPert path
df = pd.read_csv(csv_file)



filter_values = ["Frontal"] #
Labels=["Path","Enlarged Cardiomediastinum",	"Cardiomegaly",	"Lung Opacity",	"Lung Lesion",	"Edema",	"Consolidation",
        "Pneumonia",	"Atelectasis",	"Pneumothorax",	"Pleural Effusion",	"Pleural Other",	"Fracture",
        "Support Devices",   "No Finding"]
filtered_rows = df[df["Frontal/Lateral"].isin(filter_values)][Labels]
print(filtered_rows.shape)
index=2


def find_label(Labels,file_name):
    for folder in Labels:
        full_path = os.path.join(folder, file_name)
        if os.path.exists(full_path):
            return full_path  #
    return None  #



Label_with_multi=[i for i in range(14)]
print(Label_with_multi)
print(filtered_rows.shape)
print(index)
shape_filter=filtered_rows.shape[0]
filtered_rows["Label"]=filtered_rows["Path"].apply(lambda x: [])
for i in range(shape_filter):
    label_each=np.zeros(len(Label_with_multi))
    Path=filtered_rows.iloc[i,filtered_rows.columns.get_loc("Path")].split('/')


    file_path_prefix = "../Chest/chexpertchestxrays-u20210408/CheXpert{}/".format(index)

    filtered_rows.iloc[i,filtered_rows.columns.get_loc("Path")] = os.path.join(file_path_prefix, Path[2], Path[3], Path[4])

    if (i<(shape_filter-1)) and not os.path.exists(file_path_prefix+filtered_rows.iloc[i+1,filtered_rows.columns.get_loc("Path")].split('/')[2]):
        index+=1
        print(index)
    for j in range(1,15):
        if filtered_rows.iloc[i,filtered_rows.columns.get_loc(Labels[j])]==1:
            label_each[j-1]=1
        else:
            label_each[j-1] =0

    filtered_rows.iloc[i, filtered_rows.columns.get_loc("Label")].append(list(label_each))

filtered_rows["Label"]= filtered_rows["Label"].apply(lambda x: x[0])
print(filtered_rows.iloc[0, filtered_rows.columns.get_loc("Label")])
filtered_rows = filtered_rows[["Path","Label"]]


output_path="./data/Chexpert/"
output_file ="Frontal_MultiLabel2.txt"

if not os.path.exists(output_path) :
    os.makedirs(output_path)
filtered_rows.to_csv(output_path+output_file, index=False, sep=" ", header=False)

print(f"the file has been saved {output_file}")
#
#
