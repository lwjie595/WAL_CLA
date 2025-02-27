import numpy as np
import pandas as pd
import os

csv_file = "../kaggle/Data_Entry_2017.csv"  #modify to your Chest-X-ray path
df = pd.read_csv(csv_file)




filter_values = ["PA"]
filtered_rows = df[df["View Position"].isin(filter_values)][["Image Index","Finding Labels"]]
print(filtered_rows.shape)
index=1


def find_label(Labels,file_name):
    for folder in Labels:
        full_path = os.path.join(folder, file_name)
        if os.path.exists(full_path):
            return full_path
    return None


Labels=["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass","Nodule","Pneumonia","Pneumothorax","Consolidation",
        "Edema","Emphysema", "Fibrosis","Pleural_Thickening","Hernia"]
Label_with_multi=[i for i in range(14)]
print(Label_with_multi)
#
# filtered_rows = filtered_rows[filtered_rows["Finding Labels"].isin(Labels)]###

# filtered_rows = filtered_rows[x.split('|') for x in filtered_rows["Finding Labels"]]
# filtered_rows["Finding Labels"] =filtered_rows["Finding Labels"].apply(lambda x: Labels.index(x.split('|')[0]))
# filtered_rows = filtered_rows[filtered_rows["Finding Labels"].isin(Label_with_multi)]
filtered_rows = filtered_rows[~filtered_rows["Finding Labels"].isin(["No Finding"])]
print(filtered_rows.shape)
print(index)
shape_filter=filtered_rows.shape[0]
for i in range(shape_filter):
    label_each=np.zeros(len(Label_with_multi))
    tt=filtered_rows.iloc[i,df.columns.get_loc("Finding Labels")].split('|')
    for j in tt:
        if not j=="No Finding":
            label_each[Labels.index(j)]=1
    if index<10:
        file_path_prefix = "../kaggle/images_00{}/images/".format(index)
    else:
        file_path_prefix = "../kaggle/images_0{}/images/".format(index)
    filtered_rows.iloc[i,df.columns.get_loc("Image Index")] = os.path.join(file_path_prefix, filtered_rows.iloc[i,df.columns.get_loc("Image Index")])
    filtered_rows.iloc[i, df.columns.get_loc("Finding Labels")]=label_each
    if (i<(shape_filter-1)) and not os.path.exists(file_path_prefix+filtered_rows.iloc[i+1,df.columns.get_loc("Image Index")]):
        index+=1
        print(index)



output_path="./data/Chex14ray/"
output_file ="PA_MultiLabel2.txt"

if not os.path.exists(output_path) :
    os.makedirs(output_path)
filtered_rows.to_csv(output_path+output_file, index=False, sep=" ", header=False)

print(f"the file has been saved {output_file}")
#
#
