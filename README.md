This is the code for Wasserstein adversarial learning with class-level alignment (WAL-CLA) for Medica imaging in multi-label domain adaptation.

Before run the code, please download the chest X-ray dataset first, containing Chest X-ray14 from https://www.kaggle.com/datasets/nih-chest-xrays/data 
and Chexpert from https://stanfordmlgroup.github.io/competitions/chexpert/.

After download the dataset, put the dataset in the right place as the path shown in the .txt in folder "/data/Chex14ray/" or "/data/Chexpert/".
Otherwise, you can generate the domains for domain adaptation in "CSV_Read_Chex14ray.py" or "CSV_Read_Chexpert.py" to obtain the dataset path as you wish.

you can run the code by "main.py, or bash "run_train.sh"
The model of feature extractor is in "model/resnet.py", and classifier is in "model/basnet.py"
Loss function of our method is in "utils/loss.py"










