# ECE-594-BB-Project-Ideas
Ideas for the UCSB ECE 594BB AI Robustness Course 2026 Winter

* RobustBench: [link](https://robustbench.github.io/)
* Image Net tiny: [link](https://drive.google.com/file/d/1AQGoMBlmXvFhJIKaDj4YbB-v241ppLXA/view)
* Work Directory: [Colab Bruce](https://colab.research.google.com/drive/17b1X_DI12Zo_bNVs-fkRO4ZPdOg-wy-9?usp=sharing) [Colab Eli](https://colab.research.google.com/drive/17BSkgq4CwgkxlkK_jZGc9JLm0qwgpS3C?usp=sharing#scrollTo=4b40eda6)

### CIFAR10 800-Subset [link](https://drive.google.com/file/d/1odeARw_-PWqMM2ysod9xwqnzdxencLEZ/view?usp=sharing)
How to load it:
```
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Load
ds = load_dataset("parquet", data_files="/content/cifar10_balanced_224.parquet")
print(f"Numerical order with corresponding categories: \n ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']")
# OR: ds = Dataset.from_parquet("./cifar10_balanced_224.parquet")

# Verify
print(ds)
# Dataset({
#     features: ['image', 'label', 'label_name'],
#     num_rows: 1000
# })
## DatasetDict (ds) is actually a dictionary type:
print(f"type(ds) with keys: {ds.keys()}")
# <class 'datasets.dataset_dict.DatasetDict'>
print(ds['train'][0])
# {'image': <PIL.PngImagePlugin.PngImageFile>,
#  'label': 3,
#  'label_name': 'cat'}

# PyTorch transform (applied lazily)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def preprocess(batch):
    batch["tensor_values"] = [transform(img.convert("RGB")) for img in batch["image"]]
    del batch["image"]        # <-- remove PIL images so collator doesn't choke
    #del batch["label_name"]   # <-- also remove strings (optional, but cleaner)
    return batch

ds.set_transform(preprocess)

dataloader = DataLoader(ds['train'], batch_size=32, shuffle=True)

## Access the component
for batch in dataloader:
    images = batch["tensor_values"]  # [32, 3, 224, 224]
    labels = batch["label"]         # [32]
    print(f"Batch shape: {images.shape} with type {type(images)}, Labels: {type(labels[:5])}")
    break
```
