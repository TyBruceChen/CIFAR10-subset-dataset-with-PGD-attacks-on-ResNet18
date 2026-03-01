# ECE-594-BB-Project-Ideas
Ideas for the UCSB ECE 594BB AI Robustness Course 2026 Winter

* RobustBench: [link](https://robustbench.github.io/)
* Image Net tiny: [link](https://drive.google.com/file/d/1AQGoMBlmXvFhJIKaDj4YbB-v241ppLXA/view)
* Work Directory: [Colab Bruce](https://colab.research.google.com/drive/17b1X_DI12Zo_bNVs-fkRO4ZPdOg-wy-9?usp=sharing) [Colab Eli](https://colab.research.google.com/drive/17BSkgq4CwgkxlkK_jZGc9JLm0qwgpS3C?usp=sharing#scrollTo=4b40eda6)

### CIFAR10 800-Subset [link](https://drive.google.com/file/d/1odeARw_-PWqMM2ysod9xwqnzdxencLEZ/view?usp=sharing)
How to load it:
```
from datasets import load_dataset
ds = load_dataset("parquet", data_files="/content/cifar10_balanced_224.parquet")
ds['train'][0]
```
<details>
    <summary>How to load for training</summary>
Caution: the train_test_split() method has a randomness parameter default setting if use.
    
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
        
</details>

### Fine-tuned ResNet18 trained on CIFAR10-800:
<[Full fine-tuned](https://drive.google.com/file/d/1epcOcBc6n_TNAXCUA9kipf24IUXeeoXs/view?usp=sharing), 0.90875 val acc> <[Full fine-tuned w/ augmentation](https://drive.google.com/file/d/1SeHPmC2KzBrDx4X3PU9QVv6FEe8ARhKK/view?usp=sharing), 0.925 val acc, **Currently Used for Adversarial Dataset Generation**>
<details>
<summary>Configuration</summary>
Pr-trained model from 

    torchvision.models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
Preprocessing methods:
* torchvision.transforms.ToTensor()
* Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
* If w/ augmentation:
```
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),  # [0, 1] range
])
```
</details>

<details>
    <summary>Training Process Validation Accuracy</summary>
    <img width="3410" height="1760" alt="W B Chart 28_02_2026, 02_55_04" src="https://github.com/user-attachments/assets/fd52e0f7-c3ad-453c-9038-096e8cc232c2" />
    <img width="3410" height="1760" alt="W B Chart 28_02_2026, 02_53_54" src="https://github.com/user-attachments/assets/ce623cd0-6765-473d-8a9c-bbfa226839e2" />

</details>

<details>
    <summary>Load Model</summary>

    class NormalizedModel(nn.Module):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.base_model(x)

    base_model = models.resnet18(weights=None)  # no pretrained weights needed
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    
    model = NormalizedModel(
        base_model,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    model.load_state_dict(torch.load("resnet18_fft_augmentation.pth"))
    model.eval()
</details>

### CIFAR10-800 Adversarial Attack Dataset: [link](https://drive.google.com/file/d/1I7t8VGqKLKOvh-Kk8DoRtBI8eX3HI6Ps/view?usp=sharing)
* features: original image (saved in PIL image format) with corresponding $L_2$ and $L_{\infty}$ PGD perturbed images (saved in float32 to keep precision, shape in (3,224,224)), also with labels from these three images.
* Columns are ['original', 'linf_pgd', 'l2_pgd', 'true_label', 'linf_pgd_pred_label', 'l2_pgd_pred_label']
* Numer of rows: 8000 (800 each class)
* Visualization:

    <img width="525" height="180" alt="LoadAAdataset" src="https://github.com/user-attachments/assets/da4aad96-ca25-4593-9af6-8b86818539bf" />


