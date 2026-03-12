# CIFAR10 8000 subset with PGD adversarial attack on ResNet18

Based on the research Unlabeled Data Improves Adversarial Robustness and Are Vision Transformers Robust to Patch Perturbations?, we wonder if it's possible to use the robust model like Swin Transformer to instruct the training of other CNN to become robust to adverial attack too, by semi-supervised training?

Research did on UCSB ECE594BB class, Winter 2026 quarter

* RobustBench: [link](https://robustbench.github.io/)
* GradCAM visualization: [link](https://github.com/TyBruceChen/GradCAM-torch.nn.Module)
* Work Directory: [Colab Bruce](https://colab.research.google.com/drive/17b1X_DI12Zo_bNVs-fkRO4ZPdOg-wy-9?usp=sharing) [Colab Eli](https://colab.research.google.com/drive/17BSkgq4CwgkxlkK_jZGc9JLm0qwgpS3C?usp=sharing#scrollTo=4b40eda6)

### Milestones:
* Trained finetuned ResNet18 on the subset of CIFAR10 (800 samples for each category)
* Apply L2PGD and LinfPGD adversarial attack by foolbox to the model and save the dataset (~9 GB)
* Experiment on inter-model and cross model transferability

### CIFAR10 800-Subset [link](https://huggingface.co/datasets/TyBruce/CIFAR10-PGD-attack-ResNet18/blob/main/cifar10_balanced_224.parquet) (without adversarial attack) 16-sample [example](https://huggingface.co/datasets/TyBruce/CIFAR10-PGD-attack-ResNet18/blob/main/sample_16.parquet) 
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
<[Full fine-tuned](https://huggingface.co/TyBruce/CIFAR10-subset-finetuned/blob/main/resnet18_fft.pth), 0.90875 val acc> <[Full fine-tuned w/ augmentation](https://huggingface.co/TyBruce/CIFAR10-subset-finetuned/blob/main/resnet18_fft_augmentation.pth)/ [Unwrapped (without buid-in input normalization)](https://huggingface.co/TyBruceCIFAR10-subset-finetuned/blob/main/resnet18_fft_augmentation_unwrapped.pth), 0.925 val acc, **Currently Used for Adversarial Dataset Generation**>
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

Or unwrapped:

    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load("resnet18.pth", map_location=torch.device('cpu')))
</details>

### CIFAR10-800 Adversarial Attack Dataset: [link](https://huggingface.co/datasets/TyBruce/CIFAR10-PGD-attack-ResNet18/blob/main/cifar10_adversarial_224_f32.parquet)

Features:
* original image (saved in PIL image format) with corresponding $L_2$ and $L_{\infty}$ PGD perturbed images (saved in float32 to keep precision, shape in (3,224,224)), also with labels from these three images.
* Columns are ['original', 'linf_pgd', 'l2_pgd', 'true_label', 'linf_pgd_pred_label', 'l2_pgd_pred_label']
* Numer of rows: 8000 (800 each class)
* Visualization:

    <img width="525" height="180" alt="LoadAAdataset" src="https://github.com/user-attachments/assets/da4aad96-ca25-4593-9af6-8b86818539bf" />

<details>
    <summary>Dataset Configuration Details</summary>
    Based on foolbox library:

    import foolbox as fb
    epsilon_linf = 8.0/255
    epsilon_l2 = 2.0  # L2 budget (commonly 0.5~3.0)
    rel_stepsize = 0.2
    steps = 10
    
    linf_attack = fb.attacks.LinfProjectedGradientDescentAttack(
        rel_stepsize=rel_stepsize,
        steps=steps,
        random_start=True,
    )
    
    l2_attack = fb.attacks.L2ProjectedGradientDescentAttack(
        rel_stepsize=rel_stepsize,
        steps=steps,
        random_start=True,
    )
    
</details>

<details>
    <summary>Generation Result</summary>

    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 92.50% (1480/1600)
                Linf PGD: 0.00% (0/1600)
                  L2 PGD: 0.00% (0/1600)
    ============================================================
    
    ============================================================
    Per-Class Accuracy
    ============================================================
    Class                Clean   Linf PGD     L2 PGD
    ---------------------------------------------
    airplane            95.62%      0.00%      0.00%
    automobile          97.50%      0.00%      0.00%
    bird                93.12%      0.00%      0.00%
    cat                 81.25%      0.00%      0.00%
    deer                90.00%      0.00%      0.00%
    dog                 87.50%      0.00%      0.00%
    frog                93.75%      0.00%      0.00%
    horse               91.88%      0.00%      0.00%
    ship                95.62%      0.00%      0.00%
    truck               98.75%      0.00%      0.00%
    
    ============================================================
    Linf PGD: Misclassification Distribution
    ============================================================
    True Class      → Predicted As       Count
    ---------------------------------------------
    automobile      → truck                 75
    horse           → deer                  69
    dog             → cat                   67
    truck           → automobile            58
    frog            → cat                   53
    airplane        → ship                  52
    bird            → frog                  47
    cat             → dog                   46
    ship            → airplane              44
    truck           → ship                  43
    deer            → horse                 42
    airplane        → bird                  42
    frog            → bird                  41
    cat             → frog                  41
    automobile      → ship                  40
    
    ============================================================
    L2 PGD: Misclassification Distribution
    ============================================================
    True Class      → Predicted As       Count
    ---------------------------------------------
    automobile      → truck                 80
    dog             → cat                   79
    truck           → automobile            62
    ship            → airplane              60
    horse           → deer                  56
    cat             → dog                   55
    airplane        → bird                  53
    frog            → cat                   53
    deer            → horse                 47
    horse           → dog                   44
    frog            → bird                  41
    deer            → bird                  41
    airplane        → ship                  39
    bird            → frog                  36
    automobile      → ship                  35
    
    ============================================================
    Verification: Stored Labels vs Fresh Predictions
    ============================================================
</details>

<details>
    <summary>Dataset Load</summary>

    from torch.utils.data import DataLoader
    from datasets import load_dataset
    import torchvision.transforms as T
    import torch
    
    adv_ds = load_dataset("parquet", data_files="cifar10_adversarial_224_f32.parquet")['train']
    device = 'cpu' if torch.cuda.is_available() == False else 'cuda'
    
    def preprocess(batch):
        to_tensor = T.ToTensor()
        batch["orig_pixels"] = [to_tensor(img.convert("RGB")) for img in batch["original"]]
        batch["linf_pixels"] = [torch.tensor(arr) for arr in batch["linf_pgd"]]
        batch["l2_pixels"] = [torch.tensor(arr) for arr in batch["l2_pgd"]]
        del batch["original"]
        del batch["linf_pgd"]
        del batch["l2_pgd"]
        return batch
    
    adv_ds.set_transform(preprocess)
    loader = DataLoader(adv_ds, batch_size=8, shuffle=False)
    
    for batch in loader:
        orig = batch["orig_pixels"].to(device)
        linf = batch["linf_pixels"].to(device)
        l2 = batch["l2_pixels"].to(device)
        labels = batch["true_label"].to(device)
        B = labels.size(0)
        break
</details>

### Inter-model transferability:
<details>
    <summary>On different fine-tuned ResNet18</summary>

w/o augmentation:

    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                  Clean: 91.12% (1458/1600)
                  Linf PGD: 0.69% (11/1600)
                  L2 PGD: 11.25% (180/1600)
    ============================================================

w/ augmentation, but with a different random seed (shuffle differently):

    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 98.75% (1580/1600)
                Linf PGD: 0.06% (1/1600)
                  L2 PGD: 3.25% (52/1600)
    ============================================================

pre-trained model, header-only trainable:

    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 76.06% (1217/1600)
                Linf PGD: 13.19% (211/1600)
                  L2 PGD: 16.88% (270/1600)
    ============================================================

randomly initialized for training:

    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 96.19% (1539/1600)
                Linf PGD: 94.50% (1512/1600)
                  L2 PGD: 95.50% (1528/1600)
    ============================================================
</details>


<details>
    <summary>On Swin Transformer-base-patch4-window7-224</summary>

    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 95.12% (1522/1600)
                Linf PGD: 81.88% (1310/1600)
                  L2 PGD: 89.81% (1437/1600)
    ============================================================
</details>

[SwinT link](https://huggingface.co/TyBruce/CIFAR10-subset-finetuned/blob/main/swin_base_patch4_window7_224_fft_augmentation.pth)

<details>
    <summary>On ViT-base-224</summary>
    
    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 99.69% (1595/1600)
                Linf PGD: 97.56% (1561/1600)
                  L2 PGD: 99.31% (1589/1600)
    ============================================================

</details>

[ViT link](https://huggingface.co/TyBruce/CIFAR10-subset-finetuned/blob/main/vit_base_patch16_224_fft_augmentation.pth)

### Cross-model distilation:
Downsampled at 32x32 pixels, the perturbation could fail.
<details>
    <summary>ResNet18 Student Learning based on CresNet Teacher Labels</summary>
    
    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 95.62% (1530/1600)
                Linf PGD: 96.06% (1537/1600)
                  L2 PGD: 96.31% (1541/1600)
    ============================================================
</details>


<details>
    <summary>ResNet18 Student Learning based on SwinT Teacher Labels</summary>
    
    ============================================================
    Overall Accuracy (N=1600)
    ============================================================
                   Clean: 96.12% (1538/1600)
                Linf PGD: 96.12% (1538/1600)
                  L2 PGD: 96.44% (1543/1600)
    ============================================================
</details>
