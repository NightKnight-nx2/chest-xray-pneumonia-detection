import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "chest_xray")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

    image_datasets_raw = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x if x != 'val' else 'val'])
        for x in ['train', 'val', 'test']
    }

    # ── Proper 80/20 train/val split from the training set ──────────────
    # The built-in 'val' folder has only 16 images → unreliable.
    # We split the full training set: 80% train, 20% validation.
    from torch.utils.data import Subset, random_split

    train_full = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), data_transforms['train']
    )
    val_full = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), data_transforms['val']
    )

    total = len(train_full)
    val_size = int(0.20 * total)
    train_size = total - val_size

    # Reproducible split
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(total), [train_size, val_size], generator=generator)

    train_dataset = Subset(train_full, train_idx.indices)
    val_dataset   = Subset(val_full,   val_idx.indices)
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    # Class weights from training targets
    all_train_targets = [train_full.targets[i] for i in train_idx.indices]
    num_normal    = all_train_targets.count(0)
    num_pneumonia = all_train_targets.count(1)
    pos_weight = torch.tensor([num_normal / num_pneumonia]).to(device)
    print(f"Class counts — NORMAL: {num_normal}, PNEUMONIA: {num_pneumonia}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True),
        'val':   DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True),
        'test':  DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True),
    }

    class_names = train_full.classes
    print(f"Classes: {class_names}")

    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


    # ── Model ────────────────────────────────────────────────────────────
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Replace FC
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1),
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training schedule ───────────────────────────────────────────────
    #  Phase 1 (Epoch 1-6):  Only FC layers  — high LR
    #  Phase 2 (Epoch 7-20): FC + layer4     — lower LR for base params
    PHASE1_EPOCHS = 6
    PHASE2_EPOCHS = 14
    TOTAL_EPOCHS  = PHASE1_EPOCHS + PHASE2_EPOCHS

    history = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}
    best_acc  = 0.0
    best_path = os.path.join(base_dir, 'best_pytorch_model.pth')

    def run_epoch(phase, optimizer):
        is_train = (phase == 'train')
        model.train() if is_train else model.eval()

        running_loss, running_corrects = 0.0, 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                preds = (outputs > 0.0).float()
                if is_train:
                    loss.backward()
                    optimizer.step()

            running_loss     += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc  = running_corrects.double() / len(image_datasets[phase])
        return epoch_loss, epoch_acc.item()

    # ════════════════════════════════════════════
    #  Phase 1 — FC only
    # ════════════════════════════════════════════
    print("\n" + "="*50)
    print(f"PHASE 1: FC-only training (Epochs 1-{PHASE1_EPOCHS})")
    print("="*50)

    optimizer_p1 = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode='max', factor=0.5, patience=2, verbose=True
    )

    for epoch in range(PHASE1_EPOCHS):
        ep = epoch + 1
        print(f"\nEpoch {ep}/{TOTAL_EPOCHS}")
        t_loss, t_acc = run_epoch('train', optimizer_p1)
        v_loss, v_acc = run_epoch('val',   optimizer_p1)
        scheduler_p1.step(v_acc)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f"  train → loss: {t_loss:.4f}  acc: {t_acc:.4f}")
        print(f"  val   → loss: {v_loss:.4f}  acc: {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best saved (acc={best_acc:.4f})")

    # ════════════════════════════════════════════
    #  Phase 2 — FC + layer4 unfrozen
    # ════════════════════════════════════════════
    print("\n" + "="*50)
    print(f"PHASE 2: layer4 + FC fine-tuning (Epochs {PHASE1_EPOCHS+1}-{TOTAL_EPOCHS})")
    print("="*50)

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Different LRs: base (layer4) gets 10× smaller LR than head
    optimizer_p2 = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(),     'lr': 1e-3},
    ])
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-6
    )

    for epoch in range(PHASE2_EPOCHS):
        ep = epoch + PHASE1_EPOCHS + 1
        print(f"\nEpoch {ep}/{TOTAL_EPOCHS}")
        t_loss, t_acc = run_epoch('train', optimizer_p2)
        v_loss, v_acc = run_epoch('val',   optimizer_p2)
        scheduler_p2.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f"  train → loss: {t_loss:.4f}  acc: {t_acc:.4f}")
        print(f"  val   → loss: {v_loss:.4f}  acc: {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best saved (acc={best_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")

    # ── Training Curves ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves (Gradual Unfreezing)", fontsize=14)

    axes[0].plot(history['train_acc'], label='Train Accuracy')
    axes[0].plot(history['val_acc'],   label='Val Accuracy')
    axes[0].axvline(x=PHASE1_EPOCHS - 0.5, color='orange', linestyle='--', label='Layer4 Unfrozen')
    axes[0].set_title('Accuracy vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'],   label='Val Loss')
    axes[1].axvline(x=PHASE1_EPOCHS - 0.5, color='orange', linestyle='--', label='Layer4 Unfrozen')
    axes[1].set_title('Loss vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'training_curves.png'), bbox_inches='tight')
    plt.close()
    print("Saved training curves.")

    # ── Evaluation on Test Set ───────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.0).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix (Gradual Unfreezing)', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix.")
    print("\nAll done!")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
