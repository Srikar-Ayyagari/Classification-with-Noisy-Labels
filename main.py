import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

from data_processing import *
from model_components import *
from results_visualization import *

import argparse

def prepare_data(noise_rate=0.4, noise_type='symmetric'):
    global train_loader, val_loader, test_loader
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataset = add_noise_to_dataset(train_dataset, noise_rate=noise_rate, noise_type=noise_type)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def train_noise_corrected_model(device='cuda', epochs=60, noise_rate=0.4, noise_type='symmetric', learning_rate=1e-3):
    global train_loader, val_loader, test_loader
    prepare_data(noise_rate=noise_rate, noise_type=noise_type)

    model = CIFAR10MultivariateModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    helmert_matrix = create_helmert_matrix(num_classes=10)

    label_correction_ema = LabelCorrectionEMA(model, device)

    train_accuracies, val_accuracies = [], []
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        running_train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                smoothed_one_hot_labels = batch_label_smoothing(labels, num_classes=10)
                ilr_labels = ilr_transform_batch(smoothed_one_hot_labels, helmert_matrix)

                optimizer.zero_grad()
                mu, r = model(inputs)

                if epoch > 20:
                    label_shift = label_correction_ema.compute_shift(epoch, inputs, ilr_labels, mu)
                    loss = enhanced_multivariate_gaussian_loss(ilr_labels, mu, r, shift=label_shift)
                else:
                    loss = enhanced_multivariate_gaussian_loss(ilr_labels, mu, r)
                
                loss.backward()
                optimizer.step()
                label_correction_ema.update()

                predicted_one_hot = inverse_ilr_transform_batch(mu, helmert_matrix)
                _, predicted = torch.max(predicted_one_hot, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_accuracy = 100. * correct / total
                
                running_train_loss += loss.item()
                pbar.set_postfix({'Train loss': loss.item(), 'Train Accuracy': train_accuracy})
                pbar.update(1)
        
        train_accuracies.append(train_accuracy)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mu, r = model(inputs)
                predicted_one_hot = inverse_ilr_transform_batch(mu, helmert_matrix)
                _, predicted = torch.max(predicted_one_hot, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100. * correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    model.eval()
    true_labels, model_preds = [], []
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            mu, _ = model(inputs)
            predicted_one_hot = inverse_ilr_transform_batch(mu, helmert_matrix)
            _, model_pred = torch.max(predicted_one_hot, 1)
            test_total += labels.size(0)
            test_correct += (model_pred == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            model_preds.extend(model_pred.cpu().numpy())
    model_test_accuracy = 100. * test_correct / test_total
    model_f1 = f1_score(true_labels, model_preds, average='macro')
    model_conf_matrix = confusion_matrix(true_labels, model_preds)
    return train_accuracies, val_accuracies, model_f1, model_test_accuracy, model_conf_matrix

def train_baseline_model(device='cuda', epochs=60, noise_rate=0.4, learning_rate=1e-3):
    global train_loader, val_loader, test_loader

    baseline_model = models.resnet18(pretrained=False)
    baseline_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    baseline_model.maxpool = nn.Identity()
    baseline_model.fc = nn.Linear(baseline_model.fc.in_features, 10)
    baseline_model = baseline_model.to(device)

    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        baseline_model.train()
        correct, total = 0, 0
        running_train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = baseline_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_accuracy = 100. * correct / total
                
                running_train_loss += loss.item()
                pbar.set_postfix({'Train loss': loss.item(), 'Train Accuracy': train_accuracy})
                pbar.update(1)
        
        train_accuracies.append(train_accuracy)

        baseline_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = baseline_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100. * correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    baseline_model.eval()
    true_labels, model_preds = [], []
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = baseline_model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            true_labels.extend(labels.cpu().numpy())
            model_preds.extend(predicted.cpu().numpy())

    baseline_test_accuracy = 100. * test_correct / test_total
    baseline_f1 = f1_score(true_labels, model_preds, average='macro')
    baseline_conf_matrix = confusion_matrix(true_labels, model_preds)

    return train_accuracies, val_accuracies, baseline_f1, baseline_test_accuracy, baseline_conf_matrix

def run_experiments(device, noise_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], noise_type='symmetric'):
    results = {}
    performance_summary = {}
    
    for noise_rate in noise_rates:
        print(f"\n--- Experiment with Noise Rate: {noise_rate} ---")
        
        custom_train_acc, custom_val_acc, custom_f1, custom_test_acc, custom_conf_matrix = train_noise_corrected_model(
            device=device, 
            epochs=60, 
            noise_rate=noise_rate,
            noise_type=noise_type
        )
        
        baseline_train_acc, baseline_val_acc, baseline_f1, baseline_test_acc, baseline_conf_matrix = train_baseline_model(
            device=device, 
            epochs=60, 
            noise_rate=noise_rate
        )
        
        results[noise_rate] = {
            'custom_train_acc': custom_train_acc,
            'custom_val_acc': custom_val_acc,
            'baseline_train_acc': baseline_train_acc,
            'baseline_val_acc': baseline_val_acc,
            'custom_conf_matrix': custom_conf_matrix,
            'baseline_conf_matrix': baseline_conf_matrix
        }
        
        performance_summary[noise_rate] = {
            'custom_f1': custom_f1,
            'custom_test_acc': custom_test_acc,
            'baseline_f1': baseline_f1,
            'baseline_test_acc': baseline_test_acc
        }
    
    plot_noise_rate_comparison(results)
    plot_confusion_matrices(results)
    save_performance_summary(performance_summary)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified configurations.")
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the experiments on (e.g., 'cpu' or 'cuda')."
    )
    parser.add_argument(
        "--noise_rates",
        type=float,
        nargs="+",
        default=[0.4, 0.6],
        help="List of noise rates to use in the experiments."
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="symmetric",
        help="Type of noise to use (e.g., 'symmetric' or 'asymmetric')."
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    run_experiments(device, noise_rates=args.noise_rates, noise_type=args.noise_type)