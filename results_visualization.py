import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_noise_rate_comparison(results, output_folder="plots"):
    os.makedirs(output_folder, exist_ok=True)
    
    for noise_rate, data in results.items():
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(data['custom_train_acc'], label='Custom Model')
        plt.plot(data['baseline_train_acc'], label='Baseline Model', linestyle='--')
        plt.title(f'Training Accuracy (Noise {noise_rate})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(data['custom_val_acc'], label='Custom Model')
        plt.plot(data['baseline_val_acc'], label='Baseline Model', linestyle='--')
        plt.title(f'Validation Accuracy (Noise {noise_rate})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plot_path = os.path.join(output_folder, f"accuracy_plots_noise_{noise_rate}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")
    
    plt.figure(figsize=(8, 5))
    custom_final_acc = [data['custom_val_acc'][-1] for data in results.values()]
    baseline_final_acc = [data['baseline_val_acc'][-1] for data in results.values()]
    noise_rates = list(results.keys())
    
    plt.plot(noise_rates, custom_final_acc, marker='o', label='Custom Model')
    plt.plot(noise_rates, baseline_final_acc, marker='x', label='Baseline Model')
    plt.title('Final Validation Accuracy vs Noise Rate')
    plt.xlabel('Noise Rate')
    plt.ylabel('Final Validation Accuracy')
    plt.legend()
    
    combined_plot_path = os.path.join(output_folder, "final_validation_accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.close()
    print(f"Combined plot saved: {combined_plot_path}")
    
def plot_confusion_matrices(results, output_folder="plots"):
    os.makedirs(output_folder, exist_ok=True)
    
    noise_rates = list(results.keys())
    class_names = [str(i) for i in range(10)] 
    
    for noise_rate in noise_rates:
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        custom_cm = results[noise_rate]['custom_conf_matrix']
        sns.heatmap(custom_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Custom Model Confusion Matrix\n(Noise Rate: {noise_rate})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.subplot(1, 2, 2)
        baseline_cm = results[noise_rate]['baseline_conf_matrix']
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Baseline Model Confusion Matrix\n(Noise Rate: {noise_rate})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plot_path = os.path.join(output_folder, f"confusion_matrix_noise_{noise_rate}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion Matrix Plot saved: {plot_path}")

def save_performance_summary(performance_summary, output_folder="plots"):
    os.makedirs(output_folder, exist_ok=True)
    
    performance_path = os.path.join(output_folder, "model_performance_summary.csv")
    
    with open(performance_path, 'w') as f:
        f.write("Noise Rate,Custom Model F1,Custom Model Test Accuracy,Baseline Model F1,Baseline Model Test Accuracy\n")
        
        for noise_rate, metrics in performance_summary.items():
            f.write(f"{noise_rate},"
                    f"{metrics['custom_f1']:.4f},"
                    f"{metrics['custom_test_acc']:.4f},"
                    f"{metrics['baseline_f1']:.4f},"
                    f"{metrics['baseline_test_acc']:.4f}\n")
    
    print(f"Performance summary saved: {performance_path}")
    
    print("\n--- Performance Summary ---")
    for noise_rate, metrics in performance_summary.items():
        print(f"\nNoise Rate: {noise_rate}")
        print(f"Custom Model - F1 Score: {metrics['custom_f1']:.4f}, Test Accuracy: {metrics['custom_test_acc']:.4f}")
        print(f"Baseline Model - F1 Score: {metrics['baseline_f1']:.4f}, Test Accuracy: {metrics['baseline_test_acc']:.4f}")