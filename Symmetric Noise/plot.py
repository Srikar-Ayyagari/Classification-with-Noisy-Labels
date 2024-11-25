import matplotlib.pyplot as plt
import pandas as pd
import os

csv_file = "model_performance_summary.csv" 
data = pd.read_csv(csv_file)

noise_rates = data["Noise Rate"]
custom_model_acc = data["Custom Model Test Accuracy"]
baseline_model_acc = data["Baseline Model Test Accuracy"]

plt.figure(figsize=(8, 5))
plt.plot(noise_rates, custom_model_acc, marker='o', label='Custom Model')
plt.plot(noise_rates, baseline_model_acc, marker='x', label='Baseline Model')
plt.title('Final Test Accuracy vs Noise Rate')
plt.xlabel('Noise Rate')
plt.ylabel('Final Test Accuracy')
plt.legend()

plot_path = "test_accuracy_comparison.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
