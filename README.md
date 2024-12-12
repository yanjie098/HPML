# HPML

https://api.wandb.ai/links/qt2118-new-york-university/1hjbbqx4

https://api.wandb.ai/links/yanjie98-new-york-university/0rxxoc9i

Framework
•	Deep Learning Framework: PyTorch
•	Model and Tokenization: Hugging Face transformers
•	Distributed Training: PyTorch DistributedDataParallel (DDP)
•	Logging and Experiment Tracking: Weights and Biases (wandb)
•	Dataset Handling: Hugging Face datasets
•	Progress Bar: tqdm
Dataset
•	The code uses the AG News dataset, which is a collection of news articles categorized into four classes.
•	The dataset is loaded using the Hugging Face datasets library and tokenized using the GPT-2 tokenizer.
Functionalities
1.	Initialization:
o	Sets up the environment for distributed training.
o	Initializes the model, tokenizer, and dataset.
o	Configures logging.
2.	Environment Setup:
o	Ensures the distributed environment variables are set.
3.	Dataset Preparation:
o	Loads and tokenizes the AG News dataset.
o	Converts the dataset into a format suitable for PyTorch.
4.	DataLoader Creation:
o	Creates DataLoaders with distributed sampling for training and testing.
5.	Training:
o	Handles the training loop, including forward and backward passes, loss calculation, and optimization.
o	Logs training progress and metrics to wandb.
o	Supports single-device and multi-device training.
6.	Evaluation:
o	Evaluates the trained model on the test dataset.
o	Calculates accuracy and inference time.
o	Logs evaluation metrics to wandb.
7.	Main Function:
o	Determines the available devices and initiates the training process accordingly.
Limitations
•	Hardware Dependency: Performance is highly dependent on the availability of GPUs. Training on CPU will be significantly slower.
•	Scalability: While the code supports multi-GPU training, it may require further optimization for large-scale distributed training across multiple nodes.
•	Dataset Specificity: The code is tailored for the AG News dataset. Adapting it to other datasets may require modifications in the dataset preparation and tokenization steps.
•	Hyperparameter Tuning: The code uses fixed hyperparameters. Optimal performance may require tuning these parameters based on the specific use case and dataset.


The following summarizes the comparative analysis of the fine-tuning optimization techniques:

**Accuracy**  

- **Prompt Tuning**  achieved the highest accuracy of **93.88%** , demonstrating superior adaptability to the task with minimal model modifications.
 
- **Adapter Tuning**  followed closely with **91.32%** , balancing efficiency and performance.
 
- **LoRA**  showed an accuracy of **90.73%** , slightly lower but within acceptable ranges for many applications.
  
**Runtime**

- **Prompt Tuning**  had a total runtime of **2345 seconds (39.1 minutes)**  for training, significantly higher than the other methods. However, it optimized task-specific parameters effectively.
 
- **LoRA**  achieved a much faster runtime of **725 seconds (12.1 minutes)** , reflecting its lightweight and efficient structure for quick fine-tuning.
 
- **Adapter Tuning**  required **732 seconds (12.2 minutes)** , comparable to LoRA but slightly longer due to the inclusion of task-specific modules.

**Memory Usage**  

- **LoRA**  and **Prompt Tuning**  used less memory, **3753 MB**  and **3577 MB** , respectively, making them ideal for resource-constrained scenarios.
 
- **Adapter Tuning**  had the highest memory consumption at **4259 MB** , attributed to the additional lightweight modules.

**Gradient Norm and Learning Stability**  

- **Adapter Tuning**  had the lowest gradient norm of **7.77** , indicating stable training dynamics.
 
- **LoRA**  and **Prompt Tuning**  had higher gradient norms (**9.43**  and **12.88** , respectively), reflecting a more aggressive optimization approach.

**Samples per Second**  

- **LoRA**  processed the highest number of samples per second at **165.42** , showcasing efficiency in data throughput.
 
- **Adapter Tuning**  processed slightly fewer samples per second at **163.73** .
 
- **Prompt Tuning**  had a lower throughput of **63.96 samples per second** , likely due to its focus on optimizing prompt embeddings.

**Loss**  

- Training loss was consistently low across all methods: 
  - **Prompt Tuning** : 0.2193 (smallest, indicating superior convergence).
 
  - **LoRA** : 0.27.
 
  - **Adapter Tuning** : 0.2644.


---

**Key Takeaways**  
- **Prompt Tuning**  is ideal for tasks demanding high accuracy, albeit with longer runtime and lower throughput.
 
- **LoRA**  strikes a balance between speed and memory efficiency, suitable for large-scale deployments with strict resource constraints.
 
- **Adapter Tuning**  provides a middle ground with slightly higher memory and runtime requirements but stable learning dynamics.
