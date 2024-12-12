# HPML

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
