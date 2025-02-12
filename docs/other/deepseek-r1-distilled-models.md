# DeepSeek-R1蒸馏模型

> DeepSeek-R1已经在多个基准测试中超越了 SOTA 推理模型 OpenAI-o1。在这篇文章中，我们将深入研究DeepSeek-R1的6 个蒸馏模型。

![DeepSeek-R1蒸馏模型](http://www.hubwiz.com/blog/content/images/size/w2000/2025/01/deepseek-r1-distilled-models.png)

DeepSeek 在 DeepSeek-V3 之后发布了另一个革命性的模型，即 DeepSeek-R1，这看起来像是一个重大版本，因为这个模型已经在多个基准测试中超越了 SOTA 推理模型 OpenAI-o1。

除了 DeepSeek-R1，该团队还发布了许多其他模型

- DeepSeek-R1-Zero：DeepSeek-R1 的原始版本，会犯错误但更具创造性
- DeepSeek-R1-Distill-Qwen 系列：1.5B、7B、14B、32B。
- DeepSeek-R1-Distill-Llama 系列：8B、70B。

在这篇文章中，我们将深入研究6 个蒸馏模型。

## 1、什么是蒸馏？

机器学习 (ML) 中的模型蒸馏是一种用于将知识从大型复杂模型（通常称为教师模型）转移到较小、更简单的模型（称为学生模型）的技术。

目标是创建一个较小的模型，该模型保留了较大模型的大部分性能，同时在计算资源、内存使用和推理速度方面更高效。

这对于在资源受限的环境（如移动设备或边缘计算系统）中部署模型特别有用。

## 2、DeepSeek-R1 蒸馏模型

DeepSeek-R1 蒸馏模型是较大的 DeepSeek-R1 模型的较小、更高效的版本，通过称为精炼的过程创建。精炼涉及将更大、更强大的模型（在本例中为 DeepSeek-R1）的知识和推理能力转移到较小的模型中。这使得较小的模型能够在推理任务上实现具有竞争力的性能，同时在计算上更高效且更易于部署。

由于 DeepSeek-R1 模型的大小非常大，即 671B 个参数，因此无法在消费级设备上运行，因此需要蒸馏模型。

### 2.1 蒸馏的目的

蒸馏的目的是让像 DeepSeek-R1 这样的大型模型的推理能力能够被更小、更高效的模型所利用。这对于有限的计算资源特别有用，但仍然需要较高的推理性能。

蒸馏后的模型旨在保留 DeepSeek-R1 发现的强大推理模式，即使它们的参数较少。
我想他们一定是注意到普通人无法使用 DeepSeek-V3，因为它的规模太大，所以才想到这次发布蒸馏版本。

### 2.2 蒸馏过程

蒸馏模型是通过使用 DeepSeek-R1 生成的 800,000 个推理数据样本对较小的基础模型（例如 Qwen 和 Llama 系列）进行微调而创建的。

蒸馏过程涉及对推理数据的监督微调 (SFT)，但不包括额外的强化学习 (RL) 阶段。这使得该过程更高效，也更易于小型模型使用。

### 2.3 蒸馏模型变体

本文开源了基于不同大小的 Qwen 和 Llama 架构的几个提炼模型。这些包括：

- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Llama-70B

### 2.4 蒸馏模型的性能

蒸馏模型在推理基准上取得了令人印象深刻的结果，通常优于 GPT-4o 和 Claude-3.5-Sonnet 等更大的非推理模型。

例如：

- DeepSeek-R1-Distill-Qwen-7B 在 AIME 2024 上实现了 55.5% Pass@1，超越了 QwQ-32B-Preview（最先进的开源模型）。
- DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 上实现了 72.6% Pass@1，在 MATH-500 上实现了 94.3% Pass@1，显著优于其他开源模型。
- DeepSeek-R1-Distill-Llama-70B 在 AIME 2024 上实现了 70.0% Pass@1，在 MATH-500 上实现了 94.5% Pass@1，创下了密集模型的新纪录。

### 2.5 蒸馏模型的优势

- 效率：蒸馏模型比原始 DeepSeek-R1 更小，计算效率更高，使其更易于在资源受限的环境中部署。
- 推理能力：尽管规模较小，但由于从 DeepSeek-R1 转移的知识，蒸馏模型仍保留了强大的推理能力。
- 开源可用性：提炼模型是开源的，允许研究人员和开发人员在各种应用中使用和构建它们。

### 2.6 与 RL 训练模型的比较

本文将提炼模型与使用大规模 RL 训练的模型（例如 DeepSeek-R1-Zero-Qwen-32B）进行了比较，发现蒸馏通常能以更低的计算成本获得更好的性能。

例如，DeepSeek-R1-Distill-Qwen-32B 在推理基准测试中的表现优于 DeepSeek-R1-Zero-Qwen-32B，表明蒸馏是一种更经济、更有效的小型模型方法。

## 3、使用 DeepSeek-R1 蒸馏模型

可以使用Ollama或vLLM等本地LLM工具来运行DeepSeek-R1蒸馏模型。

### 3.1 使用 Ollama

![img](http://www.hubwiz.com/blog/content/images/2025/01/image-477.png)

### 3.2 使用 vLLM

```
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
```

## 4、结束语

DeepSeek-R1 蒸馏模型弥合了高性能和效率之间的差距，使更广泛的受众能够使用高级推理功能。此版本标志着AI民主化和实现尖端推理模型的实际应用迈出了重要一步。

如果你使用的是消费级 PC，我建议你尝试精简模型，因为原始 R1 模型非常庞大，可能放不下。