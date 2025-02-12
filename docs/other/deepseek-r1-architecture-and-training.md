# DeepSeek R1架构和训练过程图解

> 为了让一切变得简单，我们将使用手绘流程图和简单的计算来帮助从头开始澄清DeeoSeek-R1的核心概念。

![DeepSeek R1架构和训练过程图解](http://www.hubwiz.com/blog/content/images/size/w2000/2025/02/deepseek-r1-architecture-and-training-process.png)

如果你对 AI 感兴趣，可能听说过 DeepSeek R1。它目前在 LLM 领域很流行，并且表现优于开源和闭源模型。

为了让一切变得简单，我们将使用手绘流程图和简单的计算来帮助从头开始澄清DeeoSeek-R1的核心概念。

事实上，我们将在整个博客中使用字符串 2 + 3 * 4 等于多少？作为示例，引导你了解 DeepSeek 技术报告的每个组成部分。

## 1、快速概览

因此，在介绍技术细节之前，快速概览一下：DeepSeek-R1 不是从头开始训练的，就像从无到有一样。相反，他们从一个非常聪明的 LLM 开始，他们已经有了 DeepSeek-V3，但他们想让它成为推理超级明星。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-182.png)DeepSeek 实施快速概览

为此，他们使用了强化学习（简称 RL），当 LLM 做出有利于推理的事情时，你会奖励它，否则会惩罚它。

但这不仅仅是一个简单的训练课程。它就像是一整套步骤，他们称之为管道。他们首先尝试了纯 RL，看看推理是否会自行出现，这就是 DeepSeek-R1-Zero，有点像一个实验。然后对于真正的 DeepSeek-R1，他们通过不同的阶段使其更有条理。他们给它一些起始数据来启动它，然后进行 RL，然后是更多数据，然后是更多 RL……就像一步一步升级一样！

整个重点是让这些语言模型更好地思考问题并为你提供明智的答案，而不仅仅是吐出单词。

所以，是的，在我们研究每个步骤的疯狂细节之前，这是超简短的版本。

## 2、DeepSeek V3 (MOE) 如何思考？

正如我之前所说，DeepSeek R1 训练不是从头开始构建的，而是使用 DeepSeek V3 作为基础模型。那么我们需要了解 V3 的工作原理以及它为什么被称为 MOE？

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-183.png)DeepSeek V3 架构

DeepSeek V3 有两条主要路径。当你输入问题时，它首先会经过一个记忆系统，该系统通过查找相关信息快速构建上下文。可以将其视为快速回忆你之前遇到过的类似情况。

它的主要优势在于其决策系统。在理解你的输入后，它会使用一个智能路由器在两条路径之间做出决定：快速处理器用于简单的任务（如简单问题或常见请求）和专家系统用于复杂的问题（如分析或专业知识）。

> 这个路由器使 DeepSeek V3 成为专家混合模型 (MOE)

因为它会动态地将每个请求定向到最合适的专家组件以进行高效处理。

简单的问题通过快速路径获得快速、直接的答案，而复杂的查询则通过专家系统得到详细关注。最后，这些响应被组合成清晰、准确的输出。

## 3、DeepSeek V3 作为 RL 设置中的策略模型

现在我们已经了解了 DeepSeek v3 的思考方式，它是 DeepSeek R1 实现的起点，我所说的起点是指它已经创建了 DeepSeek R1 Zero版本，这是一个在创建最终版本之前存在一些错误的初始版本。

初始版本（R1 Zero）是使用强化学习创建的，其中 DeepSeek v3 充当 RL 代理（采取行动的参与者）。让我们首先直观地了解它的工作原理。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-184.png)V3 作为代理工作流程

RL 代理（DeepSeek V3）首先采取行动，这意味着它会为放入其环境中的给定问题生成答案和一些推理。在这种情况下，环境只是推理任务本身。

采取行动后，环境会给予奖励。这个奖励就像反馈，它告诉 DeepSeek V3 基础模型其行动有多好。积极的奖励意味着它做对了某件事，可能得到了正确的答案或推理得很好。然后，这个反馈信号返回到 DeepSeek-V3-Base，帮助它学习和调整未来如何采取行动以获得更好的奖励。

在接下来的部分中，我们将讨论这个带有奖励模型的 RL 设置以及它们使用的 RL 算法并尝试使用我们的文本输入来解决它。

## 4、GRPO 算法如何工作？

训练 LLM 的计算成本极高，而 RL 则增加了更多的复杂性。

因此，有许多可用的 RL 算法，但传统的 RL 使用一种称为“批评家”的东西来帮助主要决策部分（“参与者”，即 DeepSeek V3），正如你已经知道的那样。这个批评家通常和参与者本身一样大和复杂，这基本上使计算成本翻倍。

然而，GRPO 的做法不同，因为它直接从一组动作的结果中找出基线，即一种良好动作的参考点。因此，GRPO 根本不需要单独的批评家模型。这节省了大量计算并使事情变得更有效率。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-185.png)GRPO 算法

它从向模型提出一个问题或提示开始，称为“旧策略”。 GRPO 不会只得到一个答案，而是指示旧策略针对同一问题生成一组不同的答案。然后评估每个答案并给出奖励分数，以反映其好坏程度或可取性。

GRPO 通过将每个答案与其组中其他答案的平均质量进行比较来计算每个答案的“优势”。高于平均水平的答案获得正优势，而低于平均水平的答案获得负优势。至关重要的是，这无需单独的批评模型即可完成。

然后使用这些优势分数来更新旧策略，使其更有可能在未来产生高于平均水平的答案。这个更新的模型成为新的“旧策略”，这个过程重复进行，迭代地改进模型。

## 5、GRPO 的目标函数

显然，在这个 GRPO 背后，有复杂的数学💀总之，我们可以称之为 GRPO 背后的目标函数。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-186.png)GRPO 目标

GRPO 的目标函数有两个目标，一个是给出良好的输出（高回报），同时确保训练过程稳定且不会失控。原始函数很吓人，但我们会将其重写为更简单的形式，而不会失去其实际意义。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-187.png)目标函数（原始与简化）

让我们逐一分解。首先， `AverageResult[…]` 或 `1/n[…]` 指的是评估在许多不同情况下平均发生的情况。我们提出一个包含各种问题的模型。对于每个问题，模型都会生成一组答案。通过查看许多问题及其各自的答案组中的这些答案，我们可以计算出平均结果。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-188.png)平均的含义

在此过程中，问题被输入到旧模型中，该模型会产生多个答案（例如，答案 1、答案 2、...、答案 G）。这些答案形成一个组，通过评估不同问题中的这个组，我们得出平均结果。

`SumOf[..]` 或 `∑[…]` 指的是对一组答案（例如，答案 1、答案 2、...、答案 G）中的每个答案进行计算，然后将所有这些计算的结果加在一起。

然后是奖励部分。这是对给出良好答案的模型进行奖励的部分。它内部有点复杂，让我们放大一下：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-189.png)公式

`ChangeRatio` 告诉我们使用新模型给出此答案的几率是增加还是减少。具体来说，它着眼于：

- 使用新模型的答案几率：新模型给出特定答案的可能性有多大。
- 使用旧模型的答案几率：旧模型给出相同答案的可能性有多大。

接下来，优势 ( `Advantage`) 分数表明与同一组中的其他答案相比，答案有多好或多差。它的计算方式如下：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-190.png)优势公式

- 答案的分数：给予特定答案的奖励。
- 组的平均分数：组中所有答案的平均奖励分数。
- 组中分数的分布：组中答案分数的差异有多大。

`Advantage` 分数告诉我们答案是否优于该组内的平均水平，以及好多少。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-191.png)LimitedChangeRatio 公式（

`LimitedChangeRatio` 是 `ChangeRatio` 的修改版本。它确保 `ChangeRatio` 不会波动太大，从而保持模型的学习稳定。限制由一个称为 `Epsilon` 的小值决定， `h` 确保变化不会太剧烈。

最后， `SmallerOf[ … ]` 函数在两个选项中选择较小的值：

- `ChangeRatio × Advantage`：答案可能性的变化，乘以其优势分数。
- `LimitedChangeRatio × Advantage`：相同，但变化率有限。

通过选择较小的值，模型可确保学习过程保持平稳，不会对性能的大幅变化反应过度。结果是“好答案奖励”，鼓励模型改进而不会过度补偿。

最后，我们减去 StayStablePart。这是为了防止新模型发生太大的变化。它不是很复杂，但让我们放大它：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-192.png)StayStable 方程

`DifferenceFromReferenceModel` 测量新模型与参考模型（通常是旧模型）的差异。本质上，它有助于评估新模型与前一个模型相比所做的更改。

`Beta` 值控制模型应保持与参考模型的接近程度。较大的 `Beta` 意味着模型将优先保持更接近旧模型的行为和输出，以防止偏差太大。让我们将其可视化：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-193.png)StayStable 的视觉表示

简而言之， `StayStablePart` 确保模型逐渐学习并且不会疯狂跳跃。

## 6、DeepSeek R1 Zero 的奖励建模

现在我们已经了解了主要的理论概念，让我们使用我们的文本输入来了解创建 R1 Zero 的奖励建模是如何工作的。

请记住，对于 R1 Zero，他们保持简单直接。他们没有使用花哨的神经网络来判断答案（就像他们在后期阶段可能会做的那样），而是使用了基于规则的奖励系统。

对于我们的数学问题：“2 + 3 * 4 等于多少？”

### 6.1 基于规则的检查

系统知道正确答案是 14。它将查看 DeepSeek V3（我们的 RL 代理）生成的输出，并专门检查  `<answer>`标签内的内容。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-194.png)基于规则的检查

如果  `<answer>`标签包含“14”（或数字相同的内容），它会得到正奖励，比如说 +1。如果它错了，它会得到 0 奖励，甚至可能是负奖励（尽管本文在这个阶段为了简单起见重点关注 0）。

### 6.2 格式化奖励

但 DeepSeek R1 Zero 还需要学习正确构建其推理，并且可以使用 `<think>` 和 `<answer>` 标签，正确设置格式的奖励较少。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-195.png)格式奖励过程

检查模型输出是否正确地将推理过程包含在 `<think> …</think>` 中，并将最终答案包含在 `<answer>… </answer>`中。

> DeepSeek R1 论文明确提到避免使用 DeepSeek-R1-Zero 的神经奖励模型，以防止奖励黑客攻击并降低初始探索阶段的复杂性

## 7、奖励训练模板

为了使奖励模型有效，研究人员设计了一个特定的训练模板。该模板充当蓝图，指导 DeepSeek-V3-Base 如何在强化学习过程中构建其响应。

让我们看看原始模板并将其逐一分解：

```
A conversation between User and Assistant. The user asks a question, and 
the Assistant solves it. The assistant first thinks about the reasoning 
process in the mind and then provides the user with the answer. The reasoning 
process and answer are enclosed within <think> </think> and <answer> </answer>
tags respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {prompt}. Assistant:
```

> 翻译：
> 用户和助手之间的对话。用户提出问题，助手解决该问题。助手首先在脑海中思考推理过程，然后为用户提供答案。推理过程和答案分别包含在  `<think></think>` 和 `<answer></answer>`  标签中，即 ` <think>推理过程在这里</think>  <answer>答案在这里</answer>` 。用户：{prompt}。助手：

我们在 `{prompt}` 中插入数学问题，例如 `2 + 3 * 4 等于多少？`。重要的是那些  和  标签。这种结构化输出对于研究人员以后窥视模型的推理步骤非常重要。

当我们训练 DeepSeek-R1-Zero 时，我们使用此模板为其提供提示。对于我们的示例问题，输入将如下所示：

```
A conversation between User and Assistant. The user asks a question, and 
the Assistant solves it. The assistant first thinks about the reasoning 
process in the mind and then provides the user with the answer. The reasoning 
process and answer are enclosed within <think> </think> and <answer> </answer>
tags respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: What is 2 + 3 * 4?. Assistant:
```

> 翻译：
> 用户和助手之间的对话。用户提出问题，助手解决该问题。助手首先在脑海中思考推理过程，然后为用户提供答案。推理过程和答案分别包含在  `<think></think> 和 <answer></answer>  标签中，即  <think>推理过程在这里</think>  <answer>答案在这里</answer> `。用户：2+3*4等于多少？。助手：

我们期望模型生成符合模板的输出，例如：

```
<think>
Order of operations:
multiply before add. 3 * 4 = 12. 2 + 12 = 14
</think>
<answer>
14
</answer>
```

```
翻译：
<think>
运算顺序：先乘后加。3 * 4 = 12。2 + 12 = 14 14
<think>
<answer>
14
</answer>
```

有趣的是，DeepSeek 团队有意保持这个模板简单并专注于结构，而不是告诉模型如何推理。

## 8、DeepSeek R1 Zero 的强化学习训练过程

尽管本文没有指定强化学习预训练的确切初始数据集，但我们假设它应该以推理为重点。

他们所做的第一步是使用旧策略（即强化学习更新之前的 DeepSeek-V3-Base 模型）生成多个可能的输出。在一次训练迭代中，我们假设 GRPO 采样一组 G = 4 个输出。

例如，该模型为我们的文本输入生成以下四个输出 2 + 3 * 4 等于多少？

- o1：  `<think>2 + 3 = 5, 5 * 4 = 20</think>  <answer> 20</answer>` （运算顺序不正确）
- o2： `<think>3 * 4 = 12, 2 + 12 = 14</think> <answer>14</answer>` （正确）
- o3： `<answer>14</answer>` （正确，但缺少 `<think>`标签）
- o4： `<think>...一些胡言乱语的推理...</think>  <answer> 7<answer>` （不正确且推理不佳）

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-196.png)生成输出

每个输出将根据正确性和推理质量进行评估并分配奖励。

为了引导模型进行更好的推理，基于规则的奖励系统应运而生。每个输出都根据以下条件分配奖励：

- 准确度奖励：答案是否正确。
- 格式奖励：推理步骤是否使用  标签正确格式化。

假设奖励分配如下：

| 输出                 | 准确率奖励 | 格式奖励 | 总奖励 |
| -------------------- | ---------- | -------- | ------ |
| o1（推理错误）       | 0          | 0.1      | 0.1    |
| o2（推理正确）       | 1          | 0.1      | 1.1    |
| o3（正确但缺少标签） | 1          | 0        | 1.0    |
| o4（推理错误且较差） | 0          | 0.1      | 0.1    |

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-197.png)奖励细分

模型应该学会偏爱奖励更高的输出，同时降低生成不正确或不完整输出的概率。

为了确定每个输出对模型性能的改善或恶化程度，我们使用奖励值计算优势。优势有助于通过强化更好的输出来优化策略。

为此，让我们计算平均第一个奖励。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-198.png)平均奖励计算

标准差（近似值）= 0.5，现在计算每个输出的优势。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-199.png)计算每个输出的奖励

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-200.png)可视化优势计算

输出 o2 和 o3 获得正优势，这意味着应该鼓励它们。输出 o1 和 o4 获得负优势，这意味着应该阻止它们。

然后，GRPO 使用计算出的优势来更新策略模型 (DeepSeek-V3-Base)，以增加生成具有高优势的输出（如 o2 和 o3）的概率，并降低具有低优势或负优势的输出（如 o1 和 o4）的概率。

更新根据以下内容调整模型权重：

- 策略比率：在新策略与旧策略下生成输出的概率。
- 裁剪机制：防止过大的更新，这可能会破坏训练的稳定性。
- KL 发散惩罚：确保更新不会偏离原始模型太远。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-201.png)GRPO 工作

这确保在下一次迭代中，模型更有可能生成正确的推理步骤，同时减少不正确或不完整的响应。

因此，RL 是一个迭代过程。使用不同的推理问题重复上述步骤数千次。每次迭代都会逐渐提高模型的能力：

- 执行正确的操作顺序
- 提供逻辑推理步骤
- 始终使用正确的格式

整体训练循环如下所示：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-202.png)DeepSeek 简化训练过程

随着时间的推移，模型会从错误中吸取教训，在解决推理问题方面变得更加准确和有效。 🚀

## 9、R1 Zero 的两个主要问题

在 V3 模型上使用 RL 训练过程创建 DeepSeek-R1 Zero 后，研究人员发现训练后的模型在推理测试中表现非常出色，甚至在 AIME 2024 等任务上的得分与 OpenAI-01-0912 等更高级的模型相似。这表明使用强化学习 (RL) 来鼓励语言模型中的推理是一种很有前途的方法。

但他们也注意到 DeepSeek-R1-Zero 有一些关键问题需要解决，才能在现实世界中使用并进行更广泛的研究。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-203.png)R1 Zero 的问题（

DeepSeek 的研究人员表示，该模板有意简单且结构集中。它避免对推理过程本身施加任何特定于内容的限制。例如，它没有说：

- “你必须使用分步推理”（它只是说“推理过程”，让模型来定义它的含义）。
- “你必须使用反思性推理”
- “你必须使用特定的问题解决策略”

主要问题是  标签内的推理过程难以阅读，使人类难以理解和分析。

另一个问题是语言混合，当被问到多语言问题时，模型有时会在同一个回答中混合使用多种语言，导致输出不一致和混乱。如果你用西班牙语问它问题。突然间，它的“思维”就会变成英语和西班牙语的混杂，不太完美！这些问题，混乱的推理和语言混乱，是明显的障碍。

> 这是他们将最初的 R1 Zero 模型转变为 R1 的两个主要原因

在下一节中，我们将介绍他们如何将 R1 zero 模型改进为 R1 模型，从而提高其性能并帮助其胜过所有其他模型（无论是开源的还是封闭的）。

## 10、冷启动数据

因此，为了修复 R1 Zero 问题并真正让 DeepSeek 推理正确，研究人员进行了冷启动数据收集并包括监督微调。

你可以将其视为在真正激烈的 RL 训练之前为模型提供良好的推理基础。基本上，他们想教 DeepSeek-V3 Base 良好的推理是什么样子以及如何清晰地呈现它。

### 10.1 使用长 CoT 进行少量提示

他们为 DeepSeek-V3 Base 提供了一些问题示例以及非常详细的分步解决方案，称为思维链 (CoT)。这个想法是让模型通过示例学习并开始模仿这种分步推理风格。

让我们直观地了解这个基于示例的学习：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-204.png)基于示例的学习

对于我们的示例问题  `2 + 3 * 4 等于多少？`，他们可能会显示如下提示：

```
Problem Examples with Solutions:

Problem: What's the square root of 9 plus 5?
Solution: | special_token | First, find the square root of 9, which is 3. 
Then, add 5 to 3.  3 + 5 equals 8. | special_token | Summary: The answer is 8.

Problem: Train travels at 60 mph for 2 hours, how far?
Solution: | special_token | Use the formula: Distance = Speed times Time. 
Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. 
| special_token | Summary: Train travels 120 miles.

Problem: What is 2 + 3 * 4?
Solution:
```

那些 `| special_token |` 东西只是将推理步骤与摘要分开的标记，使模型可以清楚地学习结构。

看到这些示例后，模型应该学会以类似的格式给出答案，例如 `2 + 3 * 4 等于多少？`

```
| special_token |  Following order of operations (PEMDAS/BODMAS), 
do multiplication before addition.  So, first calculate 3 * 4 = 12. 
Then, add 2 to 12. 2 + 12 = 14. | special_token | Summary: The answer is 14.
```

### 10.2 直接提示

他们收集数据的另一种方法是直接提示模型不仅解决问题，而且还要逐步明确地展示其推理，然后仔细检查其答案。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-205.png)直接提示

这是为了促进更仔细和周到的问题解决。

对于 2 + 3 * 4 等于多少？，提示可能是：

```
Problem: Solve this, show reasoning step-by-step, and verify:
What is 2 + 3 * 4?
```

他们确实期望输出包含推理步骤和验证部分：

```
| special_token | Reasoning: To solve 2 + 3 * 4, I need to use order of 
operations. Multiplication comes before addition.
Step 1: Calculate 3 * 4 = 12.
Step 2: Add 2 to the result from step 1: 2 + 12 = 14.
Verification:  Checking order of operations again, yes, multiplication 
is before addition.  Calculation looks right.
| special_token | Summary: The answer is 14.
```

### 10.3 后处理细化

他们甚至使用了已经训练过的 R1 Zero 模型的输出。尽管 R1 Zero 存在问题，但它可以进行一些推理。因此，他们采用了 R1 Zero 的输出，并让人类标注者使其更清晰、更结构化，并纠正任何错误。

例如，混乱的 R1 Zero 输出可能是：

```
<think>  ummm... multiply 3 and 4... get 12... then add 2...</think>
<answer> 14 </answer>
```

然后人类会对其进行改进，使其更清晰、格式更好：

```
| special_token | Reasoning: To solve this, we use order of operations,
doing multiplication before addition.
Step 1: Multiply 3 by 4, which is 12.
Step 2: Add 2 to the result: 2 + 12 = 14.
| special_token | Summary: The answer is 14.
```

可视化细化过程的工作原理如下：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-207.png)细化过程

他们最终获得的冷启动数据非常好，因为：

- 高质量推理示例：每个示例都展示了良好的逐步推理。
- 一致、可读的格式： `| special_token |` 格式使所有内容统一且易于处理。
- 人工检查：他们确保过滤掉任何不好的例子，因此数据干净可靠。

获得此冷启动数据后，他们进行了监督微调 (SFT)。

## 11、监督微调

SFT 第 1 阶段的核心思想是使用监督学习来教 DeepSeek-V3-Base 如何产生高质量、结构化的推理输出。

基本上，我们向模型展示了许多良好推理的例子，并要求它学习模仿这种风格。

对于 SFT，我们需要将冷启动数据格式化为输入-目标对。对于数据集中的每个推理问题，我们都会创建一个这样的对：

输入 = 提示或问题描述本身

```
User: What is 2 + 3 * 4? Assistant:
```

这是我们输入到模型中的内容，我们的目标是相应的结构良好的推理和答案

```
| special_token | According to the order of operations (PEMDAS/BODMAS) ... 
Summary: The answer is 14.
```

这是我们希望模型学习生成的理想输出。

我们告诉模型：

> 当你看到此输入（问题）时，我们希望你产生此目标输出（良好的推理和答案）

与其用详细的文字解释并让你难以理解，不如先将其可视化，以便更容易解释 SFT

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-208.png)SFT 流程

微调过程从输入开始：提示 + 目标推理，我们在此提供一个问题和一个结构化的推理示例。这会训练模型（DeepSeek-V3-Base 模型）以生成结构良好的响应。

在预测下一个标记中，模型会生成推理序列中的下一个单词。使用损失函数将其与比较目标标记（计算损失）中的实际下一个标记进行比较。损失越大，意味着预测距离正确标记越远。

在更新模型参数中，反向传播和优化器会调整模型的权重以

改进其预测。这个过程循环往复，重复许多输入目标对，每次迭代逐渐提高模型结构化推理能力。

## 12、推理导向强化学习

他们已经为 DeepSeek V3 提供了 SFT 推理教育，但为了真正提高其推理能力，研究人员引入了推理导向学习！

在这里，我们采用 SFT 微调的 DeepSeek-V3 模型，并通过强化学习推动它变得更好。

他们确实使用了相同的 GRPO 算法，但这一阶段真正的升级是奖励系统。他们添加了一些新的、非常重要的语言一致性奖励！

还记得 R1 Zero 有时会对语言感到困惑并开始混淆它们吗？为了解决这个问题，他们专门增加了保持语言一致性的奖励。这个想法很简单，如果你用英语问问题，我们希望推理和答案也是用英语。

让我们直观地了解一下这个语言一致性奖励计算：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-209.png)一致性奖励计算

为了理解上面的图表，让我们重新回顾之前的示例输出 o1 和 o2，看看奖励如何随着这个新的语言一致性奖励而变化。为简单起见，我们假设目标语言是英语。

让我们看看这些奖励如何与我们的示例输出一起发挥作用。考虑第一个输出 o1，它错误地计算了“2 + 3 * 4”，但用英语呈现了其有缺陷的推理：

```
<think> 2 + 3 = 5, 5 * 4 = 20 </think> <answer> 20 </answer>
```

对于这个，准确度奖励自然是 0，因为答案是错误的。但是，由于假设推理 100% 使用目标语言（本例中为英语），因此它获得的语言一致性奖励为 1。

当我们计算 RL 阶段的总奖励时，我们会将这些结合起来。如果我们为准确度奖励分配权重 1，为语言一致性奖励分配较小的权重（例如 0.2），则 o1 的总奖励变为：

```
Total Reward = (1 * Accuracy Reward) + (0.2 * Language Consistency Reward)

(1 * 0) + (0.2 * 1) = 0.2
```

现在考虑输出 o2，它正确解决了问题并且还用英语进行推理：

```
<think> 3 * 4 = 12, 2 + 12 = 14 </think> <answer> 14 </answer>
```

此输出因正确答案而获得完美的准确度奖励 1。假设它的推理也是 100% 英语，它也会获得 1 的语言一致性奖励。使用与之前相同的权重，o2 的总奖励为：

```
(1 * 1) + (0.2 * 1) = 1.2
```

请注意，语言一致性奖励如何略微提高正确答案的总奖励，甚至为错误答案 o1 提供小幅正奖励，只要它保持语言一致性。

此 RL 训练循环遵循我们之前看到的相同 DeepSeek R1 Zero 训练循环：

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-210.png)推理导向循环

- 生成多个输出。
- 细化奖励，包括语言一致性。
- 使用 GRPO 进行优势估计。
- 训练模型以支持高优势输出。
- 重复该过程！

## 13、拒绝抽样

对于推理数据，DeepSeek 团队希望获得绝对最佳示例以进一步训练模型。为此，他们使用了一种称为拒绝抽样的技术。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-211.png)拒绝抽样

为了改进推理数据，DeepSeek 使用了拒绝抽样。对于“2 + 3 * 4 等于多少？”，他们会从上一阶段模型生成许多输出。想象一下得到 20（错误）和 14 …（正确，推理）这样的输出。

然后他们会评估每个输出的正确性（答案“14”）和推理的可读性。只有正确且推理充分的最佳输出才会被保留，而其他输出则被拒绝。

对于复杂的推理，生成奖励模型用于判断推理质量。严格的过滤器会删除混合语言、漫无边际的推理或不相关的代码。此过程会产生约 600k 个高质量推理样本。

除了精炼推理数据外，他们还添加了非推理数据（约 20 万个样本），用于一般技能：写作、问答、翻译等，有时还会使用思维链来完成复杂任务。

最后，SFT 第 2 阶段使用下一个标记预测在组合数据集（精炼推理 + 非推理）上训练前一个模型检查点。此阶段使用来自拒绝采样的顶级示例进一步改进推理，并将模型推广到更广泛的任务，同时保持用户友好性。

“2 + 3 * 4 等于多少？”现在是一个完美精炼的推理示例，成为此训练数据的一部分。

> 这是拒绝采样，我们拒绝低于标准的样本，只保留最好的样本以生成高质量的训练数据

## 14、适用于所有场景的 RL

在 SFT 第 2 阶段之后，我们获得了 DeepSeek V3 推理、一致说话，甚至很好地处理了一般任务！但要真正使其成为顶级的人工智能助手，研究人员必须与人类价值观进行最后的调整。这就是强化学习在所有场景中的使命（强化学习第 2 阶段）！把它看作是让DeepSeek R1真正安全的最后一道工序。

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-212.png)最终 RL 步骤

对于我们的示例“2 + 3 * 4 等于多少？”虽然准确度奖励仍然强化了正确答案，但奖励系统现在还考虑：

- 有用性，评估摘要（如果生成）是否提供了除答案之外的有用背景。
- 无害性，检查整个输出是否安全且无偏见。这些通常由根据人类偏好训练的单独奖励模型进行评估。

最终的奖励信号成为准确度、有用性和无害性分数的加权组合。

现在，训练数据包括

- 多样化组合，包括推理问题
- 一般 QA 提示
- 写作任务
- 和偏好对，其中人类指出两个模型输出中的哪一个在有用性和无害性方面更好。

训练过程遵循迭代 RL 循环（可能使用 GRPO）以根据来自这些多样化数据的组合奖励信号优化模型。

经过多次训练迭代后，模型得到改进，在推理性能和一致性（有用性/无害性）之间取得良好平衡。一旦达到这种平衡，模型就会在流行的基准数据集上进行评估，并超越其他模型的性能。

> 他们的最终检查点，高度优化的版本被命名为 DeepSeek-R1

## 15、蒸馏

![img](http://www.hubwiz.com/blog/content/images/2025/02/image-213.png)R1 的蒸馏

在 DeepSeek 团队能够创建性能良好的 DeepSeek R1 后，他们进一步将更大的模型提炼为性能更高的小型模型，供社区使用，蒸馏过程的工作原理如下：

- 数据准备：收集 800k 个推理样本。
- DeepSeek-R1 输出：对于每个样本，教师模型（DeepSeek-R1）的输出用作学生模型的目标。
- 监督式微调 (SFT)：学生模型（例如 Qwen-1.5B、Llama-14B）基于这 800k 个样本进行微调，以匹配 DeepSeek-R1 输出。
- 蒸馏模型：学生模型现在被精炼成更小的版本，但保留了 DeepSeek-R1 的大部分推理能力。
- 结果：你将获得更小、更快且具有良好推理能力的模型，随时可以部署。
  

