import{_ as s,c as n,o as e,ag as p}from"./chunks/framework.ByciF0Oj.js";const h=JSON.parse('{"title":"蒸馏DeepSeek-R1到自己的模型","description":"","frontmatter":{},"headers":[],"relativePath":"langchat/docs/distill-deepseek-r1-into-your-model.md","filePath":"langchat/docs/distill-deepseek-r1-into-your-model.md","lastUpdated":1739332406000}'),t={name:"langchat/docs/distill-deepseek-r1-into-your-model.md"};function l(i,a,o,r,c,d){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="蒸馏deepseek-r1到自己的模型" tabindex="-1">蒸馏DeepSeek-R1到自己的模型 <a class="header-anchor" href="#蒸馏deepseek-r1到自己的模型" aria-label="Permalink to &quot;蒸馏DeepSeek-R1到自己的模型&quot;">​</a></h1><p>在本博客中，我们将介绍如何使用LoRA等技术将 DeepSeek-R1 的推理能力蒸馏到较小的模型（如 Microsoft 的 Phi-3-Mini）中。</p><p><img src="http://www.hubwiz.com/blog/content/images/size/w2000/2025/02/distill-deepseek-r1-into-your-model.png" alt="蒸馏DeepSeek-R1到自己的模型" loading="lazy"></p><p>深度学习模型彻底改变了人工智能领域，但其庞大的规模和计算需求可能会成为实际应用的瓶颈。模型蒸馏是一种强大的技术，它通过将知识从大型复杂模型（教师）转移到较小、更高效的模型（学生）来解决这一挑战。</p><p>在本博客中，我们将介绍如何使用 LoRA（低秩自适应）等专门技术将 DeepSeek-R1 的推理能力蒸馏到较小的模型（如 Microsoft 的 Phi-3-Mini）中。</p><h2 id="_1、什么是蒸馏" tabindex="-1">1、什么是蒸馏？ <a class="header-anchor" href="#_1、什么是蒸馏" aria-label="Permalink to &quot;1、什么是蒸馏？&quot;">​</a></h2><p>蒸馏是一种机器学习技术，其中较小的模型（“学生”）经过训练以模仿较大的预训练模型（“老师”）的行为。目标是保留老师的大部分表现，同时显着降低计算成本和内存占用。</p><p>这个想法最早是在 Geoffrey Hinton 关于知识蒸馏的开创性论文中提出的。它不是直接在原始数据上训练学生模型，而是从老师模型的输出或中间表示中学习。这实际上是受到人类教育的启发。</p><p>为什么它很重要：</p><ul><li>成本效率：较小的模型需要更少的计算资源。</li><li>速度：非常适合延迟敏感的应用程序（例如 API、边缘设备）。</li><li>专业化：无需重新训练巨型模型即可针对特定领域定制模型。</li></ul><h2 id="_2、蒸馏类型" tabindex="-1">2、蒸馏类型 <a class="header-anchor" href="#_2、蒸馏类型" aria-label="Permalink to &quot;2、蒸馏类型&quot;">​</a></h2><p>模型蒸馏有几种方法，每种方法都有各自的优点：</p><p>数据蒸馏：</p><ul><li>在数据蒸馏中，教师模型生成合成数据或伪标签，然后用于训练学生模型。</li><li>这种方法可以应用于广泛的任务，即使是那些 logits 信息量较少的任务（例如开放式推理任务）。</li></ul><p>Logits蒸馏：</p><ul><li>Logits 是应用 softmax 函数之前神经网络的原始输出分数。</li><li>在 logits蒸馏中，学生模型经过训练以匹配教师的 logits，而不仅仅是最终预测。</li><li>这种方法保留了更多关于教师信心水平和决策过程的信息。</li></ul><p>特征蒸馏：</p><ul><li>特征提炼涉及将知识从教师模型的中间层转移到学生。</li><li>通过对齐两个模型的隐藏表示，学生可以学习更丰富、更抽象的特征。</li></ul><h2 id="_3、deepseek-的蒸馏模型" tabindex="-1">3、Deepseek 的蒸馏模型 <a class="header-anchor" href="#_3、deepseek-的蒸馏模型" aria-label="Permalink to &quot;3、Deepseek 的蒸馏模型&quot;">​</a></h2><p>为了使访问更加民主化，DeepSeek AI 发布了基于 Qwen（Qwen，2024b）和 Llama（AI@Meta，2024）等流行架构的六个蒸馏变体。他们使用 DeepSeek-R1 策划的 800k 个样本直接微调开源模型。</p><p>尽管比 DeepSeek-R1 小得多，但蒸馏模型在各种基准测试中都表现出色，通常可以匹敌甚至超越更大模型的能力。如下图所示</p><p><img src="http://www.hubwiz.com/blog/content/images/2025/02/image-75.png" alt="img" loading="lazy">Deepseek 提炼模型基准测试（<a href="https://arxiv.org/html/2501.12948v1%EF%BC%89" target="_blank" rel="noreferrer">https://arxiv.org/html/2501.12948v1）</a></p><h2 id="_4、为什么要蒸馏自己的模型" tabindex="-1">4、为什么要蒸馏自己的模型？ <a class="header-anchor" href="#_4、为什么要蒸馏自己的模型" aria-label="Permalink to &quot;4、为什么要蒸馏自己的模型？&quot;">​</a></h2><ul><li>特定任务优化</li></ul><p>预蒸馏模型在广泛的数据集上进行训练，以在各种任务中表现良好。然而，现实世界的应用程序通常需要专业化。</p><p>示例场景：你正在构建一个金融预测聊天机器人。在这种情况下，使用 DeepSeek-R1 为金融数据集生成推理轨迹（例如，股票价格预测、风险分析），并将这些知识蒸馏成一个已经了解金融细微差别的较小模型（例如：finance-LLM）。</p><ul><li>大规模成本效率</li></ul><p>虽然预蒸馏模型效率很高，但它们可能仍然不适合你的特定工作量。蒸馏你自己的模型可以让你针对确切的资源限制进行优化。</p><ul><li>基准性能 ≠ 真实世界性能</li></ul><p>预蒸馏模型在基准测试中表现出色，但基准测试通常不能代表真实世界的任务。因此，你通常需要一个在真实世界场景中表现比任何预蒸馏模型都更好的模型。</p><ul><li>迭代改进</li></ul><p>预蒸馏模型是静态的——它们不会随着时间的推移而改进。通过蒸馏自己的模型，你可以在新数据可用时不断完善它。</p><h2 id="_5、将-deepseek-r1-知识蒸馏成自定义小模型" tabindex="-1">5、将 DeepSeek-R1 知识蒸馏成自定义小模型 <a class="header-anchor" href="#_5、将-deepseek-r1-知识蒸馏成自定义小模型" aria-label="Permalink to &quot;5、将 DeepSeek-R1 知识蒸馏成自定义小模型&quot;">​</a></h2><p>首先安装库：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>pip install -q torch transformers datasets accelerate bitsandbytes flash-attn --no-build-isolation</span></span></code></pre></div><h3 id="_5-1-生成和格式化数据集" tabindex="-1">5.1 生成和格式化数据集 <a class="header-anchor" href="#_5-1-生成和格式化数据集" aria-label="Permalink to &quot;5.1 生成和格式化数据集&quot;">​</a></h3><p>你可以通过在你的环境中使用 ollama 或任何其他部署框架部署 deepseek-r1 来生成自定义域相关数据集。但是，对于本教程，我们将使用 Magpie-Reasoning-V2 数据集，其中包含 DeepSeek-R1 生成的 250K 思路链 (CoT) 推理样本，这些示例涵盖了数学推理、编码和一般问题解决等各种任务。</p><blockquote><p>数据集结构</p></blockquote><p>每个示例包括：</p><ul><li>指令：任务描述（例如，“解决这个数学问题”）。</li><li>响应：DeepSeek-R1 的分步推理 (CoT)。</li></ul><p>示例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>{</span></span>
<span class="line"><span>  &quot;instruction&quot;: &quot;Solve for x: 2x + 5 = 15&quot;,</span></span>
<span class="line"><span>  &quot;response&quot;: &quot;&lt;think&gt;First, subtract 5 from both sides: 2x = 10. Then, divide by 2: x = 5.&lt;/think&gt;&quot;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>from datasets import load_dataset</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Load the dataset</span></span>
<span class="line"><span>dataset = load_dataset(&quot;Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B&quot;, token=&quot;YOUR_HF_TOKEN&quot;)</span></span>
<span class="line"><span>dataset = dataset[&quot;train&quot;]</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Format the dataset</span></span>
<span class="line"><span>def format_instruction(example):</span></span>
<span class="line"><span>    return {</span></span>
<span class="line"><span>        &quot;text&quot;: (</span></span>
<span class="line"><span>            &quot;&lt;|user|&gt;\\n&quot;</span></span>
<span class="line"><span>            f&quot;{example[&#39;instruction&#39;]}\\n&quot;</span></span>
<span class="line"><span>            &quot;&lt;|end|&gt;\\n&quot;</span></span>
<span class="line"><span>            &quot;&lt;|assistant|&gt;\\n&quot;</span></span>
<span class="line"><span>            f&quot;{example[&#39;response&#39;]}\\n&quot;</span></span>
<span class="line"><span>            &quot;&lt;|end|&gt;&quot;</span></span>
<span class="line"><span>        )</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span></span></span>
<span class="line"><span>formatted_dataset = dataset.map(format_instruction, batched=False, remove_columns=subset_dataset.column_names)</span></span>
<span class="line"><span>formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)  # 90-10 train-test split</span></span></code></pre></div><p>将数据集构造为 Phi-3 的聊天模板格式：</p><ul><li><code>&lt;|user|&gt;</code>：标记用户查询的开始。</li><li><code>&lt;|assistant|&gt;</code>：标记模型响应的开始。</li><li><code>&lt;|end|&gt;</code>：标记回合的结束。</li></ul><p>每个 LLM 都使用特定格式来执行指令跟踪任务。将数据集与此结构对齐可确保模型学习正确的对话模式。因此，请确保根据要提取的模型格式化数据。</p><h3 id="_5-2-加载模型和标记器" tabindex="-1">5.2 加载模型和标记器 <a class="header-anchor" href="#_5-2-加载模型和标记器" aria-label="Permalink to &quot;5.2 加载模型和标记器&quot;">​</a></h3><p>向标记器添加特殊标记 <code>&lt;think&gt;</code> 和 <code>&lt;/think&gt;</code> 。</p><p>为了增强模型的推理能力，我们引入了这些标记。</p><ul><li><code>&lt;think&gt;</code>：标记推理的开始。</li><li><code>&lt;/think&gt;</code>：标记推理的结束。</li></ul><p>这些标记帮助模型学习生成结构化的、分步的解决方案。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>from transformers import AutoTokenizer, AutoModelForCausalLM</span></span>
<span class="line"><span></span></span>
<span class="line"><span>model_id = &quot;microsoft/phi-3-mini-4k-instruct&quot;</span></span>
<span class="line"><span>tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Add custom tokens</span></span>
<span class="line"><span>CUSTOM_TOKENS = [&quot;&lt;think&gt;&quot;, &quot;&lt;/think&gt;&quot;]</span></span>
<span class="line"><span>tokenizer.add_special_tokens({&quot;additional_special_tokens&quot;: CUSTOM_TOKENS})</span></span>
<span class="line"><span>tokenizer.pad_token = tokenizer.eos_token</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Load model with flash attention</span></span>
<span class="line"><span>model = AutoModelForCausalLM.from_pretrained(</span></span>
<span class="line"><span>    model_id,</span></span>
<span class="line"><span>    trust_remote_code=True,</span></span>
<span class="line"><span>    device_map=&quot;auto&quot;,</span></span>
<span class="line"><span>    torch_dtype=torch.float16,</span></span>
<span class="line"><span>    attn_implementation=&quot;flash_attention_2&quot;</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>model.resize_token_embeddings(len(tokenizer))  # Resize for custom tokens</span></span></code></pre></div><h3 id="_5-3-配置-lora-以实现高效微调" tabindex="-1">5.3 配置 LoRA 以实现高效微调 <a class="header-anchor" href="#_5-3-配置-lora-以实现高效微调" aria-label="Permalink to &quot;5.3 配置 LoRA 以实现高效微调&quot;">​</a></h3><p>LoRA 通过冻结基础模型并仅训练小型适配器层来减少内存使用量。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>from peft import LoraConfig</span></span>
<span class="line"><span></span></span>
<span class="line"><span>peft_config = LoraConfig(</span></span>
<span class="line"><span>    r=8,  # Rank of the low-rank matrices</span></span>
<span class="line"><span>    lora_alpha=16,  # Scaling factor</span></span>
<span class="line"><span>    lora_dropout=0.2,  # Dropout rate</span></span>
<span class="line"><span>    target_modules=[&quot;q_proj&quot;, &quot;k_proj&quot;, &quot;v_proj&quot;, &quot;o_proj&quot;],  # Target attention layers</span></span>
<span class="line"><span>    bias=&quot;none&quot;,  # No bias terms</span></span>
<span class="line"><span>    task_type=&quot;CAUSAL_LM&quot;  # Task type</span></span>
<span class="line"><span>)</span></span></code></pre></div><h3 id="_5-4-设置训练参数" tabindex="-1">5.4 设置训练参数 <a class="header-anchor" href="#_5-4-设置训练参数" aria-label="Permalink to &quot;5.4 设置训练参数&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>from transformers import TrainingArguments</span></span>
<span class="line"><span></span></span>
<span class="line"><span>training_args = TrainingArguments(</span></span>
<span class="line"><span>    output_dir=&quot;./phi-3-deepseek-finetuned&quot;,</span></span>
<span class="line"><span>    num_train_epochs=3,</span></span>
<span class="line"><span>    per_device_train_batch_size=2,</span></span>
<span class="line"><span>    per_device_eval_batch_size=2,</span></span>
<span class="line"><span>    gradient_accumulation_steps=4,</span></span>
<span class="line"><span>    eval_strategy=&quot;epoch&quot;,</span></span>
<span class="line"><span>    save_strategy=&quot;epoch&quot;,</span></span>
<span class="line"><span>    logging_strategy=&quot;steps&quot;,</span></span>
<span class="line"><span>    logging_steps=50,</span></span>
<span class="line"><span>    learning_rate=2e-5,</span></span>
<span class="line"><span>    fp16=True,</span></span>
<span class="line"><span>    optim=&quot;paged_adamw_32bit&quot;,</span></span>
<span class="line"><span>    max_grad_norm=0.3,</span></span>
<span class="line"><span>    warmup_ratio=0.03,</span></span>
<span class="line"><span>    lr_scheduler_type=&quot;cosine&quot;</span></span>
<span class="line"><span>)</span></span></code></pre></div><h3 id="_5-5-训练模型" tabindex="-1">5.5 训练模型 <a class="header-anchor" href="#_5-5-训练模型" aria-label="Permalink to &quot;5.5 训练模型&quot;">​</a></h3><p><code>SFTTrainer</code> 简化了指令跟随模型的监督微调。 <code>data_collator</code> 批量处理示例， <code>peft_config</code> 支持基于 LoRA 的训练。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>from trl import SFTTrainer</span></span>
<span class="line"><span>from transformers import DataCollatorForLanguageModeling</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Data collator</span></span>
<span class="line"><span>data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Trainer</span></span>
<span class="line"><span>trainer = SFTTrainer(</span></span>
<span class="line"><span>    model=model,</span></span>
<span class="line"><span>    args=training_args,</span></span>
<span class="line"><span>    train_dataset=formatted_dataset[&quot;train&quot;],</span></span>
<span class="line"><span>    eval_dataset=formatted_dataset[&quot;test&quot;],</span></span>
<span class="line"><span>    data_collator=data_collator,</span></span>
<span class="line"><span>    peft_config=peft_config</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Start training</span></span>
<span class="line"><span>trainer.train()</span></span>
<span class="line"><span>trainer.save_model(&quot;./phi-3-deepseek-finetuned&quot;)</span></span>
<span class="line"><span>tokenizer.save_pretrained(&quot;./phi-3-deepseek-finetuned&quot;)</span></span></code></pre></div><h3 id="_5-6-合并保存最终模型" tabindex="-1">5.6 合并保存最终模型 <a class="header-anchor" href="#_5-6-合并保存最终模型" aria-label="Permalink to &quot;5.6 合并保存最终模型&quot;">​</a></h3><p>训练后，必须将 LoRA 适配器与基础模型合并以进行推理。此步骤确保模型可以在没有 PEFT 的情况下独立使用。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>final_model = trainer.model.merge_and_unload()</span></span>
<span class="line"><span>final_model.save_pretrained(&quot;./phi-3-deepseek-finetuned-final&quot;)</span></span>
<span class="line"><span>tokenizer.save_pretrained(&quot;./phi-3-deepseek-finetuned-final&quot;)</span></span></code></pre></div><h3 id="_5-7-推理" tabindex="-1">5.7 推理 <a class="header-anchor" href="#_5-7-推理" aria-label="Permalink to &quot;5.7 推理&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>from transformers import pipeline</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Load fine-tuned model</span></span>
<span class="line"><span>model = AutoModelForCausalLM.from_pretrained(</span></span>
<span class="line"><span>    &quot;./phi-3-deepseek-finetuned-final&quot;,</span></span>
<span class="line"><span>    device_map=&quot;auto&quot;,</span></span>
<span class="line"><span>    torch_dtype=torch.float16</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tokenizer = AutoTokenizer.from_pretrained(&quot;./phi-3-deepseek-finetuned-final&quot;)</span></span>
<span class="line"><span>model.resize_token_embeddings(len(tokenizer))</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Create chat pipeline</span></span>
<span class="line"><span>chat_pipeline = pipeline(</span></span>
<span class="line"><span>    &quot;text-generation&quot;,</span></span>
<span class="line"><span>    model=model,</span></span>
<span class="line"><span>    tokenizer=tokenizer,</span></span>
<span class="line"><span>    device_map=&quot;auto&quot;</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Generate response</span></span>
<span class="line"><span>prompt = &quot;&quot;&quot;&lt;|user|&gt;</span></span>
<span class="line"><span>What&#39;s the probability of rolling a 7 with two dice?</span></span>
<span class="line"><span>&lt;|end|&gt;</span></span>
<span class="line"><span>&lt;|assistant|&gt;</span></span>
<span class="line"><span>&quot;&quot;&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>output = chat_pipeline(</span></span>
<span class="line"><span>    prompt,</span></span>
<span class="line"><span>    max_new_tokens=5000,</span></span>
<span class="line"><span>    temperature=0.7,</span></span>
<span class="line"><span>    do_sample=True,</span></span>
<span class="line"><span>    eos_token_id=tokenizer.eos_token_id</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(output[0][&#39;generated_text&#39;])</span></span></code></pre></div><p>下面你可以看到 phi 模型在蒸馏前后的响应。</p><blockquote><p>问题：用两个骰子掷出 7 的概率是多少？</p></blockquote><ul><li>蒸馏前的推理</li></ul><p>响应简单明了。它直接提供了计算答案的步骤。</p><p><img src="http://www.hubwiz.com/blog/content/images/2025/02/image-76.png" alt="img" loading="lazy">蒸馏前的 Phi 推理</p><ul><li>蒸馏后的推理</li></ul><p>蒸馏后的响应引入了一种更详细和结构化的方法，包括一个明确的“思考”部分，概述了思维过程和推理，这对于为复杂问题生成准确的响应非常有帮助。</p><p><img src="http://www.hubwiz.com/blog/content/images/2025/02/image-77.png" alt="img" loading="lazy">蒸馏后的 Phi 推理</p><p>最后，将蒸馏后的模型权重推送到 <a href="https://huggingface.co/GPD1/DeepSeek-R1-Distill-phi-3-mini-4k-lorar8-alpha16-50000samples" target="_blank" rel="noreferrer">huggingface hub</a>（repo_id： <code>GPD1/DeepSeek-R1-Distill-phi-3-mini-4k-lorar8-alpha16–50000samples</code>）。</p>`,73)]))}const g=s(t,[["render",l]]);export{h as __pageData,g as default};
