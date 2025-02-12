# LangChat如何接入DeepSeek-R1模型

> 本教程给使用LangChat的朋友学习如何本地部署DeepSeek-R1模型。以及如何使用LangChat的Agent功能构建知识库。

### 关于LangChat

**LangChat** 是Java生态下企业级AIGC项目解决方案，集成RBAC和AIGC大模型能力，帮助企业快速定制AI知识库、企业AI机器人。

**支持的AI大模型：** Gitee AI / 阿里通义 / 百度千帆 / DeepSeek / 抖音豆包 / 智谱清言 / 零一万物 / 讯飞星火 / OpenAI / Gemini / Ollama / Azure / Claude 等大模型。

- 官网地址：[http://langchat.cn/](http://langchat.cn/)

**开源地址：**

- Gitee：https://gitee.com/langchat/langchat
- Github：https://github.com/tycoding/langchat

## 安装DeepSeek-R1

DeepSeek-R1是由DeepSeek公司推出的开源大模型，目前最强的低成本推理模型。

**注意：** 这里仅介绍使用Ollama部署DeepSeek-R1模型，其他模型的部署方式可以参考官方文档。

### 基础概念

首先，必须给大家介绍一些基础的概念，避免大家有各种疑惑。

1. Ollama是安装模型最简单的方式，不需要python、Docker以及其他复杂的过程。
2. 无论本地安装DeepSeek-R1的哪个模型，其实区别都不大，因为都是阉割版的，所以大家不要被营销号带偏。
3. Ollama的`run`命令执行后，首先会检查本地有没有此模型，有就安装，没有就直接运行模型。（Ollama客户端兼容全平台，不是非要Linux服务器）
4. Ollama的`run`启动模型后会暴露 **11434** HTTP端口，其他所有的LLM OPS应用都是通过此端口和模型交互的，你可以访问 [http://127.0.0.1:11434](http://127.0.0.1:11434) 查看
5. DeepSeek-R1是推理模型，Embedding模型需要单独安装，这两者不是一个概念，如果需要知识库向量化，不应该用DeepSeek-R1
6. ......

### DeepSeek-R1下载哪个版本？

**实际上，本地测试而言，无论你下载哪个版本，最终的效果都是一样的**。因为都是阉割版的小参数模型，所以我推荐各位安装 **1.5B** 或者 **7/8B** 测试即可。真实的场景下，还是推荐调用API。

Ollama地址：https://ollama.com/library/deepseek-r1

 ![image-20250211164354673](http://cdn.tycoding.cn/docs/202502111644203.png)



**注意：** 

运行 `ollama run deepseek-r1` 命令默认安装的 **7B** 版本。

1. 1.5B 模型，基本上任意笔记本都能安装
2. 7/8B 模型，至少本地电脑有16G内存
3. 14B以上模型，至少本地电脑有32G内存

**作者本地电脑是 Macbook m3 16GB + 512版本**

### 1. 安装Ollama

Ollama官网地址 https://ollama.com/

![image-20250211164855062](http://cdn.tycoding.cn/docs/202502111648168.png)

这里会根据你的操作系统下载对应的安装包

**安装Ollama的步骤这里就不再解释了**

### 2. 安装DeepSeek-R1

> 如果你本地电脑是 >=16G内存，就用默认的安装命令，否则建议安装1.5B小模型

![image-20250211165252956](http://cdn.tycoding.cn/docs/202502111652139.png)

因为作者是16G笔记本，所以这里直接按照默认的下载7B模型

正常情况下，如上图所示，本地电脑就已经安装好了DeepSeek-R1模型。

### 3. 验证DeepSeek-R1模型是否启动

如上，你可以在控制台直接交互。

另外本地访问 [http://127.0.0.1:11434](http://127.0.0.1:11434) 地址，能看到如下信息：

![image-20250211170217736](http://cdn.tycoding.cn/docs/202502111702790.png)

## 启动LangChat

> 注意：LangChat至少需要以下环境：
>
> 1. MySQL8
> 2. JDK17+
> 3. PgVector等



**开源地址：**

- Gitee：https://gitee.com/langchat/langchat
- Github：https://github.com/tycoding/langchat
- GitCode: https://gitcode.com/LangChat/LangChat

首先本地IDEA打开LangChat项目（等待Maven加载完成）

![image-20250211170424610](http://cdn.tycoding.cn/docs/202502111704697.png)

### 1. 执行数据库脚本

![image-20250211170540137](http://cdn.tycoding.cn/docs/202502111705186.png)

在docs/目录下找到`langchat.sql` 在MySQL中执行此脚本。

**注意：** 此脚本包含了创建名为`langchat`的数据库（因此不需要手动创建数据库）

### 2. 修改配置文件

首先你应该检查SpringBoot的`application-*.yml`配置文件

![image-20250211170752880](http://cdn.tycoding.cn/docs/202502111707940.png)

**必须修改：**

1. MySQL连接信息
2. OSS信息（默认的`local`代表了使用tomcat的地址，当然建议使用阿里云或七牛云，或者本地用NGINX搭建本地文件服务器）

### 3. 安装PgVector

> 这里我只推荐Pgvector，不要用Redis，Pgvector可以用navicat等工具可视化查看数据表

Pgvector官方仓库：https://github.com/pgvector/pgvector?tab=readme-ov-file#installation-notes---windows

Postgres官网：https://postgresapp.com/downloads.html

注意：安装Pgvector后仍需要有Postgres 15+基础环境。所以如果你是第一次安装，你需要安装两者才行。

![image-20250211171529888](http://cdn.tycoding.cn/docs/202502111715986.png)

因为我使用的Mac，所以有多种安装方式，如果不想麻烦可以用Docker

![image-20250211171648413](http://cdn.tycoding.cn/docs/202502111716476.png)

作者贴心的给大家编译了一个pgvector发布到了阿里云仓库，直接运行此compose也可启动，省去了上麦那一系列步骤

如果上面脚本执行成功，应该在数据库能看到`langchat`

![image-20250211171821785](http://cdn.tycoding.cn/docs/202502111718850.png)

**注意：**

1. 如果是自己手动安装的Pgvector，请手动创建`langchat`数据库
2. 尽量不要自己手动编译pgvector源码，太麻烦了

### 4. 运行LangChat

上述配置完毕后，即可正常启动LangChat

![image-20250211172222107](http://cdn.tycoding.cn/docs/202502111722185.png)

启动成功后如上图，注意 **当前环境是什么就代表用了哪个配置文件**

**运行langchat-ui**

![image-20250211172348017](http://cdn.tycoding.cn/docs/202502111723098.png)

## 测试LangChat

首先进入到LangChat此页面

![image-20250211172719346](http://cdn.tycoding.cn/docs/202502111727377.png)

因为我们使用的Ollama部署的DeepSeek-R1模型，因此必须使用Ollama配置

1. 模型版本写： `deepseek-r1`
2. BaseUlr写：`http://127.0.0.1:11434/`
3. ApiKey任意填

![image-20250211172945369](http://cdn.tycoding.cn/docs/202502111729516.png)

### 测试LangChat聊天功能

> 到此为止，DeepSeek-R1模型已经启动并配置好，我们先测试Chat基础功能

![image-20250211173131416](http://cdn.tycoding.cn/docs/202502111731492.png)

![image-20250211173310009](http://cdn.tycoding.cn/docs/202502111733161.png)

如上，接口已经掉通了。

**注意： `<think>`是DeekSeek-R1的推理过程，因为他是非标准的数据格式，后面LangChat会做前端适配**

## 配置LangChat知识库

> 首先你需要安装好Pgvector和OSS

### 1. 配置向量数据库

![image-20250211173652563](http://cdn.tycoding.cn/docs/202502111736653.png)

**注意！注意！注意！**

**建议不要修改向量维度这个参数，向量数据表一旦初始化，此表的向量维度就固定了，只能接受指定向量的数据，因此仅在LangChat前端修改是无效的（需要删除原表）**

### 2. 本地下载Embedding模型

> 当然我建议大家直接使用阿里云、百度、智谱的Embedding模型，这样只需要配置ApiKey即可，但是很多朋友可能还想想本地部署，这里教大家。

同样，使用Ollama下载Embedding模型

进入官网：https://ollama.com/search?c=embedding  我们找到排名第一的Embedding模型

![image-20250211204936831](http://cdn.tycoding.cn/docs/202502112049972.png)

执行命令 `ollama pull nomic-embed-text`

![image-20250211205200587](http://cdn.tycoding.cn/docs/202502112052648.png)

![image-20250211205253605](http://cdn.tycoding.cn/docs/202502112052670.png)

如上结果，Ollama模型很小，很快运行结束，但是不要尝试执行`ollama run xxx`，因为`run`命令是针对Chat模型的，这里是Embedding模型

**Ollama Pull 了Embedding模型就自动启用了，不需要任何其他命令加载**

我们可以通过`ollama list`查看到下载的模型

![image-20250211205634137](http://cdn.tycoding.cn/docs/202502112056210.png)

### 3. 测试Embedding模型

执行如下脚本

```shell
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "The sky is blue because of Rayleigh scattering"
}'
```

![image-20250211205831626](http://cdn.tycoding.cn/docs/202502112058680.png)

如上说明Embedding模型正在运行，并且模型名称是`nomic-embed-text`，访问地址是：`http://localhost:11434/api/embeddings`

### 4. LangChat配置Embedding

![image-20250211210510117](http://cdn.tycoding.cn/docs/202502112105200.png)

- 模型版本填 `nomic-embed-text` （Select中输入并回车即可）
- BaseUrl填写：`http://localhost:11434/`



### 5. 创建LangChat知识库

知识库中只关联Embedding数据库和Embedding模型

![image-20250211210743993](http://cdn.tycoding.cn/docs/202502112107089.png)

### 6. 导入知识库文档

在上面配置好`nomic-embed-text`模型后，往知识库导入文档

![image-20250211220225536](http://cdn.tycoding.cn/docs/202502112202699.png)

如果你是按照上面步骤的Embedding模型，你应该会收到如下向量化失败错误：

![image-20250211220330043](http://cdn.tycoding.cn/docs/202502112203163.png)

### 为什么报错ERROR: expected 1024

因为我们在LangChat配置的 **1024向量纬度** 的数据库，所以生成的表也只接收1024维度的数据。

**但是，** `nomic-embed-text` 模型只能生成768维度的数据，并不是生成1024维度的数据（当然如果你使用公有云模型，他们的模型一般都能支持生成多维度的数据768、1024、1536等等）

**只不过我们下载的模型只支持生成单维度的向量数据。**

![image-20250211221101301](http://cdn.tycoding.cn/docs/202502112211381.png)

从上表中你可以查看到不同模型能生成什么维度的数据。

例如阿里云的文档中，关于Embedding模型的定义如下：（你在最初阶段就应该考虑哪种模型兼容哪种向量维度）

![image-20250211225113717](http://cdn.tycoding.cn/docs/202502112251834.png)

**遇到这种情况怎么处理？**

无论用的哪个模型，如果向量维度一旦不匹配，就必然会出现类似此报错信息。按照如下步骤开始解决此问题：（我们的前提是仍用本地的Embedding模型，当然你换一个能输出1024维度的Embedding模型也是可以的）

1. 使用如Navicat等客户端工具删除`langchat`库中的表

![image-20250211225400733](http://cdn.tycoding.cn/docs/202502112254845.png)

2. 在LangChat管理端修改 **向量数据库的纬度配置** ，如下修改为768维度

![image-20250211230358598](http://cdn.tycoding.cn/docs/202502112303718.png)

3. 重启LangChat后端项目，系统会重新生成此表，并接收768维度的向量

正常情况，重启后，我们就可以重新导入文档进行向量化了。



### 7. 重新导入知识库文档

我们准备如下这个txt文档

![image-20250211231626120](http://cdn.tycoding.cn/docs/202502112316282.png)

正常情况，后端会提示向量化成功，会有如下日志：

![image-20250211232040168](http://cdn.tycoding.cn/docs/202502112320265.png)

在向量数据库中，能看到如下分段信息：

![image-20250211231945871](http://cdn.tycoding.cn/docs/202502112319952.png)

### 8. 向量搜索测试

正常情况，如果向量化成功，你能在LangChat页面看到如下切面信息：

![image-20250211232251955](http://cdn.tycoding.cn/docs/202502112322040.png)

那我们进行一下向量检索测试：

![image-20250211232340080](http://cdn.tycoding.cn/docs/202502112323154.png)

> 到此为止，LangChat知识库配置已经结束



## 创建LangChat AI应用

上面知识库配置成功后，下面开始创建LangChat AI应用

![image-20250211232536523](http://cdn.tycoding.cn/docs/202502112325631.png)

如下所示创建AI应用，这里进需要关联我们刚才设置的DeepSeek-R1模型

### 配置LangChat应用

创建LangChat应用后，关联刚才创建的知识库，即可进行知识库问答了

![image-20250212112844178](http://cdn.tycoding.cn/docs/202502121128775.png)

关联好我们创建的知识库后，直接测试就能引用知识库的内容了

### 测试LangChat应用

![image-20250212113128034](http://cdn.tycoding.cn/docs/202502121131192.png)

如上，说明了他刚才引用了我们上传的`langchat.txt`文档



### 验证是否查询向量文本？

**验证此回答是否查询了知识库的向量信息？**

1. 我们可以在控制台看到如下打印日志：

![image-20250212113459881](http://cdn.tycoding.cn/docs/202502121134946.png)

后面的部分就是引用的知识库文档

2. 可以拿未配置知识库的普通聊天做测试

![image-20250212113735608](http://cdn.tycoding.cn/docs/202502121137702.png)

可以看到未配置知识库，是不知道LangChat是什么的。
