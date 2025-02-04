import{_ as e,c as t,o as l,ag as r}from"./chunks/framework.ByciF0Oj.js";const o="/server/model-1.png",p="/server/model-2.png",n="/server/qfan1.png",s="/server/qfan2.png",i="/server/qfan3.png",c="/server/qwen1.png",h="/server/qwen2.png",m="/server/ollama1.png",d="/server/ollama2.png",g="/server/ollama3.png",u="/server/ollama4.png",_="/server/ollama5.png",b="/server/ollama6.png",C=JSON.parse('{"title":"在LangChat中配置模型","description":"","frontmatter":{},"headers":[],"relativePath":"langchat/docs/server/models.md","filePath":"langchat/docs/server/models.md","lastUpdated":1738637454000}'),k={name:"langchat/docs/server/models.md"};function f(q,a,y,v,z,A){return l(),t("div",null,a[0]||(a[0]=[r('<h1 id="在langchat中配置模型" tabindex="-1">在LangChat中配置模型 <a class="header-anchor" href="#在langchat中配置模型" aria-label="Permalink to &quot;在LangChat中配置模型&quot;">​</a></h1><p>LangChat支持动态配置国内外数十家AI大模型，只需要在<strong>AIGC平台 -&gt; 模型管理页面</strong> 配置中配置大模型即可，配置后系统会动态刷新配置，及时应用。</p><h2 id="配置官方模型" tabindex="-1">配置官方模型 <a class="header-anchor" href="#配置官方模型" aria-label="Permalink to &quot;配置官方模型&quot;">​</a></h2><p><strong>官方模型</strong>指的是使用官方渠道购买的模型Key，例如openai官方key就是在<a href="https://openai.com/%E5%AE%98%E7%BD%91%E8%B4%AD%E4%B9%B0%E7%9A%84%E3%80%82" target="_blank" rel="noreferrer">https://openai.com/官网购买的。</a></p><p>对于这种方式，只需要按照LangChat模型配置页面，在左侧点击不同的模型供应商配置即可。</p><p><strong>注意：</strong></p><ol><li>除了Ollama这种本地模型必须配置BaseUrl之外，其他的官方模型其实都不需要配置BaseUrl</li><li>只需要按照模型供应商官方文档配置apiKey即可</li><li>对于使用第三方代理的情况，后面将详细说明</li></ol><p><img src="'+o+'" alt="" loading="lazy"></p><h2 id="配置openai" tabindex="-1">配置OpenAI <a class="header-anchor" href="#配置openai" aria-label="Permalink to &quot;配置OpenAI&quot;">​</a></h2><p>OpenAI官方只需要填写Api Key，然后选择一个模型即可。</p><blockquote><p>注意：OpenAI接口默认是不可访问的，在本地一般我们会用科学上网实现，但是在服务器上一般必须通过代理实现</p></blockquote><p><img src="'+p+'" alt="" loading="lazy"></p><h2 id="配置千帆大模型" tabindex="-1">配置千帆大模型 <a class="header-anchor" href="#配置千帆大模型" aria-label="Permalink to &quot;配置千帆大模型&quot;">​</a></h2><blockquote><p>百度千帆大模型对于新用户注册，可以免费使用限额限速的模型：ERNIE-Speed-8K，其他模型按量收费，需要自行开通</p></blockquote><p>首先你需要注册百度智能云账号：<a href="https://console.bce.baidu.com/" target="_blank" rel="noreferrer">https://console.bce.baidu.com/</a>。注册成功后转到控制台页面，在如下页面配置你的应用：</p><p><img src="'+n+'" alt="" loading="lazy"></p><p>创建应用的时候会提醒你开通哪些模型，也可以在这里编辑需要开通的模型：（如果未开通直接调用api会提醒需要开通服务）</p><p><img src="'+s+'" alt="" loading="lazy"></p><p>最终我们需要拿到上面创建应用的 <code>Api Key</code> <code>Secret Key</code>信息配置LangChat：</p><p><img src="'+i+'" alt="" loading="lazy"></p><h2 id="配置千问大模型" tabindex="-1">配置千问大模型 <a class="header-anchor" href="#配置千问大模型" aria-label="Permalink to &quot;配置千问大模型&quot;">​</a></h2><p>阿里千问模型服务，统一使用阿里的灵积服务平台：<a href="https://dashscope.aliyun.com/" target="_blank" rel="noreferrer">https://dashscope.aliyun.com/</a></p><p><img src="'+c+'" alt="" loading="lazy"></p><p>生成Api key后在LangChat模型管理页面配置模型信息即可，千问的模型也是按量计费。</p><p><img src="'+h+'" alt="" loading="lazy"></p><h2 id="配置智谱ai" tabindex="-1">配置智谱AI <a class="header-anchor" href="#配置智谱ai" aria-label="Permalink to &quot;配置智谱AI&quot;">​</a></h2><p>官方文档看这里：</p><p><a href="https://www.zhipuai.cn/" target="_blank" rel="noreferrer">https://www.zhipuai.cn/</a></p><h2 id="配置ollama" tabindex="-1">配置Ollama <a class="header-anchor" href="#配置ollama" aria-label="Permalink to &quot;配置Ollama&quot;">​</a></h2><blockquote><p>Ollama是一种本地模型部署方案，Ollama简化了模型的部署方式，通过官方一行命令即可在本地下载并启动一个模型。重点：<strong>有很多不需要GPU的模型</strong></p></blockquote><p>Ollama的官网：<a href="https://ollama.com/library" target="_blank" rel="noreferrer">https://ollama.com/library</a></p><p>例如我们想要在本地部署最新的Llama3.1模型：</p><p><img src="'+m+'" alt="" loading="lazy"></p><p>首先需要安装Ollama官方客户端，然后直接执行命令：</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">ollama</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> run</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> llama3.1</span></span></code></pre></div><p>默认会下载最小的模型，越大的模型对电脑配置要求越高，最小的模型一般不需要GPU也可运行（我使用的16/512 Mac book pro测试）。</p><p><img src="'+d+'" alt="" loading="lazy"></p><p><img src="'+g+'" alt="" loading="lazy"></p><p>下载完成后会直接运行模型，如下，非常简单，我们可以直接通过命令行对话模型：</p><p><img src="'+u+'" alt="" loading="lazy"></p><h3 id="配置ollama-1" tabindex="-1">配置Ollama <a class="header-anchor" href="#配置ollama-1" aria-label="Permalink to &quot;配置Ollama&quot;">​</a></h3><p>启动完成后，Ollama默认会暴露一个http端口：<a href="http://127.0.0.1:11434" target="_blank" rel="noreferrer">http://127.0.0.1:11434</a> 。也就是我们最终会使用<code>http://127.0.0.1:11434/api/chat</code> 接口和Ollama模型聊天。</p><p>首先我们需要在LangChat模型管理页面配置Ollama模型信息，这里的BaseUrl必须填写上述地址，然后【模型】填写你运行的模型名称即可：</p><p><img src="'+_+'" alt="" loading="lazy"></p><p>测试效果如下，在LangChat的聊天助手页面的右上角选择刚刚配置的Ollama模型即可快速使用。 （注意：最终的效果完全取决于模型的能力，和模型的参数大小有关，想要更好的效果、更快的响应速度就必须配置更高参数的列表，也就必须使用显存更高的机器）</p><p><img src="'+b+'" alt="" loading="lazy"></p><h2 id="配置azure-openai" tabindex="-1">配置Azure OpenAI <a class="header-anchor" href="#配置azure-openai" aria-label="Permalink to &quot;配置Azure OpenAI&quot;">​</a></h2><blockquote><p>由于作者没有Azure OpenAI账号，所以这里不再演示，按照表单配置参数即可</p></blockquote><p>官方文档看这里：</p><p><a href="https://learn.microsoft.com/en-us/azure/ai-services/openai/" target="_blank" rel="noreferrer">https://learn.microsoft.com/en-us/azure/ai-services/openai/</a></p><h2 id="配置gemini" tabindex="-1">配置Gemini <a class="header-anchor" href="#配置gemini" aria-label="Permalink to &quot;配置Gemini&quot;">​</a></h2><blockquote><p>注意：Google Gemini使用Google Vertex的认证方式，并不是直接填写api key就可以了，需要本地电脑安装一些google身份认证工具CLI</p></blockquote><p>官方文档看这里：</p><p><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal?hl=zh-cn#new-to-google-cloud" target="_blank" rel="noreferrer">https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal?hl=zh-cn#new-to-google-cloud</a></p><h2 id="配置claude" tabindex="-1">配置Claude <a class="header-anchor" href="#配置claude" aria-label="Permalink to &quot;配置Claude&quot;">​</a></h2><p>官方文档看这里：</p><p><a href="https://docs.anthropic.com/en/docs/welcome" target="_blank" rel="noreferrer">https://docs.anthropic.com/en/docs/welcome</a></p>',57)]))}const x=e(k,[["render",f]]);export{C as __pageData,x as default};
