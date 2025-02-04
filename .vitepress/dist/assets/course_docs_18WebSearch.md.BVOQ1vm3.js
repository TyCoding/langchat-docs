import{_ as i,c as a,o as t,ag as n}from"./chunks/framework.ByciF0Oj.js";const g=JSON.parse('{"title":"函数增强搜索","description":"","frontmatter":{"title":"函数增强搜索"},"headers":[],"relativePath":"course/docs/18WebSearch.md","filePath":"course/docs/18WebSearch.md","lastUpdated":1738637454000}'),h={name:"course/docs/18WebSearch.md"};function e(l,s,p,k,r,E){return t(),a("div",null,s[0]||(s[0]=[n(`<p><a href="https://www.searchapi.io/" target="_blank" rel="noreferrer">SearchApi</a> 是一个实时搜索引擎结果页面（SERP）API。你可以使用它进行 Google、Google News、Bing、Bing News、Baidu、Google Scholar 等搜索引擎的查询，获取有机搜索结果。</p><img src="https://minio.pigx.top/oss/202410/1728463357.png" alt="1728463357"><h2 id="使用方法" tabindex="-1">使用方法 <a class="header-anchor" href="#使用方法" aria-label="Permalink to &quot;使用方法&quot;">​</a></h2><h3 id="获取-api-key" tabindex="-1">获取 API KEY <a class="header-anchor" href="#获取-api-key" aria-label="Permalink to &quot;获取 API KEY&quot;">​</a></h3><img src="https://minio.pigx.top/oss/202410/1728397591.png" alt="1728397591"><h3 id="依赖配置" tabindex="-1">依赖配置 <a class="header-anchor" href="#依赖配置" aria-label="Permalink to &quot;依赖配置&quot;">​</a></h3><p>在你的项目 <code>pom.xml</code> 中添加以下依赖：</p><div class="language-xml vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">xml</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">dependency</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    &lt;</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">groupId</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;dev.langchain4j&lt;/</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">groupId</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">  &lt;</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">artifactId</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;langchain4j-web-search-engine-searchapi&lt;/</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">artifactId</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;/</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">dependency</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span></code></pre></div><h3 id="定义-ai-service" tabindex="-1">定义 Ai Service <a class="header-anchor" href="#定义-ai-service" aria-label="Permalink to &quot;定义 Ai Service&quot;">​</a></h3><div class="language-java vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">java</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> interface</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> ChatAssistant</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    @</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">SystemMessage</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">            	1.	搜索支持：你的职责是为用户提供基于网络搜索的支持。</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">            	2.	事件验证：如果用户提到的事件尚未发生或信息不明确，你需要通过网络搜索确认或查找相关信息。</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">            	3.	网络搜索请求：使用用户的查询创建网络搜索请求，并通过网络搜索工具进行实际查询。</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">            	4.	引用来源：在最终回应中，必须包括搜索到的来源链接，以确保信息的准确性和可验证性。</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">            &quot;&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    String </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">chat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(String </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">message</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><h3 id="注入-web-search" tabindex="-1">注入 web search <a class="header-anchor" href="#注入-web-search" aria-label="Permalink to &quot;注入 web search&quot;">​</a></h3><div class="language-java vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">java</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">@</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">Bean</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ChatAssistant </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">chatAssistant</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ChatLanguageModel chatLanguageModel) {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    SearchApiWebSearchEngine searchEngine </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SearchApiWebSearchEngine.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">builder</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            .</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">apiKey</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;p8SZVNAweqTtoZBBTVnXttcj&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 测试使用</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            .</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">engine</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;google&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            .</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">build</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">();</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    WebSearchTool webSearchTool </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> WebSearchTool.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(searchEngine);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> AiServices.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">builder</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ChatAssistant.class).</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">chatLanguageModel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(chatLanguageModel).</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">tools</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(webSearchTool).</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">build</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">();</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><h3 id="测试" tabindex="-1">测试 <a class="header-anchor" href="#测试" aria-label="Permalink to &quot;测试&quot;">​</a></h3><div class="language-java vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">java</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">String chat </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> chatAssistant.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">chat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;20241008 上证指数是多少&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">System.out.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(chat);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">截至2024年10月8日，A股市场迎来了国庆节后的首个交易日，主要指数表现强劲。具体到上证指数，开盘时涨幅达到了10.</span><span style="--shiki-light:#B31D28;--shiki-light-font-style:italic;--shiki-dark:#FDAEB7;--shiki-dark-font-style:italic;">13</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">%</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">，但之后的走势有所调整，最终收盘时上证指数报收于3489.</span><span style="--shiki-light:#B31D28;--shiki-light-font-style:italic;--shiki-dark:#FDAEB7;--shiki-dark-font-style:italic;">78点</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">，涨幅为4.</span><span style="--shiki-light:#B31D28;--shiki-light-font-style:italic;--shiki-dark:#FDAEB7;--shiki-dark-font-style:italic;">59</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">%</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">。这一天，沪深两市的成交额均非常活跃，接近或超过3.</span><span style="--shiki-light:#B31D28;--shiki-light-font-style:italic;--shiki-dark:#FDAEB7;--shiki-dark-font-style:italic;">5万亿元人民币</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">。</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">以上信息来源于多个新闻源，包括财新网、观察者网、中国新闻网、新浪财经等，您可以点击提供的链接查看更详细的信息和报道背景。请注意，这些数据和分析仅供参考，市场情况可能会随时变化。</span></span></code></pre></div><h3 id="langchain4j-支持的搜索引擎" tabindex="-1">Langchain4j 支持的搜索引擎 <a class="header-anchor" href="#langchain4j-支持的搜索引擎" aria-label="Permalink to &quot;Langchain4j 支持的搜索引擎&quot;">​</a></h3><table tabindex="0"><thead><tr><th>SearchApi 引擎</th><th>是否支持</th></tr></thead><tbody><tr><td><a href="https://www.searchapi.io/docs/google" target="_blank" rel="noreferrer">Google Web Search</a></td><td>✅</td></tr><tr><td><a href="https://www.searchapi.io/docs/google-news" target="_blank" rel="noreferrer">Google News</a></td><td>✅</td></tr><tr><td><a href="https://www.searchapi.io/docs/bing" target="_blank" rel="noreferrer">Bing</a></td><td>✅</td></tr><tr><td><a href="https://www.searchapi.io/docs/bing-news" target="_blank" rel="noreferrer">Bing News</a></td><td>✅</td></tr><tr><td><a href="https://www.searchapi.io/docs/baidu" target="_blank" rel="noreferrer">Baidu</a></td><td>✅</td></tr></tbody></table><p>任何返回 <code>organic_results</code> 数组的搜索引擎都可以被此库支持，即使不在上述列表中，只要有 <code>title</code>、<code>link</code> 和 <code>snippet</code> 字段的有机搜索结果。</p>`,17)]))}const c=i(h,[["render",e]]);export{g as __pageData,c as default};
