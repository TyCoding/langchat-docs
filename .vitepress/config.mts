import {defineConfig} from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "LangChat Docs",
  description: "LangChat Project Document",
  lastUpdated: true,
  markdown: {
    image: {
      // 默认禁用；设置为 true 可为所有图片启用懒加载。
      lazyLoading: true
    }
  },
  head: [['link', { rel: 'shortcut icon', href: '/favicon.png' }]],
  themeConfig: {
    outline: {
      level: 'deep'
    },
    search: {
      provider: 'local'
    },
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'LangChat', link: '/' },
      { text: 'LangChat文档', link: '/docs/exercise/langchat-deepseek-r1', activeMatch: '/docs' },
      {
        text: '在线预览',
        items: [
          { text: 'LangChat官网', link: 'https://langchat.cn/' },
          { text: 'LangChat后台预览', link: 'http://backend.langchat.cn/' },
          { text: 'LangChat LLM Ops', link: 'http://llm.langchat.cn' },
          { text: 'LangChat UPMS Ops', link: 'http://upms.langchat.cn' }
        ]
      }
    ],

    sidebar: {
      '/docs': [
        {
          text: 'LangChat实战',
          items: [
            { text: 'DeepSeek-R1实战', link: '/docs/exercise/langchat-deepseek-r1' },
            { text: '使用Minio作为OSS', link: '/docs/exercise/oss-minio' },
            { text: 'LLM-RAG基础概念', link: '/docs/exercise/rag' },
          ]
        },
       
        {
          text: 'LangChat配置',
          items: [
            { text: 'LangChat介绍', link: '/docs/start/introduce' },
            { text: '环境准备', link: '/docs/start/environment' },
            { text: '快速开始', link: '/docs/start/getting-started' },
            { text: '登录LangChat', link: '/docs/start/login' },
            { text: '模型配置', link: '/docs/start/models' },
            { text: '模型代理', link: '/docs/start/models-proxy' },
            { text: '知识库', link: '/docs/start/knowledge' },
            { text: '常见问题', link: '/docs/start/questions' },
          ]
        },
        {
          text: 'LangChat部署',
          items: [
            { text: 'LangChat部署教程', link: '/docs/deploy/deploy' },
          ]
        },
        {
          text: '推荐阅读',
          items: [
            { text: 'DeepSeek-R1微调指南', link: '/docs/other/deepseek-r1-tuning' },
            { text: 'DeepSeek R1架构和训练过程图解', link: '/docs/other/deepseek-r1-architecture-and-training' },
            { text: 'DeepSeek-R1蒸馏模型', link: '/docs/other/deepseek-r1-distilled-models' },
            { text: 'DeepSeek-R1的推理能力分析', link: '/docs/other/deepseek-r1-reasoning-capabilities-analysis' },
            { text: '蒸馏DeepSeek-R1到自己的模型', link: '/docs/other/distill-deepseek-r1-into-your-model' }
          ]
        },
        
      ],
      
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/TyCoding/langchat' }
    ]
  }
})
