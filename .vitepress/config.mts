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
    search: {
      provider: 'local'
    },
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'LangChat', link: '/' },
      { text: 'LangChat文档', link: '/langchat/docs/introduce',activeMatch: '/langchat/' },
      { text: 'Java AI课程', link: '/course/docs/00Introduce',activeMatch: '/course/' },
      {
        text: '在线预览',
        items: [
          { text: 'LangChat LLM Ops', link: 'http://llm.langchat.cn' },
          { text: 'LangChat UPMS Ops', link: 'http://upms.langchat.cn' }
        ]
      }
    ],

    sidebar: {
      '/langchat/': [
        {
          text: '写在前面',
          items: [
            { text: 'LangChat介绍', link: '/langchat/docs/introduce' },
            { text: 'RAG基础概念', link: '/langchat/docs/rag' },
            { text: '常见问题', link: '/langchat/docs/questions' },
            { text: '更多文档', link: '/langchat/docs/show' },
          ]
        },
        {
          text: '运行LangChat',
          items: [
            { text: '环境准备', link: '/langchat/docs/run/environment' },
            { text: '快速开始', link: '/langchat/docs/run/getting-started' },
            { text: '登录', link: '/langchat/docs/run/login' },
          ]
        },
        {
          text: 'LangChat使用',
          items: [
            { text: '模型配置', link: '/langchat/docs/server/models' },
            { text: '模型代理', link: '/langchat/docs/server/models-proxy' },
            { text: '知识库', link: '/langchat/docs/server/knowledge' },
          ]
        }
      ],
      '/course': [
        {
          text: 'LangChain4j基础',
          items: [
            { text: '开始之前', link: '/course/docs/00Introduce' },
            { text: '模型选择', link: '/course/docs/01Models' },
            { text: 'LangChain4j 介绍', link: '/course/docs/02LangChain4j' },
            { text: 'Chat API 上手', link: '/course/docs/03ChatAPI' },
            { text: 'API 进阶配置', link: '/course/docs/04ChatSetting' },
            { text: 'Chat 流式输出', link: '/course/docs/05ChatStream' },
            { text: 'Chat 视觉理解', link: '/course/docs/06ChatVL' },
            { text: 'Chat 记忆缓存', link: '/course/docs/07ChatMemory' },
            { text: '提示词工程', link: '/course/docs/08Prompt' },
            { text: 'JSON 结构化输出', link: '/course/docs/09JSONOutput' },
            { text: '业务动态 JSON 结构化', link: '/course/docs/10DynamicJSON' },
            { text: '向量化及存储', link: '/course/docs/12Embedding' },
            { text: '文本向量化分类', link: '/course/docs/13EmbeddingText' },
            { text: '动态函数调用', link: '/course/docs/14DynamicFunctionCall' },
          ]
        },
        {
          text: 'RAG',
          items: [
            { text: 'RAG API 基础', link: '/course/docs/15RAGAPI' },
            { text: 'RAG API 增强', link: '/course/docs/15RAGAPI2' },
            { text: 'RAG Easy 快速上手', link: '/course/docs/16EasyRag' },
            { text: 'RAG 结果重排', link: '/course/docs/17RAGRank' },
            { text: '函数增强搜索', link: '/course/docs/18WebSearch' },
            { text: '模型敏感词处理', link: '/course/docs/19SensitiveWord' },
            { text: 'RAG 进阶分享', link: '/course/docs/20Rag' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/TyCoding/langchat' }
    ]
  }
})
