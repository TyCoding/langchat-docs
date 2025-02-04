# LangChat介绍

[LangChat](http://github.com/tycoding/langchat) 使用Java生态，前后端分离，并采用最新的技术栈开发。后端基于SpringBoot3，前端基于Vue3。
LangChat不仅为企业提供AI领域的产品解决方案，也是一个完整的Java企业级应用案例。这个系统带你全面了解SpringBoot3和Vue3的前后端开发流程、业务模块化，以及AI应用集成方案。
无论是企业开发，还是个人学习，LangChat都将为你提供丰富的学习案例。
涉及的技术栈包括：

**后端技术：**

- SpringBoot：MVC框架
- Mybatis Plus：持久层框架
- Sa-Token：权限框架
- Hutool：Java工具类
- LangChain4j：Java LLM基础框架
- AI LLM等

**前端技术：**

- Vue3
- TypeScript
- Node
- EChart
- NaiveUI

### 适合人群

- 想要学习Vue & SpringBoot前后端分离应用开发的同学
- 想要学习AI在Java生态下集成方案的同学或企业
- 需要一套快速上手AI集成方案的企业级项目
- 需要搭建企业知识库平台的企业
- 需要快速定制化开发企业机器人应用的企业
- 需要构建高级流程化编排机器人的企业
- ......

### 项目架构

本项目后端采用Java单体服务，多模块的形式开发，具备完善且规范的代码分层结构。

```text copy
.
├── LICENSE
├── README.md
├── docker
├── docs
├── langchat-biz-ops
│   ├── langchat-biz-auth
│   ├── langchat-biz-bootstrap
│   ├── langchat-upms
│   └── pom.xml
├── langchat-common
│   ├── langchat-common-ai
│   ├── langchat-common-auth
│   ├── langchat-common-bom
│   ├── langchat-common-core
│   ├── langchat-common-es
│   ├── langchat-common-oss
│   └── pom.xml
├── langchat-llm-ops
│   ├── langchat-ai
│   ├── langchat-aigc
│   ├── langchat-llm-auth
│   ├── langchat-llm-bootstrap
│   └── pom.xml
├── langchat-ui-package
│   ├── README.md
│   ├── apps
│   ├── cspell.json
│   ├── eslint.config.mjs
│   ├── internal
│   ├── node_modules
│   ├── package.json
│   ├── packages
│   ├── pnpm-lock.yaml
│   ├── pnpm-workspace.yaml
│   ├── scripts
│   ├── stylelint.config.mjs
│   ├── tea.yaml
│   ├── turbo.json
│   ├── vben-admin.code-workspace
│   ├── vitest.config.ts
│   └── vitest.workspace.ts
└── pom.xml
```
