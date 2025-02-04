# 环境准备

在运行项目之前，需要确保已经搭建好与之相匹配的开发环境。以下是基础开发环境要求：

### 前端基础环境

- Node.js > 18

建议安装并使用 `pnpm install`，不建议使用 `npm` 或 `yarn`。如果你是中国用户并遇到下载速度问题，推荐使用以下镜像源：

::: code-group

```bash [npm]
# 设置为国内镜像
npm config set registry https://registry.npmmirror.com
# 设置为官方镜像
npm config set registry https://registry.npmjs.org/
```

```bash [yarn]
# 设置为国内镜像
yarn config set registry https://registry.npmmirror.com
# 设置为官方镜像
yarn config set registry https://registry.yarnpkg.com
```

 ```bash [pnpm]
# 设置为国内镜像
pnpm config set registry https://registry.npmmirror.com
# 设置为官方镜像
pnpm config set registry https://registry.npmjs.org/
```

:::

### 后端基础环境

后端的基础环境要求如下：

- **JDK ≥ 17**
- **MySQL ≥ 8**
- **Pg Vector**
- **Redis**

由于本项目使用了最新的技术栈，这里不再考虑支持 JDK8。当然，如果你想要迁移到 JDK8，成本也并不高。

本项目选择使用 JDK17 而非 JDK8 的主要原因：

1. 使用最新的技术栈进行开发，包括前端和后端。老版本官方已经不再提供支持。
2. 在 **LLM** 项目代码中会有很多 Prompt 内容，JDK17 的 """ 文本块写法非常方便。
3. 安全性和性能的考虑。

除此之外，代码层面并没有太多变化。如果你希望将代码迁移到 JDK8，只需要从上述两个方面进行考虑即可。

### 安装PgVector

MySQL和Redis的安装这里不再说明，主要说一下PgVector向量数据库的安装：

PgVector的开源地址：[https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
你可以使用官方提供的本地安装方式。

安装后需要**创建`langchat`数据库**，创建完成即可。

**注意：** `application-dev.yml`中写了表名`vector_1`，注意此表是项目启动自动生成的，无需手动创建。

我这里提供Docker Compose一键部署脚本，此脚本在启动容器的时候会自动创建`langchat`数据库：

![](/env-pg.png)

