# 常见的错误解决方案

> 前言：如果仔细阅读过文档，这些问题都不会出现，所以请先阅读一遍文档再运行系统

### Maven: langchat构建失败

> 作为一个开发人员，这种问题实在不该出现

请检查本地IDEA的maven配置：

1. 切换阿里国内源
2. 切换本地电脑网络
3. 使用VPN

### Consider defining a bean/Error creating bean...

此错误是因为没有修改`application.yml`中环境为`dev`，也就是没有读取到SpringBoot的`application-dev.yml`文件，默认读取`-local`文件

![](/error-1.png)

### Error: Invalid content-type: text/html; charset=utf-8

此错误一般都是因为**配置模型使用了第三方代理**产生的。一般情况第三方代理会提示修改baseUrl为xxxx.xx域名，但是其实还需要增加`/v1`后缀

![](/error-2.png)

修改后模型的BaseUrl配置应该是这样的：

![](/error-2.1.png)

### 没有匹配到模型

以及类似的错误，产生是因为后台没有配置模型信息。或者是模型信息填写的有错误导致模型配置加载失败

### Invalid content-type: application/json

正常情况下我们的聊天都是sse类型流式接口，出现此错误的原因一般是因为模型接口返回是其他的错误消息，而非标准的json类型。

这种情况需要检查后台日志，出现这种情况的可能：

1. apikey无效
2. 认证失败
3. apiKey调用受限制
4. 模型没有开通收费，不允许调用
5. ...
