# 如何登录系统

> 注意：第一次本地部署项目，建议使用管理员账号，管理员账号拥有页面所有权限，不需要做权限配置

- **管理员账号密码：** administrator / langchat
- **体验账号密码：** langchat / langchat
- **客户端账号密码：** langchat@outlook.com / langchat

## 账号体系

LangChat使用Server/Client分离的架构设计，Server和Client两端的接口、业务是完全分离的。
因此LangChat项目中存在两台用户体系：

- `sys_user`： 管理端系统用户表
- `aigc_user`： 客户端系统用户表

## 登录

![](/login-server.png)
