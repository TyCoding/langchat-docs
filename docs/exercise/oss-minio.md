# LangChat如何配置OSS - Minio
> langChat支持常见的OOS存储服务，这里介绍如何搭建与配置minio，本文提供四种方式。
> 1. docker安装minio
> 2. docker安装minio（宝塔面板）
> 3. 二进制文件安装minio
> 4. 二进制文件安装minio（宝塔面板）
## 安装minio
###  docker安装minio
这里假设你已经安装docker，如果没有安装docker，请参考[docker安装](https://www.docker.com/get-started)
```bash
docker run -d --name minio --restart always \
  -p 9000:9000 -p9001:9001 \
  -e MINIO_ROOT_USER=admin \
  -e MINIO_ROOT_PASSWORD=admin123 \
  -v /www/dk_project/dk_app/dk_minio/data:/data \
  bitnami/minio:latest
```
上面的命令中，-p 9000:9000 是暴露端口，-p 9001:9001 是管理端口，-e MINIO_ROOT_USER=admin 是设置用户名，-e MINIO_ROOT_PASSWORD=admin123 是设置密码，-v /www/dk_project/dk_app/dk_minio/data:/data 是设置数据存储路径，bitnami/minio:latest 是镜像名称。

### docker安装minio（宝塔面板）
这里假设你已经安装宝塔面板，如果没有安装宝塔面板，请参考[宝塔面板](https://www.bt.cn/)

1. 登录宝塔面板
2. 点击左侧菜单的“软件商店”
3. 在搜索框中输入“minio”，点击搜索结果中的“minio(Docker应用) 2.0.1”,然后选中弹窗中的安装配置，绑定域名这里根据实际情况填写，
无域名勾选“不使用域名即可”。
4. 其它参数可以保持默认，点击“安装应用(安装过则是重建应用)”按钮，等待安装完成。
![bt-docker-install-1.png](/public/minio/bt-docker-install-1.png)
5. 安装完成后，点击左侧菜单的“应用信息”，就可以看到minio的访问地址和用户名密码，即可进入minio的管理界面。  

** 特别说明：**
宝塔面板的docker应用都是基于docker-compose安装的，所以如果你想手动修改配置文件，可以在宝塔面板中`/www/dk_project/dk_app/dk_minio`，然后进入`docker-compose.yml`文件中进行修改。
### 二进制文件安装minio(只支持linux)
下载minio二进制文件，下载地址为`https://dl.min.io/server/minio/release/`。  
这里以linux-amd64为例，下载地址为`https://dl.min.io/server/minio/release/linux-amd64/minio` ，下载完成后，将minio文件移动到`/usr/local/bin`目录下，然后创建一个目录`/data`，然后执行`minio server /data`命令，即可启动minio服务。
```bash
cd /usr/local/bin/
wget https://dl.min.io/server/minio/release/linux-amd64/minio
mkdir -p /data/minio
# 以下三种选一个就行
minio server /data/minio
nohup minio server --console-address ":9001" --address ":9000" -C /data/minio &

```
安装完成之后，如果刚刚启动时指定了access-key和secret-key，那么在浏览器中访问`http://localhost:9001`，即可进入minio的登录界面，用户名为admin，密码为admin123。否则也可以使用默认用户名minioadmin和人默认密码minioadmin登录。

### 二进制文件安装minio（宝塔面板）
在宝塔面板的软件商店中搜索minio，然后安装即可，不要选择docker版应用，该安装方式本人未实际操作，只提供参考。

## 配置minio到LangChat
### 配置minio
##### 配置access-key和secret-key
在浏览器中访问`http://localhost:9001`,输入安装时设置的用户名和密码，进入minio的管理界面。在页面左侧点击“Access Keys”，新增access-key和secret-key。
![create-key-1.png](/public/minio/create-key-1.png)
![create-key-2.png](/public/minio/create-key-2.png)
![create-key-2.png](/public/minio/create-key-2.png)
**记得保存下生成的key，后面会用到**。

##### 创建桶
在浏览器中访问`http://localhost:9001`,输入安装时设置的用户名和密码，进入minio的管理界面。在页面左侧点击“Buckets”，
我这里新建一个桶，名称为ycs，然后点击“Create Bucket”，即可创建一个桶。  
**这里的桶的Access Policy一定要配置为“public” **。
![create-bucket-1.png](/public/minio/create-bucket-1.png)
![create-bucket-2.png](/public/minio/create-bucket-2.png)


### LangChat配置
##### langchat-common-oss的pom.xml增加对minio的依赖
```xml
<dependency>
    <groupId>io.minio</groupId>
    <artifactId>minio</artifactId>
    <version>8.5.12</version>
</dependency>
```
##### 修改application-dev.yml文件，增加以下配置
```yaml
langchat:
  oss:
    default-platform: minio
    minio:
      - platform: minio # 存储平台标识，七牛：qiniu、阿里OSS：aliyun-oss、腾讯OSS：tencent-cos
        enable-storage: true  # 启用存储
        access-key: w6YFEW7eugorhgrbUqR9
        secret-key: eN3ezX6yXXVKXEizxDynWwVaF1cDiWXMCAaLZOdQ
        bucket-name: ycs
        domain: http://47.119.118.6:9000/ycs # 访问域名，注意“/”结尾，例如：http://abc.hn-bkt.clouddn.com/
        base-path: / # 基础路径
        end-point: http://47.119.118.6:9000/
```
**注意：**
这里的`accessKey`和`secretKey`是你在minio中创建的key，`bucketName`是你的桶名称，`endpoint`是minio服务的地址，`pathStyleAccessEnabled`设置为true。
### 测试上传文件
在LangChat项目中，找到知识库管理模块，点击上传文件，上传一个文件，即可测试上传文件功能。如果上传成功，在minio中可以看到上传的文件，并且可以下载；如果解析成功，可以在知识库看到对应文成生成的切片数据。

