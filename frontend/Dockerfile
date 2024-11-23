FROM nginx:latest

# 安裝 gettext 以獲取 envsubst 命令
RUN apt-get update && apt-get install -y gettext-base

# 複製靜態文件到 NGINX 默認靜態文件目錄
COPY ./static /usr/share/nginx/html/static/
COPY ./index.html /usr/share/nginx/html/

# 複製自定義的 NGINX 配置模板
COPY ./nginx.conf.template /etc/nginx/templates/nginx.conf.template

# 設定工作目錄
WORKDIR /usr/share/nginx/html

EXPOSE 80

# 啟動時使用 envsubst 替換環境變數，並啟動 NGINX
CMD ["/bin/bash", "-c", "envsubst '${BACKEND_SERVICE_ENDPOINT}' < /etc/nginx/templates/nginx.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"]
