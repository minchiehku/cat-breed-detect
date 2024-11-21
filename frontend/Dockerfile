FROM nginx:latest

# 安裝 gettext 獲取 envsubst 命令
RUN apt-get update && apt-get install -y gettext-base

# 複製靜態文件到 Nginx 默認靜態文件目錄
COPY ./static /usr/share/nginx/html/static/
COPY ./index.html /usr/share/nginx/html/
COPY ./static/catBreedpic.jpg /usr/share/nginx/html/static/


# 複製自定義的 Nginx 配置模板
COPY ./nginx.conf /etc/nginx/templates/nginx.conf.template

# 設定工作目錄
WORKDIR /usr/share/nginx/html

EXPOSE 80

# 在容器啟動時替換配置並啟動 Nginx
CMD ["/bin/bash", "-c", "envsubst '${BACKEND_HOST}' < /etc/nginx/templates/nginx.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"]
