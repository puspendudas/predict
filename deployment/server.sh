docker stop api_predict
docker rm api_predict
docker rmi api_predict
rm -rf Server
git clone git@github.com:puspendudas/predict.git Server
cd Server
docker build --no-cache -t api_predict .
docker run -d --name api_predict -p 7000:7000 api_predict
cd ..
