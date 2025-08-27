
docker stop api_predict
docker rm api_predict
docker rmi api_predict
rm -rf api_predict
git clone git@github.com:Mukeshrinwa/api_predict.git
cd api_predict
docker build --no-cache -t api_predict .
docker run -d --name api_predict -p 8080:8080 api_predict
cd ..