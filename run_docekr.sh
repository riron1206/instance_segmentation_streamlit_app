docker build -t pytorch190_cu111_detectron2 -f Dockerfile .
docker run -p 8888:8888 -it -v $PWD/:/work -v /media/syokoi/vol1:/volume --ipc=host --rm --gpus all pytorch190_cu111_detectron2 /bin/bash
cd ../work/app
streamlit run app.py --server.port 8888