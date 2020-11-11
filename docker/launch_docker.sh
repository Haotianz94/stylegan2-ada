# Thanks @dustinfreeman for providing the script
#!/bin/bash
# docker build -f Dockerfile -t stylegan2-ada:tensorflow20.01 .

# nvidia-docker run -ti --ipc=host --shm-size 12G -v $(pwd):/home/haotian/stylegan2-ada --workdir=/home/haotian/stylegan2-ada stylegan2-ada:tensorflow20.01 /bin/bash

docker login

docker tag stylegan2-ada:tensorflow20.01 docker.io/haotianz/stylegan2-ada:tensorflow20.01

docker push haotianz/stylegan2-ada:tensorflow20.01
