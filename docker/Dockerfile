# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#require NV driver>445 
#ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.10-tf1-py3
#require NV driver>=440.10 
ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.01-tf1-py3 

FROM $BASE_IMAGE

RUN apt update && apt install -y libgl1-mesa-glx ffmpeg

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install h5py==2.9.0
RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.2
RUN pip install tqdm==4.49.0
RUN pip install opencv-python

#Add user 
RUN useradd -ms /bin/bash haotian
USER haotian