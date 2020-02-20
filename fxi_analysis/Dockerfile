FROM nvidia/opencl:runtime
# FROM continuumio/miniconda3

RUN apt update && apt install -y libncurses5 wget git

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && bash /root/miniconda.sh -b -p /root/miniconda

ENV PATH "/root/miniconda/bin:${PATH}"

ADD environment.yml bnlxray/ icxray/ ltt/ /app/

RUN conda env update --file /app/environment.yml

RUN conda install -y --override --clobber -c conda-forge opencv=4.1 && conda install -y fftw

RUN pip install astropy netCDF4

# RUN pip install -e /app/bn

ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/root/miniconda/lib:/app/ltt_distro/readline/lib"

ADD bnlxray /bnlxray/
ADD icxray /icxray/
ADD ltt /ltt/

RUN pip install -e /icxray -e /ltt && pip install zmq
