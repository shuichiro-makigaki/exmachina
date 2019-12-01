FROM python:3

ARG CMAKE_VERSION=3.10.3

RUN git clone https://github.com/shuichiro-makigaki/exmachina.git
WORKDIR /exmachina
RUN pip install -r requirements.txt
RUN dvc get . knn_index/small
RUN ln -s small/* .
RUN dvc get . knn_index/medium
RUN ln -s medium/* .

WORKDIR /usr/local/src
RUN apt-get update
RUN apt-get -y install libhdf5-dev liblz4-dev
ADD https://cmake.org/files/v3.10/cmake-${CMAKE_VERSION}-Linux-x86_64.sh .
RUN chmod +x cmake-${CMAKE_VERSION}-Linux-x86_64.sh
RUN ./cmake-${CMAKE_VERSION}-Linux-x86_64.sh --skip-license --prefix=/usr/local
RUN git clone https://github.com/mariusmuja/flann
WORKDIR /usr/local/src/flann
RUN mkdir build
WORKDIR /usr/local/src/flann/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOC=OFF ..
RUN make
RUN make install

WORKDIR /exmachina
ENTRYPOINT [ "python", "exmachina.py" ]
CMD [ "--help" ]
