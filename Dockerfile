FROM stimela/base:1.0.1
MAINTAINER <sphemakh@gmail.com>
RUN docker-apt-install python-casacore \
    casacore-dev \
    python-numpy \
    python-setuptools \
    libboost-python-dev \
    libcfitsio-dev \
    wcslib-dev
ADD . /tmp/build
RUN pip install --upgrade pip setuptools
RUN pip install /tmp/build

