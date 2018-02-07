FROM nvidia/cuda:8.0-cudnn5-devel

# Add pip to the list of packages for upgrading since there were intermittent failures
# when installing libtiff (ImportError: No module named six)
RUn apt-get -y update
RUN apt-get install -y python-pip git curl g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev
RUN apt-get -y install python-tk tk-dev
RUN apt-get -y install vim

# Add pyenv to the load path
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin
RUN echo $PYENV_ROOT && echo $PATH

# pyenv is installed within $PYENV_ROOT
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
RUN CONFIGURE_OPTS=--enable-shared pyenv install 2.7.4

# Download deepcell from git
WORKDIR $HOME
RUN git clone https://github.com/CovertLab/DeepCell.git
RUN rm ~/DeepCell/.python-version

# Create global pyenv with python 2.7.4
RUN pyenv global 2.7.4

# Install deep learning packages (add jupyter-console)
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn scikit-image matplotlib palettable libtiff tifffile h5py ipython[all]

# The code will no longer get the bleeding-edge (dev) version of Theano, since using
# versions newer than the ones used when the DeepCell docker image was checked in introduced
# errors when the code was run. This is because later versions of Theano/Keras deprecated the
# old CUDA backend in favor of the new GPUArray backend, which requires some code change.
# Therefore, we will get the older Theano and Keras versions that are compatible with the
# code until we are ready for the GPUArray backend

# bleeding-edge version
# RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# old gpuarray backend
RUN pip install theano==0.9.0 keras==1.1.0 pywavelets mahotas

# Set up Keras environment
RUN mkdir $HOME/.keras && echo '{"image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}' >> $HOME/.keras/keras.json

# Set up Theano environment
RUN echo '[global]\ndevice = gpu\nfloatX = float32' > $HOME/.theanorc

WORKDIR $HOME/DeepCell/keras_version
