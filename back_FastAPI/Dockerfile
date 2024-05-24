FROM nvidia/cuda:12.2.2-devel-ubi8

# Instalamos las dependencias necesarias para compilar Python y psycopg2
RUN yum install -y make gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget \
    postgresql-devel mesa-libGL

# Descargamos el código fuente de Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xvf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make altinstall

# Limpiamos los archivos temporales
RUN rm -rf Python-3.10.0*

# Actualizamos pip
RUN python3.10 -m pip install --upgrade pip

# Instalamos Poetry
RUN python3.10 -m pip install poetry==1.6.1

# Copiamos el código de la aplicación
COPY . /app
WORKDIR /app

# Instalamos las dependencias del proyecto usando Poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Instalamos psycopg2-binary para evitar problemas de compilación
#RUN python3.10 -m pip install psycopg2-binary sqlalchemy

# Exponemos el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
