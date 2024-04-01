FROM pytorch/pytorch:latest
RUN pip install --upgrade pip

WORKDIR /code
COPY . .
RUN pip install albumentations
RUN pip install addict
RUN pip install wandb