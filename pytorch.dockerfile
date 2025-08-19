# https://hub.docker.com/r/pytorch/pytorch/tags

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# copy requirements file
COPY requirements.txt .

# install requirements
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir jupyterlab

EXPOSE 8888 

# Run JupyterLab when the container launches
CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
