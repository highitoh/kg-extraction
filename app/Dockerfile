FROM python:3.13

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

# Workspace
RUN mkdir -p /workspace