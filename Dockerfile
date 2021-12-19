FROM python:3.8-slim

COPY --chown=root:root src /root/src/

WORKDIR /root/src

RUN pip3 install -r requirements.txt
RUN chmod +x run.py

ENV SECRET_KEY hellothere

CMD ["python", "run.py"]