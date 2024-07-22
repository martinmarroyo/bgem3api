FROM python:3.11
WORKDIR /opt/app

COPY models ./models
COPY main.py requirements.txt ./

RUN pip install --no-cache-dir -U -r requirements.txt
EXPOSE 80

ENTRYPOINT [ "fastapi" ]
CMD [ "run", "main.py", "--port", "80" ]