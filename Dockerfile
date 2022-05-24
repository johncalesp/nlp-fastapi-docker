FROM python:3.9

RUN mkdir -p /home/app

COPY ./app /home/app

WORKDIR /home/app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]