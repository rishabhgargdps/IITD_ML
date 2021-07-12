FROM python:3.8-slim-buster
WORKDIR /app
ENV FLASK_APP=API.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY . .
RUN pip install -r requirements.txt
# EXPOSE 5000
CMD ["flask", "run"]