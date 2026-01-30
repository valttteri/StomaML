FROM python:3.11

WORKDIR /frontend #location for the source code, the Dockerfile is now in the root foler

COPY . .

RUN chmod -R 777 * 

RUN pip install -r requirements.txt

EXPOSE 8050

CMD ["streamlit run", "main.py"]