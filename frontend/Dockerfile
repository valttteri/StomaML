FROM python:3.11

WORKDIR /frontend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8050

ENTRYPOINT ["/entrypoint.sh"]
#CMD ["streamlit", "run", "main.py", "--server.port=8050", "--server.address=0.0.0.0"]
