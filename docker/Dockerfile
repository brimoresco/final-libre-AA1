FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos del directorio actual en el contenedor
COPY . .


CMD ["python", "inferencia.py"]
