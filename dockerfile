# Usa una imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY . /app

# Actualiza el sistema y las herramientas necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias desde requirements.txt
RUN if [ -f requirements.txt ]; then \
    while IFS= read -r pkg || [ -n "$pkg" ]; do \
        pip install --no-cache-dir "$pkg" || echo "Failed to install $pkg, skipping."; \
    done < requirements.txt; \
fi

# Expone los puertos necesarios
EXPOSE 5000
EXPOSE 8888

# Establece variables de entorno
ENV PYTHONUNBUFFERED=1

# Comando por defecto para el contenedor
CMD ["tail", "-f", "/dev/null"]