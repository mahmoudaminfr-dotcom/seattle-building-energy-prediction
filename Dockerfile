FROM python:3.11-slim

WORKDIR /app

# Copie et installation des dépendances depuis requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du modèle et du service d'inférence
COPY model.joblib .
COPY service.py .

ENV PORT=3000
EXPOSE 3000

CMD ["bentoml", "serve", "service:SeattleEnergyService", "--port", "3000", "--host", "0.0.0.0"]