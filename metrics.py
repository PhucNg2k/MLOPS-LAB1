from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Gauge
from fastapi.responses import Response

app = FastAPI()

# Prometheus metrics
# training_progress = Gauge('training_progress_percentage', 'Training progress as a percentage')
# training_loss = Gauge('training_loss', 'Current training loss')
# training_accuracy = Gauge('training_accuracy', 'Current training accuracy')
# epochs_completed = Counter('epochs_completed', 'Number of completed epochs')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 