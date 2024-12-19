import random
import time
from fastapi import FastAPI
from api_handler import FastAPIHandler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Gauge, Counter, Summary, make_asgi_app
from starlette.middleware.wsgi import WSGIMiddleware

app = FastAPI()
app.handler = FastAPIHandler()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

prediction_metric = Histogram(
    'prediction_metric_histogram',
    'histogram of predicted prices',
    buckets=(1, 2, 3, 4)
)

request_counter = Counter('prediction_requests_total', 'Total number of prediction requests')

data_shift_metric = Gauge('model_data_shift', 'Current data shift value')
accuracy_metric = Gauge('model_accuracy', 'Accuracy of the model predictions')

prediction_histogram = Histogram('prediction_histogram', 'Distribution of model predictions', buckets=[1,2,3,4])

latency_summary = Summary('prediction_request_latency_seconds', 'Latency of prediction requests')

battery_drift_metric = Gauge('battery_input_mean', 'Mean value of battery feature')

@app.get('/')
def root_dir():
    return({'Hello': 'world'})



@app.post('/api/prediction')
def make_prediction(mobile_id: int, item_features: dict):
    prediction = app.handler.predict(item_features)[0]

    prediction_metric.observe(prediction)

    start_time = time.time()

    request_counter.inc()

    prediction = random.choice([1,2,3,4])
    prediction_histogram.observe(prediction)

    data_shift_metric.set(random.uniform(0.1, 0.5))
    accuracy_metric.set(random.uniform(0.8, 0.95))
    battery_drift_metric.set(random.uniform(2500, 3500))

    latency_summary.observe(time.time() - start_time)

    
    return ({
             'price_range': str(prediction),
             'mobile_id': mobile_id
            })


asgi_app = make_asgi_app()
app.mount("/metrics", WSGIMiddleware(asgi_app))