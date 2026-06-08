from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import json


class APIUser(HttpUser):
    wait_time = between(5, 15)

    @task(10)
    def predict(self):
        with self.client.get("/predict", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, list) or len(data) == 0:
                    resp.failure(f"Réponse inattendue : {data}")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(3)
    def health(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                resp.success()
            else:
                resp.failure(f"Health KO : {resp.text}")

    @task(1)
    def model_info(self):
        self.client.get("/model/info")
