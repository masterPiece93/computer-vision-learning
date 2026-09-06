from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def test_endpoint(self):
        # Appends to the host specified in the dashboard UI
        self.client.get("/?image_url=x.png") 
