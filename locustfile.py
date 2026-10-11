from locust import HttpUser, task, between


class AskUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask(self):
        self.client.post(
            "/ask",
            json={"question": "هل يجوز للقاضي يحكم لو مفيش نص قانون ينطبق؟"},
        )