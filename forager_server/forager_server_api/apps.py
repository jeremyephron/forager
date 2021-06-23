from django.apps import AppConfig


class ForagerServerApiConfig(AppConfig):
    name = 'forager_server_api'

    def ready(self):
        import forager_server_api.signals
