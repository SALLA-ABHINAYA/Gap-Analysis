from openai import AzureOpenAI


def get_azure_openai_client():
    client = AzureOpenAI(
        api_key="abfThcHeAaGdPRna0tPFYEI4yXaz6wLU0uKlDrnPA7llpeyirOJMJQQJ99ALACYeBjFXJ3w3AAABACOGvWJz",
        api_version="2024-02-01",
        azure_endpoint="https://smartcall.openai.azure.com/"
    )
    return client
