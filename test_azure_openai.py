from openai import AzureOpenAI

# Azure OpenAI settings
client = AzureOpenAI(
    api_key="FZyDVammGYgt1KJNmnD6zE3klmMoNJlRiGjERiFAG6VT5lnGGRJzJQQJ99ALACYeBjFXJ3w3AAABACOGT8qK",
    api_version="2024-02-15-preview",
    azure_endpoint="https://smartcall.openai.azure.com/"
)

try:
    response = client.chat.completions.create(
        model="gpt-4o",  # Your deployment name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am going to Paris, what should I see?"}
        ],
        temperature=0.7,
        max_tokens=800,
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Detailed error: {str(e)}")