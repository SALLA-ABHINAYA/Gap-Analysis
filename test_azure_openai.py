from openai import OpenAI


def test_openai_connection(api_key: str) -> str:
    """
    Test OpenAI connection with proper configuration

    Args:
        api_key (str): OpenAI API key

    Returns:
        str: Response from the model or error message
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(
            api_key="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"
        )

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Or use "gpt-3.5-turbo" for a more economical option
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I am going to Paris, what should I see?"}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"


# Example usage
if __name__ == "__main__":
    API_KEY = "your-openai-api-key"  # Replace with your OpenAI API key

    result = test_openai_connection(API_KEY)
    print(result)