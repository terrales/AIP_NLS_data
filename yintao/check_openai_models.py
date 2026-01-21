#!/usr/bin/env python3
"""
Check available OpenAI models with your API key
"""
import os
from openai import OpenAI

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return

    print(f"API Key found (length: {len(api_key)} characters)")
    print("\nQuerying OpenAI API for available models...\n")

    try:
        client = OpenAI(api_key=api_key)

        # List all available models
        models = client.models.list()

        # Filter and categorize chat models
        chat_models = []
        other_models = []

        for model in models.data:
            model_id = model.id
            if any(x in model_id.lower() for x in ['gpt', 'chat']):
                chat_models.append(model_id)
            else:
                other_models.append(model_id)

        # Sort models
        chat_models.sort()
        other_models.sort()

        print("=" * 70)
        print("CHAT/GPT MODELS (recommended for LLM ranking):")
        print("=" * 70)
        if chat_models:
            for model in chat_models:
                print(f"  • {model}")
        else:
            print("  (none found)")

        print(f"\nTotal chat models: {len(chat_models)}")

        print("\n" + "=" * 70)
        print("OTHER MODELS:")
        print("=" * 70)
        if other_models:
            for model in other_models[:20]:  # Limit to first 20
                print(f"  • {model}")
            if len(other_models) > 20:
                print(f"  ... and {len(other_models) - 20} more")
        else:
            print("  (none found)")

        print(f"\nTotal other models: {len(other_models)}")

        # Test a simple chat completion with a common model
        print("\n" + "=" * 70)
        print("TESTING MODEL ACCESS:")
        print("=" * 70)

        # Try common model names
        test_models = ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']

        for test_model in test_models:
            if test_model in chat_models:
                print(f"\nTesting {test_model}...")
                try:
                    response = client.chat.completions.create(
                        model=test_model,
                        messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
                        max_tokens=5
                    )
                    print(f"  ✓ {test_model} works! Response: {response.choices[0].message.content}")
                    break
                except Exception as e:
                    print(f"  ✗ {test_model} error: {e}")

    except Exception as e:
        print(f"ERROR: {e}")
        print("\nThis could mean:")
        print("  • Invalid API key")
        print("  • Network connection issue")
        print("  • API endpoint not accessible")

if __name__ == '__main__':
    main()
