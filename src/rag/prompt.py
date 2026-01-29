from groq import Groq


def llm(prompt):
    """Generate response using Groq API"""
    try:
        client = Groq(api_key="Groq API Key")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    print("=" * 50)
    print("Groq AI Chat - Ask me anything!")
    print("Type 'exit', 'quit', or 'q' to stop")
    print("=" * 50)
    print()

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        print("\nGroq: ", end="")
        answer = llm(prompt=prompt)
        print(answer)
        print("\n" + "-" * 50 + "\n")
