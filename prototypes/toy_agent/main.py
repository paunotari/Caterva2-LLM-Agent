"""
Entry point for running the agent.

This provides a simple command-line interface to interact with the agent.
"""

from agent import Agent


def main():
    """Run the agent in an interactive loop."""
    print("=" * 60)
    print("Basic Agent with Calculator Tool")
    print("=" * 60)
    print("\nThis agent can:")
    print("  - Answer general questions")
    print("  - Perform calculations using the calculator tool")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'reset' - Clear conversation history")
    print("=" * 60)

    # Initialize the agent
    agent = Agent()


    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            print(f"[DEBUG received: {repr(user_input)}]")  # Temporary: shows exact input value

            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'reset':
                agent.reset()
                print("\n[Conversation reset]")
                continue

            if not user_input:
                continue

            # Stronger input validation
            MAX_INPUT_CHARS = 5000  # Increased from 500, but still reasonable
            if len(user_input) > MAX_INPUT_CHARS:
                print(f"[Input too long: {len(user_input)} characters. Maximum is {MAX_INPUT_CHARS}]")
                print("[Please provide a shorter input or ask a specific question about your text]")
                continue
            # Warn on very long inputs
            if len(user_input) > 2000:
                print(f"[Warning: Long input detected ({len(user_input)} chars). This may take longer to process.]")

            # Run the agent with user input
            print("\nAgent: ", end="", flush=True)
            response = agent.run(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {type(e).__name__}: {e}]")
            print("Try again or type 'quit' to exit.")
            # Reset agent to avoid corrupted conversation state after an error
            agent.reset()


if __name__ == "__main__":
    main()
