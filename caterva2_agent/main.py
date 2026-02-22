"""
CLI entry point for the Caterva2 dataset exploration agent.

Runs an interactive loop where the user can ask natural-language questions
about datasets on the configured Caterva2 subscriber.

Commands:
  quit / exit  — exit the program
  reset        — clear conversation history and token counter
"""

from agent import Agent
from config import CATERVA2_URLBASE


def main():
    """Run the Caterva2 agent in an interactive command-line loop."""
    print("=" * 60)
    print("Caterva2 Dataset Exploration Agent")
    print("=" * 60)
    print(f"\nConnected to: {CATERVA2_URLBASE}")
    print("\nThis agent can:")
    print("  - List available dataset roots on the server")
    print("  - Browse datasets within a root")
    print("  - Describe a dataset's shape, type, and compression")
    print("\nCommands:")
    print("  'quit' or 'exit'  — exit the program")
    print("  'reset'           — clear conversation history")
    print("=" * 60)

    agent = Agent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
            print(f"[DEBUG received: {repr(user_input)}]")

            # Handle control commands
            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset()
                print("[Conversation reset]")
                continue

            if not user_input:
                continue

            # Input length guard
            MAX_INPUT_CHARS = 5000
            if len(user_input) > MAX_INPUT_CHARS:
                print(f"[Input too long: {len(user_input)} chars. Max is {MAX_INPUT_CHARS}]")
                continue

            if len(user_input) > 2000:
                print(f"[Warning: Long input ({len(user_input)} chars) — may take longer to process]")

            # Run the agent
            print("\nAgent: ", end="", flush=True)
            response = agent.run(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {type(e).__name__}: {e}]")
            print("Try again or type 'reset' to clear the conversation.")
            # Reset after unhandled exception to avoid corrupted conversation state
            agent.reset()


if __name__ == "__main__":
    main()
