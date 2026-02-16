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
            
            # Run the agent with user input
            print("\nAgent: ", end="", flush=True)
            response = agent.run(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {e}]")
            print("Try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
