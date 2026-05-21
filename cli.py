import argparse
import sys

from adapter import get_provider
from assistant import Assistant


def main():
    parser = argparse.ArgumentParser(description="argus chat CLI")
    parser.add_argument("--provider", choices=["modal", "openrouter"], default=None)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    provider = get_provider(args.provider)
    assistant = Assistant(provider, max_turns=args.max_turns)

    print(f"[argus] provider={provider.name}  persona={assistant.persona[:60]!r}")
    print("Commands: /reset  /provider modal|openrouter  /quit\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        if user_text in ("/quit", "/exit"):
            break

        if user_text == "/reset":
            assistant.reset()
            print("[history cleared]\n")
            continue

        if user_text.startswith("/provider"):
            parts = user_text.split()
            if len(parts) != 2 or parts[1] not in ("modal", "openrouter"):
                print("usage: /provider modal|openrouter\n")
                continue
            assistant.set_provider(get_provider(parts[1]))
            print(f"[switched to {parts[1]}]\n")
            continue

        try:
            reply = assistant.ask(user_text)
        except Exception as e:
            print(f"[error] {e}\n", file=sys.stderr)
            continue

        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
