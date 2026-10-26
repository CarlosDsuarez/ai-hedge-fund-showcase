#!/usr/bin/env python3
"""
AI Hedge Fund — Terminal Chat Interface
Run: python chat.py [--fresh] ["question"]
"""
import argparse
import re
import sys
import uuid


def _init_db() -> None:
    """Create the tables the chat writes to.

    main.py does this at import; nothing under chat/ did, so the first
    chat_decisions INSERT hit "no such table", got swallowed by the journal's
    never-raise policy, and the audit trail was silently empty.
    """
    try:
        from db.database import init_db

        init_db()
    except Exception as e:
        print(f"  [DB] init skipped, sin journal de decisiones: {e}")


def _handle_turn(question: str, force_fresh: bool, history: list,
                 session_id: str) -> None:
    from chat.context import load_context
    from chat.router import route_question
    from chat.executor import execute_routing
    from chat.actions import handle_action

    print("Cargando contexto...", end="\r", flush=True)
    ctx = load_context()

    print("Analizando...      ", end="\r", flush=True)
    routing = route_question(question, ctx, history=history)

    print("                   ", end="\r", flush=True)  # clear status line
    response, pending_action = execute_routing(routing, ctx, force_fresh)

    print(f"\nAsistente: {response}\n")

    # Record this turn in history for follow-up question support
    history.append({
        "question": question,
        "agents": routing.get("agents", ["portfolio_summary"]),
    })
    if len(history) > 3:
        history.pop(0)

    if pending_action:
        handle_action(pending_action, ctx, question=question,
                      session_id=session_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Chat con Agentes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python chat.py                              # modo interactivo
  python chat.py "cómo está mi portafolio"   # una pregunta
  python chat.py --fresh "señales del régimen"  # forzar análisis fresco
  python chat.py "audita mis stops --fresh"  # flag inline también funciona
        """
    )
    parser.add_argument("question", nargs="?", help="Pregunta (opcional — si se omite, abre REPL)")
    parser.add_argument("--fresh", action="store_true", help="Forzar análisis fresco (sin cache)")
    args = parser.parse_args()

    _init_db()
    session_id = uuid.uuid4().hex[:12]

    if args.question:
        # Single-shot mode — no history needed
        raw = args.question.strip()
        force_fresh = args.fresh or "--fresh" in raw
        question = re.sub(r'\s*--fresh\b', '', raw).strip()
        if not question:
            print("Error: empty question", file=sys.stderr)
            sys.exit(1)
        try:
            _handle_turn(question, force_fresh, history=[], session_id=session_id)
        except KeyboardInterrupt:
            print("\nInterrumpido.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Interactive REPL
    print("╔══════════════════════════════════════════════════╗")
    print("║   AI Hedge Fund — Chat con Agentes              ║")
    print("║   Escribe 'exit' o Ctrl+C para salir            ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    history: list = []  # last 3 turns for follow-up question context

    while True:
        try:
            raw = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not raw:
            continue

        if raw.lower() in ("exit", "quit", "salir", "bye"):
            print("Hasta luego.")
            break

        # Detect inline --fresh flag
        force_fresh = args.fresh or "--fresh" in raw
        question = re.sub(r'\s*--fresh\b', '', raw).strip()

        if not question:
            continue

        try:
            _handle_turn(question, force_fresh, history, session_id)
        except Exception as e:
            print(f"\nError inesperado: {e}\n")
            # Continue REPL — don't crash on a single bad turn


if __name__ == "__main__":
    main()
