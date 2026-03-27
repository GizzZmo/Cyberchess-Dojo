"""
Chess agent package for Cyberchess-Dojo.

Exports the four specialist agents used by the ``ChessOrchestrator``:

- ``OpeningAgent``    — ECO theory, development, castling, central control.
- ``TacticalAgent``   — Checks, captures, forks, pins, skewers, forcing sequences.
- ``PositionalAgent`` — Pawn structure, piece activity, weak squares, long-term strategy.
- ``EndgameAgent``    — King centralisation, passed pawns, pawn promotion, technique.

Each agent extends ``BaseChessAgent`` (``agents/base_agent.py``) and implements
``_build_prompt`` to provide a domain-specific system/user prompt to the LLM.
"""

from agents.opening_agent import OpeningAgent
from agents.tactical_agent import TacticalAgent
from agents.positional_agent import PositionalAgent
from agents.endgame_agent import EndgameAgent

__all__ = ["OpeningAgent", "TacticalAgent", "PositionalAgent", "EndgameAgent"]
