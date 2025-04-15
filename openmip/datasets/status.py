from enum import StrEnum


class MIPLIBStatus(StrEnum):
    EASY = "easy"
    HARD = "hard"
    OPEN = "open"


class MINLPLIBStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
