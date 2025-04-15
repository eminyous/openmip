from enum import StrEnum


class Library(StrEnum):
    MIPLIB = "miplib"
    MINLPLIB = "minlplib"
    QPLIB = "qplib"
    # CBLIB = "cblib"

class Collection(StrEnum):
    FULL = "full"
    BENCHMARK = "benchmark"
