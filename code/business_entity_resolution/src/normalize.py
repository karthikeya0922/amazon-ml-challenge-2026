"""Generic string normalization (polars expressions, vectorized).

Every rule here is either a language-agnostic string operation or a cleanup of a noise
pattern observed directly in the provided files (null markers, web-domain names, junk
prefix symbols, zero-padded numbers). Vocabulary mappings (abbreviations, state codes,
transliterations) are NOT hard-coded: they are mined from the data in ``mine_tokens.py``
and applied with ``apply_token_map`` / ``apply_part_map``.
"""
import polars as pl

# literal null markers observed in S2/S3 addresses ("null", "NULL", "<NULL>")
NULL_RE = r"(?i)<\s*null\s*>|\bnull\b"
# web-style names observed in S2/S3 ("cascade.com", "| www.shivshakti.com", "Elephantcentreeurl.Com")
WWW_RE = r"(?i)\bwww\."
TLD_RE = r"(?i)(\w)\.(com|net|org|in|co|fr|us|biz|info)\b"


def _clean(e: pl.Expr) -> pl.Expr:
    """Common cleanup -> lowercase space-separated tokens."""
    e = e.fill_null("").str.replace_all(NULL_RE, " ")
    e = e.str.replace_all(WWW_RE, " ").str.replace_all(TLD_RE, "$1 ")
    # dotted initialisms: "L.L.C." -> "LLC", "P.C." -> "PC", "S.A.S" -> "SAS"
    e = e.str.replace_all(r"\b(\p{L})\.", "$1")
    # strip accents on Latin letters only (Indic vowel signs are also combining marks: keep them)
    e = e.str.normalize("NFKD").str.replace_all(r"(\p{Latin})\p{M}+", "$1")
    e = e.str.to_lowercase()
    # keep & and + as their own tokens (their equivalence to "and" is mined, not assumed)
    e = e.str.replace_all(r"([&+])", " $1 ")
    e = e.str.replace_all(r"[^\p{L}\p{M}\p{N}&+]+", " ")
    # split letter/digit boundaries: "e0137" -> "e 0137", "1946b" -> "1946 b"
    e = e.str.replace_all(r"(\p{N})(\p{L})", "$1 $2").str.replace_all(r"(\p{L})(\p{N})", "$1 $2")
    # drop zero padding: "0137" -> "137"
    e = e.str.replace_all(r"\b0+(\d)", "$1")
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def _dedupe_adjacent(e: pl.Expr) -> pl.Expr:
    """'pb pb staffing' -> 'pb staffing' (repeated-word noise observed in S2/S3 names)."""
    return (
        e.str.split(" ")
        .list.eval(pl.element().filter(pl.element().ne_missing(pl.element().shift(1))))
        .list.join(" ")
    )


def normalize_frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add n_name, addr_parts (cleaned comma components) and n_addr (parts joined)."""
    parts = (
        pl.col("business_address")
        .fill_null("")
        .str.split(",")
        .list.eval(_clean(pl.element()))
        .list.eval(pl.element().filter(pl.element() != ""))
    )
    return lf.with_columns(
        n_name=_dedupe_adjacent(_clean(pl.col("business_name"))),
        addr_parts=parts,
    ).with_columns(n_addr=pl.col("addr_parts").list.join(" "))


def apply_token_map(col: str, mapping: dict[str, str]) -> pl.Expr:
    """Replace whole tokens of a space-separated string column using a mined mapping."""
    if not mapping:
        return pl.col(col)
    return pl.col(col).str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" ").str.strip_chars()


def apply_part_map(col: str, mapping: dict[str, str]) -> pl.Expr:
    """Replace whole address components (list column) using a mined mapping."""
    if not mapping:
        return pl.col(col)
    return pl.col(col).list.eval(pl.element().replace(mapping))
