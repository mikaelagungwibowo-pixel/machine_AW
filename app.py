# =========================================================
# Chatbot Utilities
# =========================================================
REQUIRED_FEATURES = ["USIAMASUK", "IP2", "IP3", "IP5", "rata-rata nilai", "mandiri/flagsip", "BEKERJA/TIDAK"]

FEATURE_PATTERNS = {
    "USIAMASUK": re.compile(
        r"(usia(\s*masuk)?|usiamasuk|umur)\s*[:=]?\s*(\d{1,2})",
        re.I
    ),
    "IP2": re.compile(
        r"(?<!\w)ip2\s*[:=]?\s*(0-4?)",
        re.I
    ),
    "IP3": re.compile(
        r"(?<!\w)ip3\s*[:=]?\s*([0-4](?:\       re.I
    ),
    "IP5": re.compile(
        r"(?<!\w)ip5\s*[:=]?\s*([04?)",
        re.I
    ),
    "rata-rata nilai": re.compile(
        r"(rata[- ]?rata\s*nilai|nilai\s*rata[- ]?rata|rerata)\s*[:=]?\s*(\d{1,3})",
        re.I
    ),
}
