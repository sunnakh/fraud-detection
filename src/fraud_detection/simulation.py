"""Local simulation conventions for public Handbook data.

Each value is a documented simulation convention, not a real-world fact.
"""

from datetime import timedelta

# Naive Handbook TX_DATETIME values are read as UTC.
SIMULATED_TIMEZONE = "UTC"
# ISO 4217 reserves XTS for testing; Handbook amounts have no real currency.
SIMULATED_CURRENCY = "XTS"
# Simulated outcomes become available this long after the event.
SIMULATED_OUTCOME_DELAY = timedelta(days=7)
# Supplier identity used for local simulated history.
SIMULATED_SUPPLIER_ID = "local-sim"
