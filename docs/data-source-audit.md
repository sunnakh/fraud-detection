# Data source audit

Simulated payment-card data. Every number below was observed on 2026-09-28 by
loading the pinned files through `load_reviewed_files`. Nothing here describes
real customers, real fraud rates, or real business value.

## Source and commit

- Repository: `Fraud-Detection-Handbook/simulated-data-transformed`
- Commit: `6e3ca5849b4681430388056d3f1dcfb41d4e8269`
- Files: 183 daily pickles, `2018-04-01.pkl` to `2018-09-30.pkl`

## Terms evidence

Not checked. The project owner chose on 2026-09-28 to use the data without
reviewing the repository's license or terms. Use is limited to local learning
and demonstration; nothing derived from it is redistributed. Revisit before any
public release of data, artifacts, or results.

## Files and checksums

`config/source-pins.json` records the commit and a SHA-256 per file. The loader
verifies every checksum before deserializing any pickle, because unpickling
untrusted content can execute code. Raw files live in `data/raw/` and stay out
of Git.

## Observed schema

| Column | dtype | Used |
| --- | --- | --- |
| `TRANSACTION_ID` | int64 | Yes, as a string ID |
| `TX_DATETIME` | datetime64[ns], naive | Yes, localized to UTC |
| `CUSTOMER_ID` | int64 | Yes, as a string ID |
| `TERMINAL_ID` | int64 | Not in the first feature version |
| `TX_AMOUNT` | float64 | Yes |
| `TX_TIME_SECONDS`, `TX_TIME_DAYS` | int64 | No, derived from `TX_DATETIME` |
| `TX_FRAUD` | int64 | Label only |
| `TX_FRAUD_SCENARIO` | int64 | Never; simulator truth |
| `TX_DURING_WEEKEND`, `TX_DURING_NIGHT` | int64 | No; recomputed if needed |
| `CUSTOMER_ID_*_WINDOW` (6 columns) | float64 | No; see below |
| `TERMINAL_ID_*_WINDOW` (6 columns) | float64 | No; see below |

## Row, date, and prevalence facts

- Rows: 1,754,155. Customers: 4,990. Terminals: 10,000.
- Event time: 2018-04-01 00:00:31 to 2018-09-30 23:59:57.
- Nulls: none in any column. Duplicate `TRANSACTION_ID`: 0.
- Fraud prevalence: 0.837% (`TX_FRAUD` mean 0.00837).
- Scenario counts: 0 = 1,739,474; 1 = 973; 2 = 9,077; 3 = 4,631.
- `TX_AMOUNT`: mean 53.63, median 44.64, max 2,628.00, min 0.00.
- 42 rows have `TX_AMOUNT == 0.00`, all with `TX_FRAUD == 0`. The simulator
  redraws negative amounts and rounds to cents, so these are most likely
  rounding artifacts (inferred from the generator's design, not confirmed).
  The validator keeps zero amounts and rejects negative or non-finite ones, so
  customer history counts match the source.

## Excluded fields and why

- `CUSTOMER_ID_*_WINDOW`: precomputed windows include the current transaction,
  so they leak the event being scored. Features are recomputed from strictly
  earlier, already-available transactions.
- `TERMINAL_ID_*_RISK_*`: computed from fraud labels, which are not available
  at decision time.
- `TX_FRAUD_SCENARIO`: tells the model how fraud was generated; pure leakage.

## Simulation conventions

Copied from `src/fraud_detection/simulation.py`; each is a convention, not a fact.

- `SIMULATED_TIMEZONE = "UTC"`: naive source times are read as UTC.
- `SIMULATED_CURRENCY = "XTS"`: ISO 4217 test code; amounts have no currency.
- `SIMULATED_OUTCOME_DELAY = 7 days`: when a simulated label becomes known.
- `SIMULATED_SUPPLIER_ID = "local-sim"`: supplier for seeded history.
- Provisional late cutoff: events older than 24 hours at ingest are accepted but
  not scored.

## Known limits

- Scenario 1 fraud is defined by a fixed rule (amount above 220), so a simple
  amount rule catches it; strong results there say little about real fraud.
- Scenario 2 fraud comes from compromised terminals. The first feature version
  uses only customer history, so it is expected to miss much of this scenario.
- Scenario 3 fraud inflates some amounts for compromised customers, which
  customer amount history can detect.
- Fraud generation is simple and stationary; drift and adversarial adaptation
  do not appear unless we inject them.
