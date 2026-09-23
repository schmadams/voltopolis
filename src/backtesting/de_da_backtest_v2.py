import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

from helpers.check_env import check_env
from helpers.supabase_db import SupabaseDB

check_env()

# Where per-run CSV output gets written. Override by setting the
# BESS_BACKTEST_OUTPUT_DIR environment variable if you want a different
# location (e.g. on a machine where this OneDrive path doesn't exist).
OUTPUT_DIR = Path(
    os.environ.get(
        "BESS_BACKTEST_OUTPUT_DIR",
        r"C:\Users\samad\OneDrive\Documents\lattis prep\german bess\backtest results",
    )
)

# Supabase/PostgREST caps each request at 1000 rows by default, so wide
# date ranges need paginating with .range() rather than a single .execute().
PAGE_SIZE = 1000

# Assumption: region code for Germany in the auction_prices table is "DE_LU"
# (post-2018 German/Luxembourg bidding zone). Change if your table uses "DE".
DE_REGION = "DE_LU"

# BESS specs
POWER_MW = 1.0
ENERGY_MWH = 2.0
ROUND_TRIP_EFFICIENCY = 0.8

# Split round-trip efficiency evenly across charge and discharge legs
CHARGE_EFFICIENCY = np.sqrt(ROUND_TRIP_EFFICIENCY)
DISCHARGE_EFFICIENCY = np.sqrt(ROUND_TRIP_EFFICIENCY)


def read_de_da_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Read DE day-ahead auction prices from the auction_prices table for the
    given delivery_start date range, paginating as needed.
    """
    db = SupabaseDB()
    rows = []
    offset = 0

    while True:
        response = (
            db.client.table("auction_prices")
            .select("delivery_start, region, auction, price")
            .eq("region", DE_REGION)
            .eq("auction", "day-ahead")
            .gte("delivery_start", start.isoformat())
            .lte("delivery_start", end.isoformat())
            .order("delivery_start")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )

        page = response.data
        if not page:
            break

        rows.extend(page)
        offset += PAGE_SIZE

        if len(page) < PAGE_SIZE:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["delivery_start"] = pd.to_datetime(df["delivery_start"])

    return df


def optimise_day(prices: np.ndarray, dt_hours: float) -> dict:
    """
    Solve the perfect-foresight dispatch LP for a single day, treated as
    independent (SoC starts and ends at 0). Decision variables are
    per-period charge and discharge power, [c_1..c_T, d_1..d_T], in MW.

    Returns per-period charge/discharge power, state of charge at the end
    of each period, and period-level cash P&L, alongside the day's total
    profit - everything needed to reconstruct a day interval-by-interval.
    """
    n_periods = len(prices)

    # Objective: minimise (charge cost - discharge revenue), i.e.
    # minimise price*c - price*d, since linprog minimises by default.
    #
    # --- bug fix vs. the original version -----------------------------
    # The original objective was `prices` / `-prices` with no dt_hours
    # factor, so it implicitly treated every period as 1 hour of
    # dispatch. That's only correct when dt_hours == 1 (hourly data).
    # price [EUR/MWh] * power [MW] is an instantaneous rate, not cash -
    # you have to multiply by the period length in hours to get the EUR
    # actually earned/spent in that interval. For 15-min data
    # (dt_hours = 0.25) the old objective overstated profit by 1/dt_hours,
    # i.e. 4x, even though the SoC constraints below were already
    # correctly scaled by dt_hours.
    # --------------------------------------------------------------------
    cost_vector = dt_hours * np.concatenate([prices, -prices])

    # Power limits: 0 <= c_t <= POWER_MW, 0 <= d_t <= POWER_MW
    bounds = [(0, POWER_MW)] * n_periods + [(0, POWER_MW)] * n_periods

    # SoC at end of period t = dt * cumsum(charge_eff * c - d / discharge_eff)
    # Need 0 <= SoC_t <= ENERGY_MWH for every t, and SoC_T = 0.
    charge_contribution = dt_hours * CHARGE_EFFICIENCY
    discharge_contribution = dt_hours / DISCHARGE_EFFICIENCY

    lower_tri = np.tril(np.ones((n_periods, n_periods)))
    soc_from_charge = lower_tri * charge_contribution
    soc_from_discharge = lower_tri * discharge_contribution

    # SoC_t <= ENERGY_MWH  ->  soc_from_charge @ c - soc_from_discharge @ d <= ENERGY_MWH
    A_ub_upper = np.hstack([soc_from_charge, -soc_from_discharge])
    b_ub_upper = np.full(n_periods, ENERGY_MWH)

    # SoC_t >= 0  ->  -soc_from_charge @ c + soc_from_discharge @ d <= 0
    A_ub_lower = np.hstack([-soc_from_charge, soc_from_discharge])
    b_ub_lower = np.zeros(n_periods)

    A_ub = np.vstack([A_ub_upper, A_ub_lower])
    b_ub = np.concatenate([b_ub_upper, b_ub_lower])

    # SoC_T = 0 (end the day empty)
    A_eq = np.hstack([soc_from_charge[-1:], -soc_from_discharge[-1:]])
    b_eq = np.array([0.0])

    result = linprog(
        cost_vector,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return {
            "profit": np.nan,
            "charge": None,
            "discharge": None,
            "soc": None,
            "period_profit": None,
        }

    charge = result.x[:n_periods]
    discharge = result.x[n_periods:]

    # Cash flow per period, at the grid meter: what you actually pay for
    # power drawn and get paid for power exported in that interval.
    # Efficiency losses are already reflected in how much power the LP
    # chose to move (via the SoC constraints) - there's no separate cash
    # term for them, you're simply never paid for the energy lost to
    # round-trip inefficiency.
    period_profit = prices * (discharge - charge) * dt_hours

    # State of charge at the END of each period (MWh held in the battery)
    soc = np.cumsum(charge_contribution * charge - discharge_contribution * discharge)

    profit = float(period_profit.sum())

    return {
        "profit": profit,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "period_profit": period_profit,
    }


def build_interval_detail(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the daily independent-SoC optimisation across all days present in
    prices_df and return ONE ROW PER MARKET INTERVAL (e.g. every 15 minutes),
    with columns:

      delivery_start      - interval start timestamp
      date                - calendar date (for grouping/filtering)
      price_eur_mwh        - day-ahead price for that interval
      charge_mw            - battery charge power in that interval
      discharge_mw          - battery discharge power in that interval
      net_mw                - discharge_mw - charge_mw (+ve = exporting)
      action                - "charge" / "discharge" / "idle" label
      soc_mwh               - state of charge at the END of that interval
      period_profit_eur     - cash P&L earned/spent in that interval
      cum_profit_day_eur    - running total of period_profit_eur, reset each day

    This is the table to filter down to a single `date` and step through
    by hand to sanity-check the optimiser's dispatch decisions.
    """
    prices_df = prices_df.sort_values("delivery_start").copy()
    prices_df["date"] = prices_df["delivery_start"].dt.date

    interval_rows = []

    for date, day_df in prices_df.groupby("date"):
        day_df = day_df.sort_values("delivery_start").reset_index(drop=True)
        prices = day_df["price"].to_numpy()

        # Infer resolution in hours from consecutive timestamps
        deltas = day_df["delivery_start"].diff().dropna()
        dt_hours = deltas.dt.total_seconds().median() / 3600 if not deltas.empty else 1.0

        result = optimise_day(prices, dt_hours)

        if result["charge"] is None:
            # Optimiser failed for this day - still emit rows so the gap is
            # visible in the output rather than silently disappearing.
            day_rows = pd.DataFrame(
                {
                    "delivery_start": day_df["delivery_start"],
                    "date": date,
                    "price_eur_mwh": prices,
                    "charge_mw": np.nan,
                    "discharge_mw": np.nan,
                    "net_mw": np.nan,
                    "action": "optimiser_failed",
                    "soc_mwh": np.nan,
                    "period_profit_eur": np.nan,
                }
            )
            day_rows["cum_profit_day_eur"] = np.nan
            interval_rows.append(day_rows)
            continue

        charge = result["charge"]
        discharge = result["discharge"]

        action = np.where(
            charge > 1e-6, "charge", np.where(discharge > 1e-6, "discharge", "idle")
        )

        day_rows = pd.DataFrame(
            {
                "delivery_start": day_df["delivery_start"],
                "date": date,
                "price_eur_mwh": prices,
                "charge_mw": charge,
                "discharge_mw": discharge,
                "net_mw": discharge - charge,
                "action": action,
                "soc_mwh": result["soc"],
                "period_profit_eur": result["period_profit"],
            }
        )
        day_rows["cum_profit_day_eur"] = day_rows["period_profit_eur"].cumsum()

        interval_rows.append(day_rows)

    return pd.concat(interval_rows, ignore_index=True)


def build_daily_summary(interval_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse interval-level detail back to one row per day, for the top-level PnL chart."""
    summary = (
        interval_df.groupby("date")
        .agg(
            n_periods=("price_eur_mwh", "size"),
            profit_eur=("period_profit_eur", "sum"),
            end_of_day_soc_mwh=("soc_mwh", "last"),
        )
        .reset_index()
    )
    summary["cumulative_profit_eur"] = summary["profit_eur"].cumsum()
    return summary


def get_day_detail(interval_df: pd.DataFrame, date) -> pd.DataFrame:
    """Slice out a single day's interval-level detail, e.g. for eyeballing in a notebook."""
    date = pd.Timestamp(date).date()
    return interval_df[interval_df["date"] == date].reset_index(drop=True)


def export_daily_csvs(interval_df: pd.DataFrame, out_dir: Path | None = None) -> None:
    """Write one CSV per calendar day, named YYYY-MM-DD.csv, for manual review.

    Defaults to OUTPUT_DIR / "daily_detail" if out_dir isn't given.
    """
    out_dir = Path(out_dir) if out_dir is not None else OUTPUT_DIR / "daily_detail"
    out_dir.mkdir(parents=True, exist_ok=True)
    for date, day_df in interval_df.groupby("date"):
        day_df.to_csv(out_dir / f"{date}.csv", index=False)


def validate_interval_detail(interval_df: pd.DataFrame, tol: float = 1e-4) -> pd.DataFrame:
    """
    Cheap per-day self-checks, useful for spotting a broken day before
    digging into the CSV by hand:

      - soc_below_zero              SoC dipped below 0
      - soc_above_capacity          SoC exceeded ENERGY_MWH
      - soc_not_empty_at_end        SoC didn't return to ~0 by the last interval
                                     (expected under the independent-day, end-empty
                                     assumption baked into optimise_day)
      - simultaneous_charge_discharge  charge and discharge both non-zero in the
                                     same interval (feasible for the LP, but never
                                     optimal, so likely signals a modelling issue
                                     if it shows up)

    Returns one row per day; a day with all four flags False is clean.
    """
    checks = []
    for date, day_df in interval_df.groupby("date"):
        soc = day_df["soc_mwh"].to_numpy()
        charge = day_df["charge_mw"].to_numpy()
        discharge = day_df["discharge_mw"].to_numpy()

        checks.append(
            {
                "date": date,
                "soc_below_zero": bool(np.any(soc < -tol)),
                "soc_above_capacity": bool(np.any(soc > ENERGY_MWH + tol)),
                "soc_not_empty_at_end": bool(abs(soc[-1]) > tol) if len(soc) else True,
                "simultaneous_charge_discharge": bool(np.any((charge > tol) & (discharge > tol))),
            }
        )
    return pd.DataFrame(checks)


def plot_daily_pnl(daily_summary: pd.DataFrame) -> None:
    """
    Plot daily and cumulative PnL from the backtest results.
    Expects the output of build_daily_summary() (columns: date, profit_eur,
    cumulative_profit_eur).
    """
    daily_summary = daily_summary.sort_values("date")

    fig = px.bar(
        daily_summary,
        x="date",
        y="profit_eur",
        title="Daily PnL",
        labels={"date": "Date", "profit_eur": "Daily profit (EUR)"},
    )
    fig.add_scatter(
        x=daily_summary["date"],
        y=daily_summary["cumulative_profit_eur"],
        name="Cumulative profit",
        yaxis="y2",
        mode="lines",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis2=dict(title="Cumulative profit (EUR)", overlaying="y", side="right"),
        showlegend=False,
    )
    fig.show()


if __name__ == "__main__":
    start = pd.Timestamp("2022-01-01", tz="Europe/Brussels")
    end = pd.Timestamp.now(tz="Europe/Brussels")

    prices_df = read_de_da_prices(start, end)

    # One row per market interval (e.g. 15 minutes), across the whole
    # backtest window: price, battery action, running SoC, and P&L.
    # Filter by `date` (in pandas or Excel) to step through any single day.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    interval_detail = build_interval_detail(prices_df)
    interval_detail.to_csv(OUTPUT_DIR / "interval_detail.csv", index=False)

    # Uncomment to also get one CSV file per calendar day, written to
    # OUTPUT_DIR / "daily_detail":
    # export_daily_csvs(interval_detail)

    # Sanity checks - print anything that looks off before trusting the numbers.
    checks = validate_interval_detail(interval_detail)
    flagged = checks[checks.drop(columns="date").any(axis=1)]
    if not flagged.empty:
        print("Days flagged by validate_interval_detail():")
        print(flagged)
    else:
        print("validate_interval_detail(): all days clean.")

    daily_summary = build_daily_summary(interval_detail)

    print(daily_summary)
    print(f"\nTotal profit: EUR {daily_summary['profit_eur'].sum():,.0f}")
    print(f"Average daily profit: EUR {daily_summary['profit_eur'].mean():,.0f}")

    plot_daily_pnl(daily_summary)