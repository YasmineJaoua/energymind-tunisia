"""
escalation_logic.py
Confidence- and impact-based routing between autonomous handling and
required human approval, following the escalation table defined in
the Project Proposal. Three impact tiers (low/medium/high) give a
genuine middle ground between silent logging and immediate escalation.
"""

import pandas as pd
from pathlib import Path
from enum import Enum

IMPACT_LEVELS = {
    "outage": "high",
    "voltage_sag": "high",
    "voltage_swell": "high",
    "frequency_deviation": "high",
    "falsified_spike": "high",   # possible cyberattack signature
    "too_clean": "high",         # possible cyberattack signature
    "thd_spike": "medium",
    "stuck_sensor": "medium",
    "bias": "low",
    "drift": "low",
}

CONFIDENCE_THRESHOLD = 0.75


class Action(str, Enum):
    AUTONOMOUS_LOG = "Log + low-priority dashboard notification"
    PERIODIC_REVIEW = "Autonomous log, flagged for periodic review"
    NOTIFY_24H = "Log + notify, review within 24 hours"
    ESCALATE = "Escalate to human operator immediately"
    ESCALATE_PRECAUTION = "Escalate to human operator (precaution)"


def get_impact(anomaly_type):
    return IMPACT_LEVELS.get(anomaly_type, "medium")


def decide_action(confidence, anomaly_type):
    """
    Three-tier impact logic:
      High impact  -> escalate immediately (high conf) or as a
                       precaution (low conf) - never stays silent.
      Medium impact -> notify within 24h either way - genuinely
                       between "ignore" and "escalate now."
      Low impact   -> autonomous log, with or without periodic review
                       flagging depending on confidence.
    """
    impact = get_impact(anomaly_type)
    high_confidence = confidence >= CONFIDENCE_THRESHOLD

    if impact == "high":
        return Action.ESCALATE if high_confidence else Action.ESCALATE_PRECAUTION
    elif impact == "medium":
        return Action.NOTIFY_24H
    else:  # low impact
        return Action.AUTONOMOUS_LOG if high_confidence else Action.PERIODIC_REVIEW


def route_event(verdict, confidence, anomaly_type, explanation):
    action = decide_action(confidence, anomaly_type)
    return {
        "verdict": verdict,
        "anomaly_type": anomaly_type,
        "confidence": confidence,
        "impact": get_impact(anomaly_type),
        "action": action.value,
        "explanation": explanation,
    }


def main():
    example_events = [
        ("sensor_fault", 0.98, "bias", "Voltage reading inconsistent with measured current/power."),
        ("genuine_grid_event", 0.55, "voltage_sag", "Voltage dropped sharply; current rose consistently."),
        ("sensor_fault", 0.60, "too_clean", "Signal variance far below normal for this window."),
        ("genuine_grid_event", 0.97, "outage", "Voltage, current, and power all dropped to zero together."),
        ("sensor_fault", 0.80, "drift", "Voltage has been steadily rising over the past hour."),
        ("sensor_fault", 0.85, "stuck_sensor", "Voltage has not changed at all for 40 minutes."),
        ("genuine_grid_event", 0.90, "thd_spike", "Harmonic distortion elevated, consistent with equipment fault."),
    ]

    print("--- Escalation routing demonstration ---\n")
    log = []
    for verdict, confidence, atype, explanation in example_events:
        record = route_event(verdict, confidence, atype, explanation)
        log.append(record)
        print(
            f"[{atype}] verdict={verdict}, confidence={confidence:.0%}, "
            f"impact={record['impact']} -> {record['action']}"
        )

    log_df = pd.DataFrame(log)
    log_path = Path("reports") / "escalation_log_sample.csv"
    log_path.parent.mkdir(exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nSaved sample escalation log to {log_path}")


if __name__ == "__main__":
    main()