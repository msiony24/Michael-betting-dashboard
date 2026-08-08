from engine.madden_roster_validator import build_validation_report, save_validation_report


def main() -> None:
    report = build_validation_report()
    json_path, csv_path = save_validation_report(report)
    summary = report["summary"]

    print("\n=== Madden 27 Roster Validation ===")
    print(f"Madden player rows: {summary['madden_player_rows']}")
    print(f"Canonical NFL teams in Madden file: {summary['madden_teams_canonical']}/32")
    print(f"Automated player ratings: {summary['rated_player_rows']} players across {summary['rated_teams']}/32 teams")
    print(f"Players with nflverse performance: {summary['players_with_nflverse_performance']}")
    print(f"Critical errors: {summary['critical_error_count']}")
    print(f"Warnings: {summary['warning_count']}")

    if report["critical_errors"]:
        print("\nCritical errors:")
        for item in report["critical_errors"]:
            print(f"- {item}")
    if report["warnings"]:
        print("\nWarnings:")
        for item in report["warnings"]:
            print(f"- {item}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")

    if not summary["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
