from engine.madden_personnel_audit import build_personnel_audit

if __name__ == "__main__":
    report=build_personnel_audit()
    print(report["summary"])
