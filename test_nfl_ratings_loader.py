from engine.nfl_ratings_loader import load_team_quality


def main():
    chiefs = load_team_quality("Kansas City Chiefs")
    bills = load_team_quality("Buffalo Bills")

    print()
    print("Kansas City Chiefs")
    print(chiefs)
    print()

    print("Buffalo Bills")
    print(bills)
    print()


if __name__ == "__main__":
    main()
