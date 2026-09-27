from feature_store import refresh_feature_store


def main() -> None:
    refresh_feature_store()
    print("All feature models are available.")


if __name__ == "__main__":
    main()
