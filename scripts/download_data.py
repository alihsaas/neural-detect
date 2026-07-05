from neural_detect.datasets import ensure_backbone, ensure_voc, write_download_record


def main() -> None:
    record = ensure_voc()
    record.update(ensure_backbone())
    write_download_record(record)
    for name, details in record.items():
        print(f"{name}: {details['bytes']} bytes, sha256 {details['sha256']}")


if __name__ == "__main__":
    main()
