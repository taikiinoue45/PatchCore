import hashlib


def test_checksum() -> None:

    file_path_list = [
        ".github/workflows/create-release.yml",
        ".github/workflows/dev.yml",
        ".github/workflows/format-python.yml",
        ".github/workflows/format-yaml.yml",
        ".github/workflows/main.yml",
        "run.py",
        "tests/test_coverage.py",
    ]

    checksum_list = [
        "be71018b6c2b6669fa5b6a0f7229c6e9184b1937c46e20490bd2efeece3b069c",
        "e973de739342b41c45c85b4443bd24ec6ec09f4873eed13edc9da5e48673381a",
        "981bd32ad4febe14a44c1c7a0a00245c747373f39d6e22884f80f018bebc68c8",
        "e18032c3c704dc1a391d72cf8bded1c98b0dffcc80d00e84f5a78daf53e4cc0f",
        "2d11abd5a160f01a7ff286472a82c5991d986f2b8032950be3d120e9a6ec4634",
        "e85e8a7170864332c146c6b50c699b79e0ae934f1b4597f130798aca589c56f1",
    ]

    for file_path, correct_checksum in zip(file_path_list, checksum_list):

        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        assert correct_checksum == checksum, f"{file_path} checksum {checksum} is incorrect."
