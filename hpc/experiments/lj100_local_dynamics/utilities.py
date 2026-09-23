import hashlib
from pathlib import Path

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            value.update(part)
    return value.hexdigest()


def digest(state):
    value = hashlib.sha256()
    for key in sorted(state):
        value.update(key.encode())
        value.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()

