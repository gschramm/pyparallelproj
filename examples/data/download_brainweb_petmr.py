import urllib.request
import zipfile
from pathlib import Path

if not Path('./brainweb_petmr').exists():
    if not Path('./brainweb_petmr.zip').exists():
        urllib.request.urlretrieve(
            'https://zenodo.org/record/4897350/files/brainweb_petmr.zip',
            'brainweb_petmr.zip')
    else:
        with zipfile.ZipFile('./brainweb_petmr.zip', 'r') as zip_ref:
            zip_ref.extractall('.')

    Path('./brainweb_petmr.zip')