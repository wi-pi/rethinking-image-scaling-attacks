import sys
from pathlib import Path
path = Path(__file__).parent.parent / 'ext/scalingattack/scaleatt/'
sys.path.insert(1, str(path))
