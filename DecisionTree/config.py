from pathlib import Path
import gc

gc.enable()

PATH = Path(r'tiffs')
THRESHOLD = 0.5
TRACK_CLIP_LENGTH = 5
VIEW_TRACKS = True
#REFERENCE_DIR = Path(r'training_data/LiveCell_CTC_format_2D/01')