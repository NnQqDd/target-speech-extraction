import os
from pathlib import Path
import glob
from tqdm import tqdm
import pandas as pd

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LibriSpeech')

TRAIN_DIR = 'train-clean-100'
TEST_DIR = 'test-clean'
VAL_DIR = 'dev-clean'
 
all_df = []
for partition, src_dir in zip(['train', 'test', 'val'], [TRAIN_DIR, TEST_DIR, VAL_DIR]): 
	src_path = Path(os.path.join(BASE_PATH, src_dir))
	part_df = [] 
	print(f'Generate metadata for {src_path}...')
	for speaker_id in tqdm(os.listdir(os.path.join(src_path))):
		pattern = str(Path(os.path.join(src_path, speaker_id)) / '**' / '*.flac')
		
		audio_paths = []
		for audio_path in glob.glob(pattern, recursive=True):
			audio_paths.append(os.path.join(src_dir, audio_path.replace(str(src_path) + '/', '')))


		data = {
    		'speaker_id': [speaker_id]*len(audio_paths),
    		'filepath': audio_paths
		}
		part_df.append(pd.DataFrame(data))

	part_df = pd.concat(part_df, ignore_index=True)
	part_df['partition'] = partition

	all_df.append(part_df)

all_df = pd.concat(all_df, ignore_index=True)
all_df.to_csv(os.path.join(BASE_PATH, 'metadata.csv'), index=False)