import shutil
import os
import argparse
import yaml
import soundfile as sf
import torch

from modules.ddp_dataset import get_dataloaders


parser = argparse.ArgumentParser(description='OMG!')
parser.add_argument('-c', '--config')
parser.add_argument('-n', '--numbers', type=int)
args = parser.parse_args()
print(args.numbers)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
    print(config['dataset'])

dataloaders = get_dataloaders(config, distributed=False)

script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    shutil.rmtree(os.path.join(script_dir, 'samples'))
except Exception as e:
    print(e)
os.makedirs(os.path.join(script_dir, 'samples'), exist_ok=True)

cnt = 0
wave_len = config['dataset']['wave_len']
for mixtures, srcs in dataloaders['val']:
    for i in range(len(mixtures)):
        cnt += 1
        sf.write(f'samples/mixture-{cnt}.wav', mixtures[i][:wave_len].cpu().numpy(), config['dataset']['sample_rate'])
        sf.write(f'samples/reference-{cnt}.wav', (mixtures[i][wave_len:] - 1).cpu().numpy(), config['dataset']['sample_rate'])
        sf.write(f'samples/target-{cnt}.wav', srcs[0][i][:wave_len].cpu().numpy(), config['dataset']['sample_rate'])
        sf.write(f'samples/reference-{cnt}-0.wav', (srcs[0][i][wave_len:] - 1).cpu().numpy(), config['dataset']['sample_rate'])
        assert torch.allclose(mixtures[i][wave_len:], srcs[0][i][wave_len:])
        for j in range(1, len(srcs)):
            sf.write(f'samples/interference-{cnt}-{j}.wav', srcs[j][i][:wave_len].cpu().numpy(), config['dataset']['sample_rate'])
        print(cnt, end=' ')
        if cnt == args.numbers:
            break
    if cnt == args.numbers:
        break
print()