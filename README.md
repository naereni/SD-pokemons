# SD-pokemons
Finetune SD-1.5 to gen new pokemons

## Install
### Venv
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Weights
Веса для LoRA адаптера лежат по [ссылке](), файл с весами переложить в `src/LoRA-pokemons-weights/`

### Data
Все инструкции находятся в папке `data`

## Usage

## Model
По сути просто дефолтная **SD-1.5** с **LoRA** адаптером, который натренерован на датасете с покемонами
Как примет качественного рефакторинга **blob** кода во чтото читаемое

[Отчет с Wandb](https://wandb.ai/team24/text2image-fine-tune/reports/LoRA-tune-pokemons--Vmlldzo2NzM0Mzk5)