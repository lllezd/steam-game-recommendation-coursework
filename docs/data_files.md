# Required Local Data Files

Large data files are not included in this GitHub repository because they are too large for a clean review-oriented repo.

For full local reproduction, place the source and generated data under the same relative paths used by the original project:

## Raw Steam Recommendation Data

- `data/raw/game_recommendations/recommendations.csv`
- `data/raw/game_recommendations/games.csv`
- `data/raw/game_recommendations/games_metadata.json`
- `data/raw/game_recommendations/users.csv`

## External Game Metadata

- `data/raw/artermiloff_games/games_march2025_cleaned.csv`
- `data/raw/artermiloff_games/games_march2025_full.csv`
- `data/raw/artermiloff_games/games_may2024_cleaned.csv`
- `data/raw/artermiloff_games/games_may2024_full.csv`
- `data/raw/fronkon_games/games.csv`
- `data/raw/fronkon_games/games.json`

## Intermediate Processed Files

The data preparation pipeline creates intermediate `.parquet` files such as:

- `data/processed/item_metadata.parquet`
- `data/processed/cleaned_interactions.parquet`
- `data/processed/user_histories_mvp_v4_temporal.parquet`

## Final Training Splits

The experiments use final temporal splits such as:

- `data/final/tabular_temporal/train_tabular.parquet`
- `data/final/tabular_temporal/val_tabular.parquet`
- `data/final/tabular_temporal/test_tabular.parquet`
- `data/final/instruction_temporal/train.jsonl`
- `data/final/instruction_temporal/val.jsonl`
- `data/final/instruction_temporal/test.jsonl`
- `data/final/instruction_temporal_retrieved_topk/train.jsonl`
- `data/final/instruction_temporal_retrieved_topk/val.jsonl`
- `data/final/instruction_temporal_retrieved_topk/test.jsonl`

These files are intentionally excluded from GitHub. Recreate them with the scripts in `scripts/data_preparation/` or place local copies in the paths above.