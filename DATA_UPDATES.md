# BracketLab Data Updates

## Where to get each input

### Torvik
- Source: Bart Torvik team ratings / team results
- Current repo workflow: `python3 fetch_torvik.py`
- Output file: `torvik_ratings.csv`

### KenPom
- Source: your KenPom account export or copy/paste from team ratings page
- Paste into: `kenpom_ratings.csv`
- Required columns:

```csv
Team,AdjEM,Tempo
```

If you copy the full KenPom table as raw text first, save it to a file like `kenpom_raw.txt`, then run:

```bash
python3 import_ratings_table.py --source kenpom --input kenpom_raw.txt --output kenpom_ratings.csv
```

### EvanMiya
- Source: your EvanMiya account export or copy/paste from team ratings page
- Paste into: `evanmiya_ratings.csv`
- Required columns:

```csv
Team,AdjEM,Tempo
```

If you copy the full EvanMiya table as raw text first, save it to a file like `evanmiya_raw.txt`, then run:

```bash
python3 import_ratings_table.py --source evanmiya --input evanmiya_raw.txt --output evanmiya_ratings.csv
```

### Injuries
- Source: your manual injury tracking
- Update: `injury_adjustments.csv`
- Required columns:

```csv
Team,Player,AdjEM_delta,Status,Note
```

Guideline:
- `-1.0` to `-1.5`: rotation player
- `-2.0` to `-2.5`: strong starter / secondary star
- `-3.0` to `-4.0`: major star / All-America-level absence

### Official bracket
- Update `bracket_round1.csv` with the real round-of-64 matchups
- Update `play_in.csv` with the real First Four

## Fast refresh workflow

After updating the CSV files:

```bash
cd /Users/owenrosen/Desktop/march_model22
python3 refresh_data.py
streamlit run app.py
```

If Torvik is already refreshed and you only want validation + rebuild:

```bash
python3 refresh_data.py --skip-torvik-fetch
```

## Selection Sunday checklist

1. Refresh Torvik
2. Paste fresh KenPom data
3. Paste fresh EvanMiya data
4. Update injuries
5. Replace `bracket_round1.csv`
6. Replace `play_in.csv`
7. Run `python3 refresh_data.py`
8. Launch `streamlit run app.py`
