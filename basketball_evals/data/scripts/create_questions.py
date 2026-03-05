import pandas as pd
import json
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(script_dir)

# Read the JSONL file
df = pd.read_json(os.path.join(data_dir, 'player-season-stats.jsonl'), lines=True)

all_questions = []
few_shot = []

# Effective Field Goal % (eFG%) - row 0 (few-shot) + row 0 (few-shot) + rows 25-49
efg_rows = list(df.iloc[[0]].iterrows()) + list(df.iloc[25:50].iterrows())
for idx, (_, row) in enumerate(efg_rows):
    player = row['Player']
    fg = int(row['FG'])
    three_p = int(row['3P'])
    fga = int(row['FGA'])
    efg = round((fg + 0.5 * three_p) / fga, 3)
    
    reasoning = f"eFG% = (FG + 0.5 * 3P) / FGA = ({fg} + 0.5 * {three_p}) / {fga} = ({fg} + {0.5 * three_p}) / {fga} = {fg + 0.5 * three_p} / {fga} = {efg}"
    
    question = {
        "input": f"{player} has made {fg} total field goals, {three_p} three-point field goals, and has attempted {fga} field goals. What is his effective field goal percentage (eFG%) as a decimal? Round to the nearest 3 decimal places.",
        "target": efg,
        "reasoning": reasoning
    }
    if idx == 0:
        few_shot.append(question)
    else:
        all_questions.append(question)

# True Shooting % (TS%) - row 1 (few-shot) + row 1 (few-shot) + rows 50-74
ts_rows = list(df.iloc[[1]].iterrows()) + list(df.iloc[50:75].iterrows())
for idx, (_, row) in enumerate(ts_rows):
    player = row['Player']
    pts = int(row['PTS'])
    fga = int(row['FGA'])
    fta = int(row['FTA'])
    ts = round(pts / (2 * (fga + 0.44 * fta)), 3)
    
    denominator_inner = fga + 0.44 * fta
    denominator = 2 * denominator_inner
    reasoning = f"TS% = PTS / (2 * (FGA + 0.44 * FTA)) = {pts} / (2 * ({fga} + 0.44 * {fta})) = {pts} / (2 * ({fga} + {round(0.44 * fta, 2)})) = {pts} / (2 * {round(denominator_inner, 2)}) = {pts} / {round(denominator, 2)} = {ts}"
    
    question = {
        "input": f"{player} has scored {pts} total points, attempted {fga} field goals, and attempted {fta} free throws. What is his true shooting percentage (TS%) as a decimal? Round to the nearest 3 decimal places.",
        "target": ts,
        "reasoning": reasoning
    }
    if idx == 0:
        few_shot.append(question)
    else:
        all_questions.append(question)

# 3-Point Attempt Rate (3PAr) - row 2 (few-shot) + row 2 (few-shot) + rows 75-99
three_par_rows = list(df.iloc[[2]].iterrows()) + list(df.iloc[75:100].iterrows())
for idx, (_, row) in enumerate(three_par_rows):
    player = row['Player']
    three_pa = int(row['3PA'])
    fga = int(row['FGA'])
    three_par = round(row['3PAr'], 3)
    
    reasoning = f"3PAr = 3PA / FGA = {three_pa} / {fga} = {three_par}"
    
    question = {
        "input": f"{player} has attempted {three_pa} three-point field goals and {fga} total field goals. What is his 3-point attempt rate (3PAr) as a decimal? Round to the nearest 3 decimal places.",
        "target": three_par,
        "reasoning": reasoning
    }
    if idx == 0:
        few_shot.append(question)
    else:
        all_questions.append(question)

# Free Throw Rate (FTr) - row 3 (few-shot) + row 3 (few-shot) + rows 100-124
ftr_rows = list(df.iloc[[3]].iterrows()) + list(df.iloc[100:125].iterrows())
for idx, (_, row) in enumerate(ftr_rows):
    player = row['Player']
    fta = int(row['FTA'])
    fga = int(row['FGA'])
    ftr = round(row['FTr'], 3)
    
    reasoning = f"FTr = FTA / FGA = {fta} / {fga} = {ftr}"
    
    question = {
        "input": f"{player} has attempted {fta} free throws and {fga} field goals. What is his free throw rate (FTr) as a decimal? Round to the nearest 3 decimal places.",
        "target": ftr,
        "reasoning": reasoning
    }
    if idx == 0:
        few_shot.append(question)
    else:
        all_questions.append(question)

# Save all questions to JSONL file
with open(os.path.join(data_dir, 'player-season-stats-questions.jsonl'), 'w') as f:
    for question in all_questions:
        f.write(json.dumps(question) + '\n')

# Save few-shot examples to JSONL file
with open(os.path.join(data_dir, 'player-season-stats-fewshot.jsonl'), 'w') as f:
    for question in few_shot:
        f.write(json.dumps(question) + '\n')

print(f"Successfully created {len(all_questions)} questions")
print(f"Successfully created {len(few_shot)} few-shot examples")
