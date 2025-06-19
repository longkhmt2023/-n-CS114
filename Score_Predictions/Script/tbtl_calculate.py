import pandas as pd

input_df = pd.read_csv('../Input - Output/output.csv')
input_df['tbtl'] = 0.2*input_df['qt'] + 0.4*input_df['th']+0.4*input_df['ck']
input_df.to_csv('../Input - Output/output.csv', index=False)