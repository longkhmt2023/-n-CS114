import pandas as pd

input_df = pd.read_csv('../Input - Output/Output.csv')
input_df1 = pd.read_csv('../Input - Output/Output1.csv')
input_df['tbtl'] = 0.2*input_df1['predicted_qt'] + 0.4*input_df['predicted_th']+0.4*input_df['predicted_ck']
input_df.to_csv('../Input - Output/Output2.csv', index=False)