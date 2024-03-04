import pandas as pd
from ydata_profiling import ProfileReport
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")

report = ProfileReport(df, title='Financial Statement')
report.to_file("Financial_Statement_analysis.html")


