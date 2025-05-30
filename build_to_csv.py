# This file converts all the features imported from an API to csv files
#
#
# @author: mjNotFound-19
# @modified: 5/30/25

""" WARNING: If you don't change the import to target the specific API you might face some problems in the long run with your data being all over the place."""
from fast_f1 import build
import fastf1

year = 2023

fastf1.Cache.enable_cache('cache/')
schedule = fastf1.get_event_schedule(year)
num_rounds = schedule.shape[0]

races = ["Bahrain", "Jeddah", "Melbourne", "Baku", "Miami", "Monaco", "Barcelona", "Montreal", "Spielberg", "Silverstone", "Budapest", "Spa", "Zandvoort", "Monza", "Singapore", "Suzuka", "Lusail", "Austin", "Mexico City", "Sao Paulo", "Las Vegas", "Abu Dhabi"]


source = "fastf1"
rounds = list(range(1, num_rounds))

for round_num in rounds:
	print(f"processing round {round}")
	df = build(year, round_number = round_num)
	if df.empty:
		print(f"skipped round {round_num} due to lack of data")
	else:
		df.to_csv(f"data/processed/{source}_round{round_num}_{year}_features.csv", index = False)
		print(f"Saved ROUND {round} Successfully")
