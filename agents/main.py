# Initialize converter
from agents.conversion_script import OCELConverter
import json

converter = OCELConverter(api_key="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", model="gpt-4")

# Convert log
ocel_log = converter.convert_log("fx_trade_log.csv")
# Save as JSON
with open("ocel_output.json", "w") as f:
    json.dump(ocel_log, f, indent=2)
