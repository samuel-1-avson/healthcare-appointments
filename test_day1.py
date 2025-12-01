# test_day1.py
from src.utils import load_config, setup_logging, print_pipeline_banner
from src.data_loader import DataLoader

# Setup
print_pipeline_banner()
config = load_config("config/config.yaml")
logger = setup_logging(config['logging']['level'])

# Test data loading
loader = DataLoader(config)
df = loader.load(source="auto")

print(f"\n✅ Successfully loaded {len(df):,} appointments!")
print(f"✅ Columns: {df.columns.tolist()}")
print(f"✅ Date range: {df['AppointmentDay'].min()} to {df['AppointmentDay'].max()}")