from src.scrapping.scrapper import fetch_job_listings

url = "https://nofluffjobs.com/pl/jobs?criteria=python"  # exemplary filter
df = fetch_job_listings(url)
print(df.head())
df.to_csv("data/raw/jobs.csv", index=False)
