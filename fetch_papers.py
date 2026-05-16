import requests
import json
import os
import time

# --- CONFIGURATION ---
QUERY = 'brain tumor classification AND (CNN OR "Vision Transformer")'
MAX_PAPERS = 50
SAVE_DIR = "medical_corpus"
API_KEY = "YOUR_NCBI_API_KEY" # Leave as None if you don't have one

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_pmcids(query, limit):
    print(f"Searching for papers: {query}...")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "api_key": API_KEY
    }
    response = requests.get(url, params=params).json()
    return response['esearchresult']['idlist']

def download_bioc_json(pmcid):
    # BioC API Endpoint for full-text JSON
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{pmcid}/unicode"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch PMC{pmcid}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading PMC{pmcid}: {e}")
        return None

# --- EXECUTION ---
pmc_ids = get_pmcids(QUERY, MAX_PAPERS)
print(f"Found {len(pmc_ids)} papers. Starting download...")

for i, pmcid in enumerate(pmc_ids):
    filename = os.path.join(SAVE_DIR, f"PMC{pmcid}.json")
    
    if os.path.exists(filename):
        continue
        
    data = download_bioc_json(pmcid)
    if data:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"[{i+1}/{len(pmc_ids)}] Saved PMC{pmcid}.json")
    
    # Respect rate limits (0.1s delay with API key, 0.4s without)
    time.sleep(0.1 if API_KEY else 0.4)

print(f"\nTask 01 Complete! Your corpus is ready in ./{SAVE_DIR}")