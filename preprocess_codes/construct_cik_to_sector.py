import json

def cik_to_sector(jsonl_filepath, json_filepath):
    cik_to_sector = {}

    with open(jsonl_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            cik = data['CIK']
            sector = data['Sector']
            cik_to_sector[str(cik)] = sector

    with open(json_filepath, 'w') as f:
        json.dump(cik_to_sector, f, indent=4)

if __name__ == '__main__':
    cik_to_sector('spx_list.jsonl', 'cik_to_sector.json')