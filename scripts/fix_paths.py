import json

with open('data/items.json', 'r') as f:
    items = json.load(f)

for item in items:
    # Revert to local path
    if item['path'].startswith('http://'):
        filename = item['path'].split('/')[-1]
        item['path'] = f"data/images/{filename}"

with open('data/items.json', 'w') as f:
    json.dump(items, f, indent=2)

print('✅ Reverted to local paths!')
