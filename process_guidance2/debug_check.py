import json

# Check RCOphth AMD - too many sections
with open('json_g2/RCOphth_AMD_2024.json', encoding='utf-8') as f:
    doc = json.load(f)
print(f"RCOphth AMD - {len(doc['sections'])} sections. First 12:")
for s in doc['sections'][:12]:
    print(f"  [{s['section_number']:>3}] {s['section_title'][:65]:<65} | {len(s['content']):>5} chars")

print()

# Check EURETINA AMD - 1 section
with open('json_g2/EURETINA_AMD_2023.json', encoding='utf-8') as f:
    doc2 = json.load(f)
print(f"EURETINA AMD - {len(doc2['sections'])} sections")
print("First 500 chars of full text:")
print(doc2['sections'][0]['content'][:500])

print()

# Check survey ophth
with open('json_g2/REVIEW_SURV_OPHTH_2026.json', encoding='utf-8') as f:
    doc3 = json.load(f)
print(f"Survey Ophth - {len(doc3['sections'])} sections")
print("First 500 chars:")
print(doc3['sections'][0]['content'][:500])
