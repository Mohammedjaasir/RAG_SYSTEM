import json

# Load the extraction results
with open('social_kitchen_result.json', 'r') as f:
    data = json.load(f)

# Display formatted results
print('\n' + '='*70)
print('SOCIAL KITCHEN RECEIPT EXTRACTION RESULTS')
print('='*70)

print(f"\nSupplier: {data.get('supplier_name')}")
print(f"Address: {data.get('address')}")
print(f"Receipt #: {data.get('receipt_number')}")
print(f"GST/VAT #: {data.get('vat_number')}")
print(f"Date: {data.get('date')}")

print('\nFINANCIAL SUMMARY:')
print(f"  Total Amount: Rs. {data.get('total_amount')}")
print(f"  Net Amount: Rs. {data.get('net_amount')}")
print(f"  VAT/Tax: Rs. {data.get('vat_amount')}")

print('\nITEMS EXTRACTED:')
for i, item in enumerate(data.get('items', []), 1):
    print(f"  {i}. {item['name']}")
    print(f"     Qty: {item['quantity']} x Rs.{item['unit_price']} = Rs.{item['total_price']}")

print('\nCONFIDENCE ANALYSIS:')
print(f"  Overall Confidence: {data['_metadata']['overall_confidence']:.2%}")
print(f"  Relevance Scores: {data['_metadata']['relevance_scores']}")

warnings = data.get('_low_confidence_warnings', [])
print(f'\nLOW CONFIDENCE WARNINGS: {len(warnings)}')
for w in warnings:
    print(f"  - {w['field']}: {w['score']:.0%} ({w['action']})")

print('\n' + '='*70)
