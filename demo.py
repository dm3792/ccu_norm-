from scoring.average_precision import calculate_average_precision

a = calculate_average_precision([{'timestamp': 287, 'impact_scalar': 5, 'comment': 'Pre-change: Host talked about patterns of dirty women in Beijing, with female guest listening calmly. Shift: Host turned to female guest to have her more engaged, and female guest excitedly shared her story. Evidence: Host raised his voice, in addition to using gesture. Host made a joke about the female guest and she laughed and started talking fast.', 'annotator': 212, 'file_id': 'M01000FT6', 'type': 'video'}],[{'file_id': 'M0100053I', 'type': 'audio', 'timestamp': 120.88, 'llr': -5.325099468231201}])
print(a)
