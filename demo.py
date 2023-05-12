from scoring.average_precision import calculate_average_precision
one =  [
          {'file_id': 'M010015BY', 'timestamp': 1160.2, 'type': 'audio', 'impact_scalar': 4},
          {'file_id': 'M010015BY', 'timestamp': 1287.6, 'type': 'audio', 'impact_scalar': 2},
          {'file_id': 'M010029SP', 'timestamp': 288.0, 'type': 'text', 'impact_scalar': 1,'hello':9},
          {'file_id': 'M010005QD', 'timestamp': 90.2, 'type': 'video', 'impact_scalar': 5},
          {'file_id': 'M010019QD', 'timestamp': 90, 'type': 'text', 'impact_scalar': 5}
      ]
two = [
          {'file_id': 'M010015BY', 'timestamp': 1160.2, 'type': 'audio', 'llr': 1.5},
          {'file_id': 'M010015BY', 'timestamp': 1287.67, 'type': 'audio', 'llr': 1.5},
          {'file_id': 'M010029SP', 'timestamp': 278, 'type': 'text', 'llr': 1.5},
          {'file_id': 'M010005QD', 'timestamp': 90.2, 'llr': 1.5, 'type': 'video'},
          {'file_id': 'M010019QD', 'timestamp': 190, 'llr': 1.5, 'type': 'text'}
      ]
a = calculate_average_precision(one,two)
print(a)
