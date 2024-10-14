# features_and_targets.py

features_and_targets = [
{'features' : ['acceleration', 'AccX', 'AccY', 'AccZ', 'acceleration_magnitude', 'gyroscope_magnitude', 
            'directional_distance_to_intersection', 'Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 
            'Reoriented_AccX', 'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'speed_kmh'},


{'features' : ['speed_kmh', 'AccX', 'AccY', 'AccZ',  'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection','Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'acceleration'},

{'features' : ['speed_kmh', 'acceleration', 'gyroscope_magnitude', 'AccX', 'AccY', 'AccZ', 
            'directional_distance_to_intersection', 'Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 
            'Reoriented_AccX', 'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'acceleration_magnitude'},


{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccY', 'AccZ',  'acceleration_magnitude', 'directional_distance_to_intersection','Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'gyroscope_magnitude' },


{'features' : ['speed_kmh', 'acceleration', 'AccY', 'AccZ','acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection','Pitch_Angle', 'Roll_Angle', 'Yaw_Angle',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' :  'Reoriented_AccX'},


{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccZ', 'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection','Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 'Reoriented_AccX', 'Reoriented_AccZ'],
'target' :  'Reoriented_AccY'},


{'features' : ['speed_kmh', 'acceleration', 'AccY', 'AccX', 'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection','Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 'Reoriented_AccX',
       'Reoriented_AccY'],
'target' :  'Reoriented_AccZ'},


{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccY', 'AccZ', 'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection', 'Roll_Angle', 'Yaw_Angle', 'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'Pitch_Angle' },


{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccY', 'AccZ',  'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection','Pitch_Angle',  'Yaw_Angle', 'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'Roll_Angle' },


{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccY', 'AccZ',  'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection','Pitch_Angle', 'Roll_Angle',  'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'Yaw_Angle' },


{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccY', 'AccZ', 'GyY', 'GyZ', 'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection', 'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'GyX' 
},

{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccY', 'AccZ', 'GyX', 'GyZ', 'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection', 'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'GyY' },


{'features' : ['speed_kmh', 'acceleration', 'AccX', 'AccY', 'AccZ', 'GyX', 'GyY', 'acceleration_magnitude','gyroscope_magnitude', 'directional_distance_to_intersection', 'Reoriented_AccX',
       'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'GyZ' },


{'features' : ['AccX', 'AccY', 'AccZ', 'acceleration_magnitude', 'gyroscope_magnitude', 'acceleration',
            'Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 'Reoriented_AccX', 'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'speed_kmh'},


{'features' : ['speed_kmh', 'AccX', 'AccY', 'AccZ', 'acceleration_magnitude', 'gyroscope_magnitude',
            'Pitch_Angle', 'Roll_Angle', 'Yaw_Angle', 'Reoriented_AccX', 'Reoriented_AccY', 'Reoriented_AccZ'],
'target' : 'acceleration'}

]