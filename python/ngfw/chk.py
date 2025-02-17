
features201 = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt'] 

features20 = ['dur', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sbytes', 'dbytes', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_src_ltm', 'ct_dst_sport_ltm','ct_srv_dst']

if set(features201) == set(features20):
    print ("True")
