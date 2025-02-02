import json
import sys
import os
import re

if __name__=='__main__':

    notes = f'''usage: {sys.argv[0]} configfile

'''

    if len( sys.argv ) < 2:
        print(notes)
        exit(-1)
    
    config_file = sys.argv[1]
    if not os.path.exists(config_file) :
        print(f'{config_file} does not exist\n')
        exit(-1)
    
    config_file_lines = open(config_file, 'r').readlines();
    rx_nc = re.compile('^\s*#')
    config = json.loads(''.join([s for s in config_file_lines if not rx_nc.match(s)]))
