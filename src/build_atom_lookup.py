import os
import json

class BuildAtoms():
   
   def build_lookup_table(self) -> None:
        assert 'construct/lookup_map.json' in os.listdir('/dataset/')
        with open('/dataset/construct/lookup_map.json') as data:
            d = json.loads(data)
        
        print(d)
        return


if __name__ == '__main__':
    build = BuildAtoms()
    build.build_lookup_table()