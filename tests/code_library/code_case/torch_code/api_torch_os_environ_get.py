
import os

print("#########################case1#########################")
os.environ.get("WORLD_SIZE",1)
print("#########################case2#########################")
os.environ.get('LOCAL_RANK',1)
print("#########################case3#########################")
os.environ.get('RANK',1)
