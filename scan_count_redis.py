import redis
import time

redis_cache = redis.Redis(host="ai-online.redis.rds.inagora.org", port=6379, password="6JGowG87sKPA!", db=0)

print("begin")
print(redis_cache.dbsize())
t=365*24*3600
begin_pos = 0
all_count=0
while True:
    result = redis_cache.scan(begin_pos,match="uv:*",count=10000)
    return_pos,datalist = result
    # print(datalist)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), return_pos)
    if len(datalist) > 0:
        all_count = all_count + len(datalist)

    if return_pos == 0:
        break
    if len(datalist)==0:
        break
    begin_pos = return_pos

print("over,count:"+str(all_count))