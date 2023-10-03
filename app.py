from flask import Flask, render_template
from slurm_commands import *
from apscheduler.schedulers.background import BackgroundScheduler
from cachetools import TTLCache
import datetime

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'  # Use a simple cache
cache = TTLCache(maxsize=1, ttl=3600)  # Cache data for 1 hour

def update_data():
    conn_manager = RemoteConnectionManager()
    info = aggregate_partition_info(conn_manager, locked_usage=True)
    user_info = aggregate_user_info(conn_manager)

    users_with_no_gpu_usage = filter_users_with_no_GPU_usage(user_info)
    users_with_partial_gpu_usage = filter_users_with_partial_GPU_usage(user_info)

    # Sort by gpu usage and keep the first 5
    user_info = dict(sorted(user_info.items(), key=lambda x: x[1]['gpu_allocated'], reverse=True)[:5])
    users_with_partial_gpu_usage = dict(sorted(users_with_partial_gpu_usage.items(), key=lambda x: x[1]['gpu_locked'],
                                               reverse=True)[:5])
    users_with_no_gpu_usage = dict(sorted(users_with_no_gpu_usage.items(), key=lambda x: x[1]['gpu_locked'],
                                          reverse=True)[:5])

    # Store the updated data in the cache
    cache['data'] = {
        'partitions': info,
        'users_info': user_info,
        'users_with_no_gpu_usage': users_with_no_gpu_usage,
        'users_with_partial_gpu_usage': users_with_partial_gpu_usage,
        'last_update_time': "{:%H:%M:%S%z}".format(datetime.datetime.now())
    }

# Schedule the data update based on the office hours schedule
scheduler = BackgroundScheduler()
# one job for 9-17 office hours every 30 minutes
scheduler.add_job(update_data, 'cron', hour='9-17', minute='*/15')
# for the rest of the day every 1 hour
scheduler.add_job(update_data, 'cron', hour='0-8,18-23', minute='0')
scheduler.start()

@app.route('/')
def index():
    # Check if data is in the cache
    data = cache.get('data')

    if data is None:
        # If data is not in the cache, fetch it and store it in the cache
        update_data()
        data = cache['data']

    return render_template('index.html', partitions=data['partitions'], users_info=data['users_info'],
                           users_with_no_gpu_usage=data['users_with_no_gpu_usage'],
                           users_with_partial_gpu_usage=data['users_with_partial_gpu_usage'],
                           last_update_time=data['last_update_time'])

if __name__ == '__main__':
    app.run(debug=True)
