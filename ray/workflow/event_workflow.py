import time
import ray
from ray import workflow

# Create an event which finishes after 2 seconds.
event1_task = workflow.wait_for_event(workflow.event_listener.TimerListener, time.time() + 5)

# Create another event which finishes after 1 seconds.
event2_task = workflow.wait_for_event(workflow.event_listener.TimerListener, time.time() + 5)

@ray.remote
def gather(*args):
    return args

# Gather will run after 2 seconds when both event1 and event2 are done.
workflow.run(gather.bind(event1_task, event2_task))