import time
import threading
import asyncio

class BestFriendAutomation:
    def __init__(self, name="BF Automation"):
        self.name = name
        self.tasks = []
        self.running = True

    def schedule_task(self, task_description, start_time, duration):
        task = {
            "description": task_description,
            "start_time": start_time,
            "duration": duration,
            "completed": False
        }
        self.tasks.append(task)
        print(f"Task '{task_description}' scheduled for {start_time}.")

    async def run_scheduled_tasks(self):
        while self.running and self.tasks:
            current_time = time.time()
            for task in self.tasks:
                if not task["completed"] and current_time >= task["start_time"]:
                    print(f"Starting task: {task['description']}")
                    await asyncio.sleep(task["duration"])
                    task["completed"] = True
                    print(f"Task '{task['description']}' completed.")
            await asyncio.sleep(1)  # Check the task list every second
        print(f"{self.name} has completed all scheduled tasks.")

    def stop_execution(self):
        self.running = False
        print(f"{self.name} has been stopped.")

    def auto_optimize(self):
        # Add logic to adjust and optimize task execution here
        pass

# Example usage
bfa = BestFriendAutomation("CharlieBot Automation")
bfa.schedule_task("Start Data Analysis", time.time() + 5, 10)
bfa.schedule_task("Send Report", time.time() + 20, 5)

# Run the tasks asynchronously
asyncio.run(bfa.run_scheduled_tasks())

# The loop can be stopped by calling bfa.stop_execution()
